// CPU-only diagnostic ABI for the Python driver beside this file. Runs the actual engine and quant kernel.
#include "moe/expert_stream/host/cpu_experts.h"
#include <barrier>
#include <cstdio>
#include <exception>

namespace es = sglang::expert_stream;
namespace ce = sglang::cpu_experts;

// Fixed DSV4.1/3-bit synthetic fixture. Pointers are process-local and owned by the caller throughout this call.
extern "C" int engine_measure(uint64_t kernel_address, const uint64_t* slab_ptrs, const uint64_t* slab_bytes,
                              const uint64_t* inputs, const uint64_t* outputs, int rows, int routes, int reps,
                              int gap_us, int warm_us, int draft_group0,
                              const uint64_t* runtime_counters, int64_t* records, char* error, size_t error_bytes) {
    try {
        constexpr int hidden = 5120, intermediate = 2304, capacity = 12, workers = 10, columns = 8;
        if (rows < 1 || rows > 6 || routes < 1 || routes > es::wire::Wire::kLanes || reps < 1 || reps > 200 ||
            gap_us < 0 || gap_us > 200000 || warm_us < 0)
            throw std::invalid_argument("measurement arguments out of bounds");
        auto* kernel = reinterpret_cast<const ce::CpuExpertKernel*>(kernel_address);
        if (!kernel) throw std::invalid_argument("missing kernel");
        std::barrier rendezvous(2);
        std::exception_ptr errors[2];
        std::thread runners[2];
        for (int group = 0; group < 2; ++group) runners[group] = std::thread([&, group] {
            bool participating = true;
            try {
                // Producers stay off their expert team's CPUs; all experiment threads stay within 0..63.
                cpu_set_t producer; CPU_ZERO(&producer); CPU_SET(group ? 16 : 0, &producer);
                if (sched_setaffinity(0, sizeof(producer), &producer)) throw std::runtime_error("producer pin failed");
                ce::ExpertLayer shape;
                shape.capacity = capacity; shape.hidden = hidden; shape.intermediate = intermediate;
                shape.activation = 0; shape.act_limit = 10; shape.slab_count = 6;
                for (int s = 0; s < 6; ++s) {
                    shape.slabs[s] = reinterpret_cast<const void*>(slab_ptrs[group * 6 + s]);
                    shape.slot_bytes[s] = slab_bytes[group * 6 + s];
                }
                const int32_t params[2] = {3, 0};
                const auto layer = kernel->make_layer(shape, {reinterpret_cast<const std::byte*>(params), sizeof(params)});
                std::vector<int> cores;
                for (int w = 0; w < workers; ++w) cores.push_back((group ? 18 : 6) + w);
                std::vector<int32_t> slots(rows * routes);
                std::vector<float> weights(rows * routes, 1.0f);
                for (int t = 0; t < rows; ++t) for (int k = 0; k < routes; ++k) slots[t * routes + k] = k;
                // Native reference uses the same kernel arithmetic outside all measured engine jobs.
                std::vector<float> reference(rows * hidden);
                ce::ForwardCall call{rows, routes, workers, reinterpret_cast<const void*>(inputs[group]),
                                     slots.data(), weights.data(), reference.data(), false, cores};
                kernel->check(layer, call);
                std::exception_ptr reference_error;
                std::thread reference_thread([&] {
                    try { kernel->forward(layer, call); } catch (...) { reference_error = std::current_exception(); }
                });
                reference_thread.join();
                if (reference_error) std::rethrow_exception(reference_error);
                // First touch remains on the expert node, independently of the runtime's initial affinity.
                cpu_set_t node; CPU_ZERO(&node); CPU_SET(cores[0], &node);
                if (sched_setaffinity(0, sizeof(node), &node)) throw std::runtime_error("allocation pin failed");
                const int64_t input_bytes = 2 * rows * hidden;
                std::vector<uint8_t> staged(input_bytes + es::CpuTokenTable::kHeaderBytes +
                                             4 * es::wire::Wire::kLanes * (1 + rows));
                std::memcpy(staged.data(), reinterpret_cast<const void*>(inputs[group]), input_bytes);
                es::CpuExpertLayers layers(1);
                if (!layers.set(0, layer)) throw std::runtime_error("layer install failed");
                es::CpuExpertConfig config;
                config.kernel = kernel; config.layers = &layers; config.x_base = staged.data();
                config.x_stride = staged.size(); config.x_token_bytes = 2 * hidden; config.tokens = rows;
                config.out_base = reinterpret_cast<uint8_t*>(outputs[group]);
                config.out_stride = 4 * rows * hidden; config.hidden = hidden;
                config.threads = workers; config.cores = cores;
                config.keep_warm_ns = int64_t(warm_us) * 1000;
                config.check_calls = false;
                es::BasicCpuExpertEngine<es::InstrBuild> engine(config, "engine-measure: ", "measure-g" + std::to_string(group));
                engine.write_calibration_table(0, rows);
                alignas(64) std::array<uint8_t, es::draft::kChannelBytes> channel{};
                engine.start();
                if (group == 0 && draft_group0) {
                    auto source = std::make_unique<es::DraftSource>();
                    source->channel = channel.data(); source->x = staged.data(); source->slots = slots.data();
                    source->weights = weights.data(); source->out = reinterpret_cast<float*>(outputs[group]);
                    source->hidden = hidden; source->layers = {layer};
                    engine.attach_draft(std::move(source), 5'000'000'000);
                }
                CPU_ZERO(&producer); CPU_SET(group ? 16 : 0, &producer);
                if (sched_setaffinity(0, sizeof(producer), &producer)) throw std::runtime_error("producer repin failed");
                for (int rep = 0; rep < reps; ++rep) {
                    // Both native producers rendezvous before each paired target submission.
                    if (gap_us) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
                    rendezvous.arrive_and_wait();
                    es::CpuJob job;
                    job.row = 0; job.k = routes; job.seq = engine.claim(1); job.per_token = rows > 1;
                    for (int k = 0; k < routes; ++k) { job.slots[k] = k; job.weights[k] = 1; job.lanes[k] = k; }
                    auto* rec = records + (group * reps + rep) * columns;
                    for (int c = 0; c < 4; ++c)
                        rec[4 + c] = __atomic_load_n(reinterpret_cast<const uint64_t*>(runtime_counters[c]), __ATOMIC_RELAXED);
                    rec[0] = es::now_ns();
                    if (!engine.submit(job)) throw std::runtime_error("job ring full");
                    rec[1] = es::now_ns();
                    const int64_t deadline = rec[0] + 5'000'000'000;
                    while (!engine.done(job.seq)) {
                        _mm_pause();
                        if (es::now_ns() > deadline) throw std::runtime_error("engine job exceeded five seconds");
                    }
                    rec[2] = es::now_ns(); rec[3] = job.seq;
                    if (std::memcmp(reference.data(), reinterpret_cast<const void*>(outputs[group]), reference.size() * sizeof(float)))
                        throw std::runtime_error("engine output differs from direct forward");
                }
                rendezvous.arrive_and_drop(); participating = false;
                engine.stop();
            } catch (...) {
                errors[group] = std::current_exception();
                if (participating) rendezvous.arrive_and_drop();
            }
        });
        for (auto& runner : runners) runner.join();
        for (const auto& failure : errors) if (failure) std::rethrow_exception(failure);
        return 0;
    } catch (const std::exception& e) {
        std::snprintf(error, error_bytes, "%s", e.what());
        return 1;
    }
}
