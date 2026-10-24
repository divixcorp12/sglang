// One job at a time through a JobWindow: the single-job path of a kernel's job interface (binding.hpp's
// CpuExpertJobKernel), for tests, the benches and other harnesses that hold no engine.
//
// JobRunner<Kernel> owns a one-slot window over a Team and keeps its workers on the scheduler loop for its lifetime, so a
// run() pays no thread start and the slot's scratch stays warm. forward_alone is a run on a Team made for the call alone.
#pragma once
#include "binding.hpp"
#include "job_window.hpp"
#include "team.hpp"
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace sglang::cpu_experts {

/// \brief Runs a kernel's jobs one at a time on `team`, the caller as worker 0. The team serves the runner until it is
/// destroyed: nothing else serves it in between, and the runner dies before the team.
template <CpuExpertJobKernel Kernel>
class JobRunner {
public:
    using Layer = typename Kernel::Layer;

    explicit JobRunner(Team& team) : window_(team, GateStats{&gate_}) { window_.start(); }
    ~JobRunner()
    {
        window_.close();
        window_.drain();
        window_.join();
    }
    JobRunner(const JobRunner&) = delete;
    JobRunner& operator=(const JobRunner&) = delete;

    /// \brief Admits `call` and runs it to completion. A staged call's gate is waited on by the scheduler and its
    /// stats land in call.stage_two (calls, spun, called_ns, waited_ns). Throws what Kernel::prepare throws, with `out`
    /// untouched; std::invalid_argument when the job reports status 2, std::runtime_error for another status.
    void run(const Layer& layer, const ForwardCall& call)
    {
        gate_ = call.stage_two;
        if (!window_.admit(++seq_, 0, layer, call, kNoOutKey, call.stage_two))
            throw std::logic_error("CPU experts: the single-job window refused a job");
        for (;;) {
            const uint32_t seen = window_.word();
            const WindowStep r = window_.step(0);
            if (r == WindowStep::kRan) continue;
            // After `seen`: a job another worker completed before it moved the word already, and would never wake us.
            if (!window_.any_active()) break;
            window_.idle_wait(seen, r == WindowStep::kGated, 0);
        }
        const int status = window_.slot(0).status();
        if (status == 2) throw std::invalid_argument("CPU experts: the job refused its input (status 2)");
        if (status != 0)
            throw std::runtime_error("CPU experts: the job failed (status " + std::to_string(status) + ")");
    }

private:
    // Copies the scheduler's gate stats for the job into the caller's StageGate when the job completes.
    struct GateStats {
        StageGate* const* gate;
        template <class S>
        void on_complete(S& s) noexcept
        {
            StageGate* g = *gate;
            if (g == nullptr) return;
            g->calls += s.gate_calls;
            g->spun = g->spun || s.gate_spun;
            g->called_ns = s.gate_called_ns;
            g->waited_ns = s.gate_waited_ns;
        }
    };

    StageGate* gate_ = nullptr;
    JobWindow<Kernel, 1, GateStats> window_;
    uint32_t seq_ = 0;
};

/// \brief A forward of `Kernel` (a format's kernel type) on a team made for this call alone, `threads` workers with
/// worker i on cores[i] (empty: unpinned): tests and other harnesses that hold no Team. The call runs as one job of a
/// JobRunner; the engine runs on its own.
template <CpuExpertJobKernel Kernel>
void forward_alone(const typename Kernel::Layer& layer, const ForwardCall& call, std::span<const int> cores, int threads)
{
    Team team(cores, threads, &Kernel::warm, 0);
    JobRunner<Kernel> runner(team);
    runner.run(layer, call);
}

}  // namespace sglang::cpu_experts
