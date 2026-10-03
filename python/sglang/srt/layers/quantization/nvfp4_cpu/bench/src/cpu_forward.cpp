// Bare CpuExpertForward timing; no Python, CUDA or service transport.
#include "cpu_experts_cabi.h"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sched.h>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
constexpr uint64_t max_bytes = uint64_t(2) << 30;
size_t pad(size_t x, size_t n) { return (x+n-1)/n*n; }
int number(const std::string& s) {
    size_t end = 0; int n = std::stoi(s, &end);
    if (end != s.size()) throw std::runtime_error("Invalid integer: " + s);
    return n;
}
std::vector<int32_t> list(const std::string& s, bool ranges) {
    std::vector<int32_t> result; std::stringstream stream(s); std::string part;
    while (std::getline(stream, part, ',')) {
        const auto dash = ranges ? part.find('-') : std::string::npos;
        int first = number(part.substr(0,dash));
        int last = dash == std::string::npos ? first : number(part.substr(dash+1));
        if (first < 0 || last < first || last >= CPU_SETSIZE) throw std::runtime_error("Invalid CPU/count range");
        for (int n = first; n <= last; ++n) {
            if (std::find(result.begin(),result.end(),n) != result.end()) throw std::runtime_error("Duplicate CPU/count");
            result.push_back(n);
        }
    }
    if (result.empty() || s.back() == ',') throw std::runtime_error("Empty CPU/count list");
    return result;
}
struct Options {
    std::string cpus = "18-33", experts = "1,3,5", fixture, write_fixture;
    int workers = 16, node = 1, warmup = 128, gap_us = 0;
    int hidden = 5120, intermediate = 2304, capacity = 5, layers = 1, layout = 0;
    int seed = 20261002;
    bool validate_only = false;
};
Options parse(int& argc, char** argv) {
    Options o; int keep = 1;
    for (int i=1; i<argc; ++i) {
        std::string arg(argv[i]); const auto eq = arg.find('=');
        std::string key = arg.substr(0,eq), value = eq == std::string::npos ? "" : arg.substr(eq+1);
        if ((key == "--fixture" || key == "--write-fixture") && value.empty())
            throw std::runtime_error("Fixture path must not be empty");
        if (key == "--cpus") o.cpus=value;
        else if (key == "--experts") o.experts=value;
        else if (key == "--fixture") o.fixture=value;
        else if (key == "--write-fixture") o.write_fixture=value;
        else if (key == "--workers") o.workers=number(value);
        else if (key == "--numa-node") o.node=number(value);
        else if (key == "--warmup-forwards") o.warmup=number(value);
        else if (key == "--gap-us") o.gap_us=number(value);
        else if (key == "--hidden") o.hidden=number(value);
        else if (key == "--intermediate") o.intermediate=number(value);
        else if (key == "--capacity") o.capacity=number(value);
        else if (key == "--layers") o.layers=number(value);
        else if (key == "--w13-layout") o.layout=number(value);
        else if (key == "--seed") o.seed=number(value);
        else if (arg == "--validate-only") o.validate_only=true;
        else if (arg == "--help") {
            std::cout << "NVFP4 CPU forward (" NVFP4_BENCH_BACKEND ")\n"
                "--cpus=18-33 --workers=16 --numa-node=1 --experts=1,3,5\n"
                "--fixture=FILE or synthetic --hidden=5120 --intermediate=2304\n"
                "--capacity=5 --layers=1 --w13-layout=0|1|2 --seed=20261002\n"
                "--write-fixture=NEW_FILE --validate-only --warmup-forwards=128 --gap-us=0\n"
                "Memory policy is inherited; Google Benchmark flags also accepted.\n";
            std::exit(0);
        } else argv[keep++]=argv[i];
    }
    argc=keep; argv[keep]=nullptr;
    if (o.workers < 1 || o.workers > CPU_SETSIZE || o.node < 0 || o.warmup < 0 || o.gap_us < 0)
        throw std::runtime_error("Invalid workers/node/warmup/gap");
    if (!o.fixture.empty() && !o.write_fixture.empty()) throw std::runtime_error("Cannot read and write fixture together");
    return o;
}
struct Header {
    uint32_t hidden, intermediate, capacity, layers, layout, separate_up;
    float limit, inv13, inv2;
};
std::array<uint64_t,7> sizes(const Header& h) {
    if (h.hidden < 16 || h.intermediate < 16 || h.hidden > (1<<20) || h.intermediate > (1<<20)
        || h.hidden%16 || h.intermediate%16 || h.capacity < 1 || h.capacity > 8
        || h.layers < 1 || h.layers > 256 || h.layout > 2 || h.separate_up > 1
        || (h.layout==2 && h.intermediate%64) || !std::isfinite(h.limit) || h.limit<0
        || !std::isfinite(h.inv13) || h.inv13<=0 || !std::isfinite(h.inv2) || h.inv2<=0)
        throw std::runtime_error("Invalid fixture dimensions/layout/scales");
    std::array<uint64_t,7> s{uint64_t(h.hidden)*h.intermediate, uint64_t(h.hidden)*h.intermediate/2,
        pad(2*h.intermediate,128)*pad(h.hidden/16,4), pad(h.hidden,128)*pad(h.intermediate/16,4),
        4,4,h.separate_up?4u:0u};
    uint64_t total=uint64_t(h.hidden)*2;
    for (auto n:s) total+=n*h.capacity;
    if (total*h.layers > max_bytes) throw std::runtime_error("Fixture exceeds 2 GiB safety limit");
    return s;
}
struct Layer {
    std::array<std::vector<uint8_t>,7> slabs;
    std::vector<uint16_t> x;
    int64_t handle=-1;
    ~Layer() { if (handle>=0) sglang_nvfp4_cpu_experts_free_layer(handle); }
};
struct Fixture {
    Header h;
    std::array<uint64_t,7> strides;
    std::vector<std::unique_ptr<Layer>> layers;
    // Input format is little-endian, never a serialized C struct/pointer.
    static void read(std::istream& in, void* p, size_t bytes) {
        if (!in.read(static_cast<char*>(p), bytes)) throw std::runtime_error("Truncated fixture");
    }
    explicit Fixture(const Options& o) {
        uint32_t endian=1;
        if (*reinterpret_cast<uint8_t*>(&endian)!=1) throw std::runtime_error("Little-endian host required");
        std::ifstream in;
        if (!o.fixture.empty()) {
            in.open(o.fixture,std::ios::binary); char magic[8]; read(in,magic,8);
            if (std::memcmp(magic,"NVF4B001",8)) throw std::runtime_error("Wrong fixture magic/version");
            read(in,&h.hidden,4); read(in,&h.intermediate,4); read(in,&h.capacity,4);
            read(in,&h.layers,4); read(in,&h.layout,4); read(in,&h.separate_up,4);
            read(in,&h.limit,4); read(in,&h.inv13,4); read(in,&h.inv2,4);
        } else h={uint32_t(o.hidden),uint32_t(o.intermediate),uint32_t(o.capacity),uint32_t(o.layers),
                  uint32_t(o.layout),0,0,1,1};
        strides=sizes(h); std::mt19937 rng(o.seed);
        for (uint32_t l=0;l<h.layers;++l) {
            auto layer=std::make_unique<Layer>();
            for (int i=0;i<7;++i) {
                auto& slab=layer->slabs[i]; slab.resize(strides[i]*h.capacity);
                if (!slab.empty() && in.is_open()) read(in,slab.data(),slab.size());
                else if (i<2) for (auto& b:slab) b=uint8_t(rng());
                else if (i<4) for (auto& b:slab) b=uint8_t(16+rng()%16);
                else for (uint32_t e=0;e<h.capacity && !slab.empty();++e) {
                    float alpha=.5f+float(rng()%1024)/1024;
                    std::memcpy(slab.data()+e*4,&alpha,4);
                }
            }
            layer->x.resize(h.hidden);
            if (in.is_open()) read(in,layer->x.data(),h.hidden*2);
            else for (auto& v:layer->x) v=uint16_t((rng()&1 ? 0x8000:0) | 0x2c00 | (rng()%1024));
            layers.push_back(std::move(layer));
        }
        if (in.is_open() && in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Trailing fixture bytes");
        // Validate complete payloads before registering any raw-pointer views.
        for (const auto& l:layers) {
            for (int i=2;i<4;++i) for (uint8_t b:l->slabs[i])
                if ((b&127)==127) throw std::runtime_error("NaN block scale in fixture");
            for (int i=4;i<7;++i) for (size_t p=0;p<l->slabs[i].size();p+=4) {
                float a; std::memcpy(&a,l->slabs[i].data()+p,4);
                if (!std::isfinite(a)) throw std::runtime_error("Nonfinite alpha");
            }
            for (auto x:l->x) if ((x&0x7c00)==0x7c00) throw std::runtime_error("Nonfinite activation");
        }
    }
    void register_layers() {
        for (auto& layer:layers) {
            SglangNvfp4CpuLayer d{};
            d.abi_version=1; d.capacity=h.capacity; d.hidden=h.hidden; d.intermediate=h.intermediate;
            d.w13_layout=h.layout; d.act_limit=h.limit; d.inv_input_scale13=h.inv13; d.inv_input_scale2=h.inv2;
            for (int i=0;i<7;++i) {
                d.slabs[i]=layer->slabs[i].empty()?nullptr:layer->slabs[i].data(); d.slot_bytes[i]=strides[i];
            }
            if (sglang_nvfp4_cpu_experts_register_slabs(&d,&layer->handle)) throw std::runtime_error("Registration failed");
        }
    }
    void write(const std::string& path) const {
        int fd=open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0644);
        if (fd<0) throw std::runtime_error("Cannot create new fixture (already exists?)");
        auto put=[fd](const void* ptr,size_t bytes) {
            auto p=static_cast<const uint8_t*>(ptr);
            while (bytes) { auto n=::write(fd,p,bytes); if (n<=0) throw std::runtime_error("Fixture write failed"); p+=n; bytes-=n; }
        };
        try {
            put("NVF4B001",8);
            put(&h.hidden,4); put(&h.intermediate,4); put(&h.capacity,4); put(&h.layers,4);
            put(&h.layout,4); put(&h.separate_up,4); put(&h.limit,4); put(&h.inv13,4); put(&h.inv2,4);
            for (const auto& layer:layers) {
                for (const auto& slab:layer->slabs) if (!slab.empty()) put(slab.data(),slab.size());
                put(layer->x.data(),h.hidden*2);
            }
            if (close(fd)) throw std::runtime_error("Fixture close failed");
        } catch (...) { close(fd); throw; }
    }
};

// Double-precision decoded-weight reference, outside timing. Recover scales
// by iterating the production permute's output axes into logical row/group
// order, rather than using the native kernel's address helper.
double decode_half(uint16_t h) {
    int exponent=(h>>10)&31, mantissa=h&1023;
    double v=exponent?std::ldexp(1.0+mantissa/1024.0,exponent-15):std::ldexp(double(mantissa),-24);
    return h&32768?-v:v;
}
double decode_scale(uint8_t b) {
    int e=(b>>3)&15, m=b&7;
    double v=e?std::ldexp(1.0+m/8.0,e-7):std::ldexp(double(m),-9);
    return b&128?-v:v;
}
std::vector<double> linear_scales(const uint8_t* p,int rows,int groups) {
    std::vector<double> out(size_t(rows)*groups); size_t offset=0;
    for (int rt=0;rt<int(pad(rows,128))/128;++rt)
      for (int kt=0;kt<int(pad(groups,4))/4;++kt)
        for (int lane=0;lane<32;++lane)
          for (int rg=0;rg<4;++rg)
            for (int ki=0;ki<4;++ki,++offset) {
                int r=rt*128+rg*32+lane, g=kt*4+ki;
                if (r<rows && g<groups) out[size_t(r)*groups+g]=decode_scale(p[offset]);
            }
    return out;
}
double dot(const uint8_t* w,const double* sf,int k,const std::vector<double>& x) {
    static constexpr double values[]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
    double sum=0;
    for (int c=0;c<k;++c) sum+=values[(w[c/2]>>(4*(c%2)))&15]*sf[c/16]*x[c];
    return sum;
}
float alpha(const std::vector<uint8_t>& slab,int slot) { float v; std::memcpy(&v,slab.data()+slot*4,4); return v; }
std::vector<double> reference(const Fixture& f,const Layer& l,int experts) {
    const int h=f.h.hidden,n=f.h.intermediate;
    std::vector<double> x(h),middle(n),out(h,0);
    for (int i=0;i<h;++i) x[i]=decode_half(l.x[i]);
    for (int e=0;e<experts;++e) {
        auto sf13=linear_scales(l.slabs[2].data()+e*f.strides[2],2*n,h/16);
        auto sf2=linear_scales(l.slabs[3].data()+e*f.strides[3],h,n/16);
        const double ga=alpha(l.slabs[4],e)*f.h.inv13;
        const double ua=alpha(l.slabs[f.h.separate_up?6:4],e)*f.h.inv13;
        for (int i=0;i<n;++i) {
            int gate=i,up=i+n;
            if (f.h.layout==1) std::swap(gate,up);
            if (f.h.layout==2) { up=(i/64)*128+i%64; gate=up+64; }
            auto base=l.slabs[0].data()+e*f.strides[0];
            double g=dot(base+size_t(gate)*h/2,sf13.data()+size_t(gate)*h/16,h,x)*ga;
            double u=dot(base+size_t(up)*h/2,sf13.data()+size_t(up)*h/16,h,x)*ua;
            if (f.h.limit>0) { g=std::min(g,double(f.h.limit)); u=std::clamp(u,-double(f.h.limit),double(f.h.limit)); }
            const double sigmoid=g>=0?1/(1+std::exp(-g)):std::exp(g)/(1+std::exp(g));
            middle[i]=g*sigmoid*u;
        }
        // Round routing coefficients exactly as the ABI's FP32 inputs.
        const float route=1.f/experts;
        const double a=alpha(l.slabs[5],e)*f.h.inv2*route;
        for (int i=0;i<h;++i) out[i]+=a*dot(l.slabs[1].data()+e*f.strides[1]+size_t(i)*n/2,
                                            sf2.data()+size_t(i)*n/16,n,middle);
    }
    return out;
}
std::set<int> task_ids() {
    std::set<int> out;
    for (const auto& e:fs::directory_iterator("/proc/self/task")) out.insert(number(e.path().filename().string()));
    return out;
}
void verify_workers(const std::set<int>& before,const std::vector<int32_t>& cores) {
    std::vector<int> assigned;
    for (int tid:task_ids()) {
        if (tid!=getpid() && before.count(tid)) continue;
        cpu_set_t mask; CPU_ZERO(&mask);
        if (sched_getaffinity(tid,sizeof(mask),&mask) || CPU_COUNT(&mask)!=1) throw std::runtime_error("Worker not individually pinned");
        for (int c=0;c<CPU_SETSIZE;++c) if (CPU_ISSET(c,&mask)) assigned.push_back(c);
    }
    auto expected=cores; std::sort(expected.begin(),expected.end()); std::sort(assigned.begin(),assigned.end());
    if (assigned!=expected) throw std::runtime_error("Worker count/core mismatch");
}
struct Workload {
    Fixture& f; const Options& o; int experts;
    std::vector<int32_t> slots;
    std::vector<float> routes,out;
    std::vector<std::vector<double>> refs;
    Workload(Fixture& fixture,const Options& options,int count):f(fixture),o(options),experts(count),out(f.h.hidden) {
        for (int i=0;i<experts;++i) { slots.push_back(i); routes.push_back(1.f/experts); }
        for (const auto& l:f.layers) refs.push_back(reference(f,*l,experts));
    }
    void forward(size_t layer) {
        const auto& l=*f.layers[layer];
        int rc=sglang_nvfp4_cpu_experts_forward(l.handle,l.x.data(),slots.data(),routes.data(),experts,out.data(),o.workers,0);
        if (rc) throw std::runtime_error("CpuExpertForward status "+std::to_string(rc));
    }
    void validate() {
        for (size_t l=0;l<f.layers.size();++l) {
            forward(l);
            for (size_t i=0;i<out.size();++i) {
                const double gold=refs[l][i], tolerance=1e-4+1e-4*std::abs(gold);
                if (!std::isfinite(out[i]) || !std::isfinite(gold) || std::abs(out[i]-gold)>tolerance)
                    throw std::runtime_error("Dense reference mismatch at layer "+std::to_string(l)+" element "+std::to_string(i));
            }
        }
    }
    uint64_t bytes() const {
        uint64_t b=0; for (auto stride:f.strides) b+=stride;
        return b*experts;
    }
};
bool failed=false;
double quantile(std::vector<double>& samples,double p) {
    const size_t index=size_t(std::ceil(p*samples.size()))-1;
    std::nth_element(samples.begin(),samples.begin()+index,samples.end()); return samples[index]*1e6;
}
void run(benchmark::State& state,Workload& w) {
    try {
        w.validate(); for (int i=0;i<w.o.warmup;++i) w.forward(size_t(i)%w.f.layers.size());
        std::vector<double> samples; size_t layer=0; double total=0;
        for (auto _ : state) {
            (void)_;
            if (w.o.gap_us) std::this_thread::sleep_for(std::chrono::microseconds(w.o.gap_us));
            const auto begin=std::chrono::steady_clock::now(); w.forward(layer);
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            state.SetIterationTime(seconds); total+=seconds; samples.push_back(seconds);
            layer=(layer+1)%w.f.layers.size(); benchmark::DoNotOptimize(w.out.data()); benchmark::ClobberMemory();
        }
        w.validate();
        if (!samples.empty()) {
            state.counters["p50_us"]=quantile(samples,.50);
            state.counters["p95_us"]=quantile(samples,.95);
            state.counters["p99_us"]=quantile(samples,.99);
            state.counters["logical_weight_GBps"]=double(w.bytes())*samples.size()/total/1e9;
        }
        state.counters["experts"]=w.experts; state.counters["workers"]=w.o.workers;
        state.counters["layers"]=w.f.layers.size(); state.counters["weight_bytes_per_forward"]=w.bytes();
    } catch (const std::exception& e) { failed=true; state.SkipWithError(e.what()); }
}
} // namespace
int main(int argc,char** argv) {
    try {
        const auto o=parse(argc,argv); benchmark::Initialize(&argc,argv);
        if (benchmark::ReportUnrecognizedArguments(argc,argv)) return 1;
        auto cores=list(o.cpus,true), counts=list(o.experts,false);
        if (o.workers>int(cores.size())) throw std::runtime_error("Insufficient CPUs");
        cores.resize(o.workers);
        for (int c:cores) if (!fs::exists("/sys/devices/system/cpu/cpu"+std::to_string(c)+"/node"+std::to_string(o.node)))
            throw std::runtime_error("CPU not on requested NUMA node");
        if (sglang_nvfp4_cpu_experts_set_cores(cores.data(),cores.size())) throw std::runtime_error("Core list not allowed by affinity/cgroup");
        cpu_set_t caller; CPU_ZERO(&caller); CPU_SET(cores.front(),&caller);
        if (sched_setaffinity(0,sizeof(caller),&caller)) throw std::runtime_error("Caller pin failed");
        const auto before=task_ids(); Fixture fixture(o);
        if (!o.write_fixture.empty()) { fixture.write(o.write_fixture); std::cout<<"Fixture: "<<o.write_fixture<<'\n'; return 0; }
        fixture.register_layers();
        std::vector<std::unique_ptr<Workload>> workloads;
        for (int count:counts) {
            if (count<1 || count>int(fixture.h.capacity)) throw std::runtime_error("Expert count outside fixture capacity");
            auto w=std::make_unique<Workload>(fixture,o,count); w->validate(); workloads.push_back(std::move(w));
        }
        verify_workers(before,cores);
        std::cerr<<"Dense reference verified; individually pinned CPUs:"; for (int c:cores) std::cerr<<' '<<c;
        std::cerr<<"; memory policy inherited; inputs="<<(o.fixture.empty()?"synthetic":"fixture")<<'\n';
        if (o.validate_only) return 0;
        benchmark::AddCustomContext("backend",NVFP4_BENCH_BACKEND);
        benchmark::AddCustomContext("inputs",o.fixture.empty()?"synthetic":"fixture");
        benchmark::AddCustomContext("fixture",o.fixture);
        benchmark::AddCustomContext("seed",std::to_string(o.seed));
        benchmark::AddCustomContext("hidden",std::to_string(fixture.h.hidden));
        benchmark::AddCustomContext("intermediate",std::to_string(fixture.h.intermediate));
        benchmark::AddCustomContext("capacity",std::to_string(fixture.h.capacity));
        benchmark::AddCustomContext("act_limit",std::to_string(fixture.h.limit));
        benchmark::AddCustomContext("inv_input_scale13",std::to_string(fixture.h.inv13));
        benchmark::AddCustomContext("inv_input_scale2",std::to_string(fixture.h.inv2));
        benchmark::AddCustomContext("w13_layout",std::to_string(fixture.h.layout));
        benchmark::AddCustomContext("workers",std::to_string(o.workers));
        std::string assigned;
        for (int c:cores) assigned+=(assigned.empty()?"":",")+std::to_string(c);
        benchmark::AddCustomContext("cpu_list",assigned);
        benchmark::AddCustomContext("gap_us",std::to_string(o.gap_us));
        benchmark::AddCustomContext("compiler",__VERSION__);
        benchmark::AddCustomContext("memory_policy","inherited; no benchmark membind");
        for (auto& w:workloads) {
            auto ptr=w.get(); benchmark::RegisterBenchmark((std::string(NVFP4_BENCH_BACKEND)+"/experts:"+std::to_string(w->experts)).c_str(),
                [ptr](benchmark::State& state) { run(state,*ptr); })->UseManualTime()->Unit(benchmark::kMicrosecond);
        }
        benchmark::RunSpecifiedBenchmarks(); benchmark::Shutdown(); verify_workers(before,cores);
        return failed?1:0;
    } catch (const std::exception& e) { std::cerr<<"Error: "<<e.what()<<'\n'; return 1; }
}
