#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" __attribute__((noinline, visibility("default")))
void magic_trace_stop_indicator(uint32_t sequence, int64_t delay_ns, int reason) {
  asm volatile("" : : "r"(sequence), "r"(delay_ns), "r"(reason) : "memory");
}
__attribute__((noinline)) uint64_t work(uint64_t n) {
  for (int i = 0; i < 64; ++i) n = n * 6364136223846793005ULL + 1;
  asm volatile("" : "+r"(n) : : "memory");
  return n;
}
int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: probe READY_JSON COLLECTOR_READY main|worker trigger|no-trigger\n");
    return 2;
  }
  const bool worker = std::string(argv[3]) == "worker";
  const bool trigger = std::string(argv[4]) == "trigger";
  auto run = [&] {
    FILE* out = std::fopen(argv[1], "wx");
    if (!out) throw std::runtime_error("cannot create fresh probe ready file");
    std::fprintf(out, "{\"pid\":%d,\"tid\":%ld}\n", getpid(), syscall(SYS_gettid));
    std::fclose(out);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(45);
    while (!std::filesystem::exists(argv[2])) {
      if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("collector readiness timeout");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto start = std::chrono::steady_clock::now();
    uint64_t n = 1;
    bool fired = false;
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(trigger ? 2 : 5)) {
      n = work(n);
      const auto elapsed = std::chrono::steady_clock::now() - start;
      if (trigger && !fired && elapsed > std::chrono::milliseconds(50)) {
        fired = true;
        magic_trace_stop_indicator(1, std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(), 1);
      }
    }
    std::printf("done %llu\n", static_cast<unsigned long long>(n));
  };
  if (worker) { std::thread thread(run); thread.join(); }
  else run();
}
