// Diagnostic admission gate. Create a four-byte uint32 file before launching; keep it
// mapped and never truncate/replace it. Zero closes admission, one opens admission.
// The path is SGLANG_EXPERT_STREAM_TRACE's gate_path (TraceOptions); an empty one always admits.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sglang::expert_stream {
class TraceGate {
 public:
    explicit TraceGate(const std::string& path) {
        if (path.empty()) return;
        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("cannot open CPU expert trace gate");
        struct stat st{};
        if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size != sizeof(uint32_t)) {
            close(fd);
            throw std::runtime_error("CPU expert trace gate must be a four-byte regular file");
        }
        void* p = mmap(nullptr, sizeof(uint32_t), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("cannot map CPU expert trace gate");
        flag_ = static_cast<const uint32_t*>(p);
        if (__atomic_load_n(flag_, __ATOMIC_ACQUIRE) > 1) {
            munmap(p, sizeof(uint32_t)); flag_ = nullptr;
            throw std::runtime_error("CPU expert trace gate must contain zero or one");
        }
    }
    ~TraceGate() { if (flag_) munmap(const_cast<uint32_t*>(flag_), sizeof(uint32_t)); }
    TraceGate(const TraceGate&) = delete;
    TraceGate& operator=(const TraceGate&) = delete;
    bool enabled() const { return !flag_ || __atomic_load_n(flag_, __ATOMIC_ACQUIRE) == 1; }
 private:
    const uint32_t* flag_ = nullptr;
};
} // namespace sglang::expert_stream
