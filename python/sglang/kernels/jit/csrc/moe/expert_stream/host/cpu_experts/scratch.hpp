// A job's scratch: one buffer, kept with the job's slot and grown only, so a steady state allocates nothing and
// first-touches nothing. Each prepare lays its regions out afresh (ScratchLayout), then points into them.
#pragma once
#include <cstddef>
#include <new>
#include <type_traits>
#include <vector>

namespace sglang::cpu_experts {
// Internal linkage, like routes.hpp: each quant library's translation unit owns its state.
namespace {

// 64-byte-aligned storage, so a region that starts a cache line can be split between workers by whole lines.
template <class T>
struct CacheAligned
{
    using value_type = T;
    CacheAligned() = default;
    template <class U> CacheAligned(const CacheAligned<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{64})); }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t{64}); }
    template <class U> bool operator==(const CacheAligned<U>&) const { return true; }
};

// Byte offsets of a job's regions, each on its own 64-byte boundary.
struct ScratchLayout
{
    size_t bytes = 0;

    template <class T>
    size_t add(size_t count)
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);
        const size_t at = (bytes + 63) / 64 * 64;
        bytes = at + count * sizeof(T);
        return at;
    }
};

class Scratch
{
public:
    // Grows the buffer to the layout; new bytes are zero, old ones keep the previous job's values.
    void fit(const ScratchLayout& layout)
    {
        if (buffer_.size() < layout.bytes) buffer_.resize(layout.bytes);
    }

    template <class T>
    T* at(size_t offset)
    {
        return reinterpret_cast<T*>(buffer_.data() + offset);
    }

private:
    std::vector<std::byte, CacheAligned<std::byte>> buffer_;
};

}  // namespace
}  // namespace sglang::cpu_experts
