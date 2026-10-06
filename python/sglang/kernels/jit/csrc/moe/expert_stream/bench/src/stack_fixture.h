// The real workload for the stack, built from the eight-layer DSV4.1 fixture (fixture.h).
//
// StackFixture holds, per layer, the six EXL3 slabs of capacity() slots (kGroupSlots per NUMA group), the row-image file the tier reads, and the CPU
// experts' x and output rows. The interface carries no ATen type, so translation units that include the host headers
// (and tvm-ffi's) never include ATen. cpu_forward.cpp's kernel runtime is configured through the helpers at the end.
#pragma once

#include "expert_stream/host/cpu_experts/kernel.hpp"
#include "expert_stream/lease_layout.h"
#include "row_images.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace fullstack {

// Owns the slabs, the x and output rows, and the written row-image files' paths. Construct it on a worker-node thread:
// the slabs, x and output rows are first-touched there. Not copyable; the pointers it returns live as long as it does.
class StackFixture {
 public:
  static constexpr int64_t kGroupSlots = 8;  // per NUMA group: 5 experts + 3 staging slots
  static constexpr int64_t kStaging = 3;
  int64_t capacity() const {
    return kGroupSlots * ::sglang::expert_stream::wire::Wire::kNodes;
  }

  // Where the tier maps `expert`: its home group's slots, after that group's kStaging staging slots took and returned
  // the first misses (the lowest free slot each time), so a group's j-th expert is at its j-th slot. That holds when
  // each group's experts are loaded in ascending order in posts of kStaging, none mixed with another group's.
  static int32_t slot_of(int64_t expert) {
    using Wire = ::sglang::expert_stream::wire::Wire;
    return static_cast<int32_t>(Wire::home(expert) * kGroupSlots + expert / Wire::kNodes);
  }

  // Loads `fixture` and writes (or reuses) one row-image file per layer under `image_dir`, which must accept O_DIRECT.
  // Above one NUMA group, each group's slots of every slab are bound to that group's node (group_nodes[g]) before the
  // images are written.
  StackFixture(
      const std::filesystem::path& fixture, const std::filesystem::path& image_dir, std::vector<int> group_nodes = {});
  ~StackFixture();
  StackFixture(const StackFixture&) = delete;
  StackFixture& operator=(const StackFixture&) = delete;

  int64_t rows() const;
  int64_t experts() const;
  int64_t hidden() const;
  const RowSet& row_set() const;
  uint8_t* x_row(int64_t row) const;  // the row's FP16 input, padded to 16 bytes, as the post kernel stages it
  int64_t x_stride() const;
  // Group g's part 0 (CPU hits) at [2 g hidden, (2 g + 1) hidden), its part 1 (CPU misses) in the next hidden floats.
  float* out_row(int64_t row) const;
  int64_t out_stride() const;         // bytes
  void write_x(int64_t row) const;    // the post's x store: the layer's fixture input into the row's x
  // The row's layer over its capacity() slot views, made by the EXL3 kernel (exl3_cpu_kernel).
  ::sglang::cpu_experts::ExpertLayer make_layer(int64_t row) const;
  // Fills slot_of(e) of the row's slabs with expert e from the row's image file: the slot the tier's reader would pick
  // (see slot_of). For the bare forwards, which run before the stack exists.
  void preload_slots(int64_t row) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Configures cpu_forward.cpp's kernel runtime: AVX512BW exactly (EXL3_MOE_CPU_MAX_ISA=bw set at launch),
// single-threaded ATen, no dynamic OpenMP. Throws if the kernel picked another ISA.
void configure_cpu_kernel_runtime();

// Frees the calling thread's OpenMP team. The kernel forks one team per calling thread (libgomp keeps a helper pool
// per master), and with GOMP_SPINCOUNT=INFINITE an idle team spins forever on its cores, so the bare caller's helpers
// would compete with the CPU expert thread's. Throws if the team cannot be released.
void release_kernel_team();

// Bit-exact comparison against a reference file (fixture.h's compare_reference); throws naming the file.
void check_reference(const std::filesystem::path& path, const std::vector<float>& actual);

// Reads the reference as check_reference does and throws, naming the file, when any |a - b| > tol * max(1, |b|): the
// sum of two groups' parts differs from one engine's sum by fp32 reassociation.
void check_reference_close(const std::filesystem::path& path, const std::vector<float>& actual, float tol);

}  // namespace fullstack
