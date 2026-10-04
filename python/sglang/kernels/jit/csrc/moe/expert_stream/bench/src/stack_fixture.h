// The real workload for the stack, built from the eight-layer DSV4.1 fixture (fixture.h).
//
// StackFixture holds, per layer, the six EXL3 slabs of kCapacity slots, the row-image file the tier reads, and the CPU
// experts' x and output rows. The interface carries no ATen type, so translation units that include the host headers
// (and tvm-ffi's) never include ATen. cpu_forward.cpp's kernel runtime is configured through the helpers at the end.
#pragma once

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
  static constexpr int64_t kCapacity = 8;  // slots per slab: 5 experts + 3 staging slots
  static constexpr int64_t kStaging = 3;

  // Loads `fixture` and writes (or reuses) one row-image file per layer under `image_dir`, which must accept O_DIRECT.
  StackFixture(const std::filesystem::path& fixture, const std::filesystem::path& image_dir);
  ~StackFixture();
  StackFixture(const StackFixture&) = delete;
  StackFixture& operator=(const StackFixture&) = delete;

  int64_t rows() const;
  int64_t experts() const;
  int64_t hidden() const;
  const RowSet& row_set() const;
  uint8_t* x_row(int64_t row) const;  // the row's FP16 input, padded to 16 bytes, as the post kernel stages it
  int64_t x_stride() const;
  float* out_row(int64_t row) const;  // part 0 (CPU hits) at [0, hidden), part 1 (CPU misses) at [hidden, 2 hidden)
  int64_t out_stride() const;         // bytes
  void write_x(int64_t row) const;    // the post's x store: the layer's fixture input into the row's x
  // Registers the row's kCapacity slot views with sglang_exl3_cpu_experts_register_layer; returns the layer handle.
  int64_t register_layer(int64_t row) const;
  // Fills slot e of the row's slabs with expert e from the row's image file: the slot the tier's reader would pick (the
  // lowest free slot after reserve_staging(kStaging)). For the bare forwards, which run before the stack exists.
  void preload_slots(int64_t row) const;
  static void free_layer(int64_t handle);

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

}  // namespace fullstack
