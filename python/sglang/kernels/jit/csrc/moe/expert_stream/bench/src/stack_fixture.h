// The real workload for the stack, from the eight-layer DSV4.1 fixture (fixture.h): each layer's six EXL3 slabs of
// kCapacity slots, its row-image file, and the CPU experts' x and output rows. No ATen type in this interface, so the
// translation units that include the host headers (and tvm-ffi's) never include ATen. Construct it on a worker-node
// thread: the slabs, x and output rows are first-touched there.
#pragma once

#include "row_images.h"
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace fullstack {

class StackFixture {
 public:
  static constexpr int64_t kCapacity = 8;  // 5 experts + 3 staging slots
  static constexpr int64_t kStaging = 3;

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
  int64_t register_layer(int64_t row) const;  // exl3_moe_cpu_make_layer over the row's kCapacity slot views
  // Slot e of the row's slabs := expert e, from the row's image file: where the tier's reader puts it (the lowest free
  // slot after reserve_staging(kStaging)). For the bare forwards, which run before the stack exists.
  void preload_slots(int64_t row) const;
  static void free_layer(int64_t handle);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// cpu_forward.cpp's kernel runtime: AVX512BW exactly (EXL3_MOE_CPU_MAX_ISA=bw set at launch), single-threaded ATen,
// no dynamic OpenMP. Throws.
void configure_cpu_kernel_runtime();

// The kernel forks an OpenMP team per calling thread (libgomp keeps a helper pool per master), and with
// GOMP_SPINCOUNT=INFINITE an idle team spins forever on its cores. Frees the calling thread's team, so the bare
// caller's helpers stop competing with the CPU expert thread's. Throws.
void release_kernel_team();

// fixture.cpp's compare_reference: bit-exact, or throws naming the file.
void check_reference(const std::filesystem::path& path, const std::vector<float>& actual);

}  // namespace fullstack
