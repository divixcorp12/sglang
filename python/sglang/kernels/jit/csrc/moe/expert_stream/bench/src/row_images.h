// Row images: the on-disk expert rows the bench's tier reads, and the files it writes for it.
//
// A row image holds one expert's six slab rows back to back. The tier's RowReader reads it with O_DIRECT straight into
// the slab rows, so every slab row is a multiple of 512 bytes and every image starts on a page boundary. This mirrors
// python/sglang/srt/layers/moe/exl3_row_image.py. No host header and no ATen here.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace fullstack {

constexpr int kNames = 6;  // Exl3RowLayout::kNames: w13_trellis, w13_suh, w13_svh, w2_trellis, w2_suh, w2_svh
constexpr int64_t kImageAlign = 512;  // O_DIRECT granularity

// Where each slab row sits inside one expert's image, and the image's size and stride in the file.
struct ImageLayout {
  std::array<int64_t, kNames> row_bytes{};     // each name's slab row
  std::array<int64_t, kNames> name_offsets{};  // where it sits in the image
  int64_t image_bytes = 0;
  int64_t row_stride = 0;  // image_bytes rounded up to a page: expert e's image is at e * row_stride
};

// Lays the names' slab rows back to back. Throws unless each is a positive multiple of 512 bytes (O_DIRECT).
ImageLayout image_layout(const std::array<int64_t, kNames>& row_bytes);

// What the tier reads: one image file per streamed row and that row's six slabs, `capacity` slots each.
// The slabs are owned by the caller (StackFixture).
struct RowSet {
  ImageLayout layout;
  int64_t experts = 0;
  int64_t capacity = 0;
  std::vector<std::string> paths;                   // [row]
  std::vector<std::array<uint8_t*, kNames>> slabs;  // [row][name]: slot s at slabs[row][n] + s * row_bytes[n]
};

// Creates `dir` and refuses it unless a file there opens with O_DIRECT (tmpfs does not).
void require_o_direct(const std::filesystem::path& dir);

// Writes `path` as a row-image layer file: expert e's image at e * row_stride, zero padded. `fill(e, image)` writes
// image_bytes bytes into a zeroed row. The file is written to "<path>.tmp", fsynced and renamed.
//
// A non-empty `stamp` lets an existing file of the right size be kept when "<path>.stamp" holds the same stamp.
// Returns true when the file was written, false when it was reused. Throws on an I/O error.
bool write_row_image(
    const std::filesystem::path& path,
    const ImageLayout& layout,
    int64_t experts,
    const std::function<void(int64_t expert, uint8_t* image)>& fill,
    const std::string& stamp);

}  // namespace fullstack
