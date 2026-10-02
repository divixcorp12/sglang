#include "row_images.h"

#include "aligned.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace fullstack {
namespace fs = std::filesystem;

ImageLayout image_layout(const std::array<int64_t, kNames>& row_bytes) {
  ImageLayout layout;
  int64_t cursor = 0;
  for (int n = 0; n < kNames; ++n) {
    if (row_bytes[n] <= 0 || row_bytes[n] % kImageAlign != 0)
      throw std::runtime_error(
          "row images: name " + std::to_string(n) + "'s slab row is " + std::to_string(row_bytes[n]) +
          " B, not a positive multiple of 512");
    layout.row_bytes[n] = row_bytes[n];
    layout.name_offsets[n] = cursor;
    cursor += row_bytes[n];
  }
  layout.image_bytes = cursor;
  layout.row_stride = round_up(cursor, 4096);
  return layout;
}

void require_o_direct(const fs::path& dir) {
  fs::create_directories(dir);
  const fs::path probe = dir / ".o_direct_probe";
  const int fd = ::open(probe.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0644);
  if (fd < 0)
    throw std::runtime_error(
        "the image directory " + dir.string() + " does not accept O_DIRECT (" + std::strerror(errno) +
        "): use a disk filesystem, not tmpfs");
  ::close(fd);
  fs::remove(probe);
}

namespace {

// The file's contents, or an empty string when it cannot be read.
std::string read_text(const fs::path& path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

// Writes all `bytes`, retrying on EINTR and short writes.
void write_all(int fd, const uint8_t* data, size_t bytes, const fs::path& path) {
  while (bytes > 0) {
    const ssize_t n = ::write(fd, data, bytes);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("write " + path.string() + ": " + std::strerror(errno));
    }
    data += n;
    bytes -= static_cast<size_t>(n);
  }
}

}  // namespace

bool write_row_image(
    const fs::path& path,
    const ImageLayout& layout,
    int64_t experts,
    const std::function<void(int64_t, uint8_t*)>& fill,
    const std::string& stamp) {
  const auto file_bytes = static_cast<uintmax_t>(experts * layout.row_stride);
  const fs::path stamp_path = path.string() + ".stamp";
  std::error_code size_error;
  const uintmax_t size = fs::file_size(path, size_error);
  if (!stamp.empty() && !size_error && size == file_bytes && read_text(stamp_path) == stamp) return false;
  std::error_code ignored;
  fs::remove(stamp_path, ignored);  // a file without its stamp is never reused, so drop the stamp first
  const fs::path tmp = path.string() + ".tmp";
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) throw std::runtime_error("open " + tmp.string() + ": " + std::strerror(errno));
  std::vector<uint8_t> row(static_cast<size_t>(layout.row_stride));
  try {
    for (int64_t e = 0; e < experts; ++e) {
      std::fill(row.begin(), row.end(), 0);
      fill(e, row.data());
      write_all(fd, row.data(), row.size(), tmp);
    }
    if (::fsync(fd) != 0) throw std::runtime_error("fsync " + tmp.string() + ": " + std::strerror(errno));
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
  fs::rename(tmp, path);
  if (!stamp.empty()) {
    std::ofstream out(stamp_path);
    out << stamp;
    if (!out) throw std::runtime_error("cannot write " + stamp_path.string());
  }
  return true;
}

}  // namespace fullstack
