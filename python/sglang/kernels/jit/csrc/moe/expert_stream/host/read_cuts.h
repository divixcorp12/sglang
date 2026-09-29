// Cutting a read into legs its block device takes whole (plan 2026-09-28-iopoll-read-cuts). Under IORING_SETUP_IOPOLL
// io_uring issues a read non-blocking and the polled bio carries REQ_NOWAIT; the block layer refuses to split such a
// bio (-EAGAIN) and io_uring punts the read to an io-wq worker. A read needs a split when it is larger than the
// queue's max_sectors_kb / max_segments, or when two of its iovecs meet off the NVMe PRP boundary (virt_boundary_mask).
// Cutting at both keeps every leg one request (analysis/dsv41-drive/iopoll/diagnosis.md).
#pragma once

#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/uio.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>

namespace sglang::expert_stream {

constexpr int64_t kCutPage = 4096;
constexpr int64_t kFallbackCutBytes = 128 * 1024;
constexpr uint64_t kFallbackVirtMask = 4095;

struct DeviceLimits {
  int64_t cut_bytes = kFallbackCutBytes;   // the largest leg: whole pages
  uint64_t virt_mask = kFallbackVirtMask;  // a join off (mask + 1) starts a new leg; 0: no such rule
  std::string source;                      // the queue directory, "fallback: <why>", or a test's label
};

// The largest leg the queue takes as one request: max_sectors_kb, and max_segments pages less one (a leg that starts
// mid-page spans one page more than its length), rounded down to whole pages, never below one page. max_segments 0
// (unknown) leaves max_sectors_kb alone.
inline int64_t cut_bytes_for(int64_t max_sectors_kb, int64_t max_segments) {
  int64_t bytes = max_sectors_kb * 1024;
  if (max_segments > 1) bytes = std::min(bytes, (max_segments - 1) * kCutPage);
  return std::max(kCutPage, bytes / kCutPage * kCutPage);
}

inline bool read_sysfs_number(const std::string& path, int64_t* out) {
  std::ifstream f(path);
  long long v = -1;
  if (!(f >> v) || v < 0) return false;
  *out = static_cast<int64_t>(v);
  return true;
}

// Limits from a block queue directory (.../queue). virt_boundary_mask absent (older kernels) counts as 4095.
inline bool limits_from_queue_dir(const std::string& dir, DeviceLimits* out, std::string* why) {
  int64_t kb = 0, segments = 0, mask = 0;
  if (!read_sysfs_number(dir + "/max_sectors_kb", &kb) || kb == 0) {
    *why = dir + "/max_sectors_kb unreadable";
    return false;
  }
  if (!read_sysfs_number(dir + "/max_segments", &segments) || segments == 0) {
    *why = dir + "/max_segments unreadable";
    return false;
  }
  if (!read_sysfs_number(dir + "/virt_boundary_mask", &mask)) mask = static_cast<int64_t>(kFallbackVirtMask);
  *out = DeviceLimits{cut_bytes_for(kb, segments), static_cast<uint64_t>(mask), dir};
  return true;
}

// The queue directory of the block device holding a file (st_dev): /sys/dev/block/M:m resolved, and for a partition
// its disk's. Empty with the reason when there is none (tmpfs and other major-0 filesystems, a missing link).
inline std::string queue_dir_for(dev_t dev, std::string* why) {
  if (major(dev) == 0) {
    *why = "not on a block device (st_dev major 0)";
    return {};
  }
  const std::string link = "/sys/dev/block/" + std::to_string(major(dev)) + ":" + std::to_string(minor(dev));
  char real[PATH_MAX];
  if (realpath(link.c_str(), real) == nullptr) {
    *why = link + " does not resolve";
    return {};
  }
  std::string dir(real);
  struct stat st;
  if (stat((dir + "/partition").c_str(), &st) == 0) dir = dir.substr(0, dir.rfind('/'));
  if (stat((dir + "/queue").c_str(), &st) != 0) {
    *why = dir + "/queue absent";
    return {};
  }
  return dir + "/queue";
}

// The file's device limits, or the fallback (128 KiB, 4 KiB boundary) with its reason in `source`.
inline DeviceLimits device_limits(int fd) {
  std::string why = "fstat failed";
  struct stat st;
  if (fstat(fd, &st) == 0) {
    const std::string dir = queue_dir_for(st.st_dev, &why);
    DeviceLimits found;
    if (!dir.empty() && limits_from_queue_dir(dir, &found, &why)) return found;
  }
  DeviceLimits fallback;
  fallback.source = "fallback: " + why;
  return fallback;
}

// One leg over `out`: iovecs [first, first + count), `bytes` long; `gap` when a boundary gap (not the size) opened it.
struct CutLeg {
  unsigned first = 0;
  unsigned count = 0;
  int64_t bytes = 0;
  bool gap = false;
};

// Cut a read's iovecs into legs: a new leg at every join off the virt boundary, and wherever a leg reaches
// lim.cut_bytes (splitting that iovec in two). `out` receives the legs' iovecs in order (at most count + legs - 1),
// `legs` the legs over them. Returns the leg count, or max_legs + 1 when `legs` or `out` is too small.
inline unsigned cut_legs(
    const iovec* in, unsigned count, const DeviceLimits& lim, iovec* out, unsigned max_out, CutLeg* legs,
    unsigned max_legs) {
  const auto on_boundary = [&](uintptr_t at) { return lim.virt_mask == 0 || (at & lim.virt_mask) == 0; };
  unsigned n = 0, k = 0;
  for (unsigned i = 0; i < count; ++i) {
    auto* at = static_cast<uint8_t*>(in[i].iov_base);
    size_t left = in[i].iov_len;
    const bool gap = i > 0 && !(on_boundary(reinterpret_cast<uintptr_t>(in[i - 1].iov_base) + in[i - 1].iov_len) &&
                                on_boundary(reinterpret_cast<uintptr_t>(at)));
    bool open_leg = n == 0 || gap;
    bool first_piece = true;
    while (left > 0) {
      if (!open_leg && legs[n - 1].bytes >= lim.cut_bytes) open_leg = true;
      if (open_leg) {
        if (n == max_legs) return max_legs + 1;
        legs[n++] = CutLeg{k, 0, 0, first_piece && gap};
        open_leg = false;
      }
      if (k == max_out) return max_legs + 1;
      CutLeg& leg = legs[n - 1];
      const size_t take = std::min<size_t>(left, static_cast<size_t>(lim.cut_bytes - leg.bytes));
      out[k++] = iovec{at, take};
      ++leg.count;
      leg.bytes += static_cast<int64_t>(take);
      at += take;
      left -= take;
      first_piece = false;
    }
  }
  return n;
}

// An upper bound on the legs of a read of `longest` bytes over `iovecs` iovecs cut at `cut_bytes`, including the
// fixed modes' further cuts at registered-buffer changes. A join splits a read at most once (a gap, or a buffer
// change), and a run between split joins of r bytes gives at most ceil(r / cut) legs, so the legs are at most
// 1 + (iovecs - 1) + sum(ceil(r / cut) - 1) <= iovecs + longest / cut.
inline size_t leg_bound(int64_t longest, size_t iovecs, int64_t cut_bytes) {
  return static_cast<size_t>(longest / cut_bytes) + iovecs;
}

}  // namespace sglang::expert_stream
