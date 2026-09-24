// Can io_uring O_DIRECT readv land an expert row straight in the pinned per-name slabs, with no bounce?
//
// The slabs are built as production builds them (expert_host_tier.allocate_host_slab with a placement):
// mmap, mbind to a node, cudaHostRegister. A "row image" is the six slab rows back to back (the re-laid file
// format under consideration); here it is simply a page-aligned range of a real mirror shard, since only the
// placement of the bytes matters, not their meaning. Every landed byte is compared with a buffered pread.
//
//   T0  statx DIO alignment of each file
//   T1  one readv of a whole row image into slot `slot` of the six slabs (small rows off page boundaries)
//   T2  the row cut into production-shaped sub-reads (page-aligned file cuts), half on each drive, one ring
//   T3  negative controls: a segment length off 512, a segment address off 4, a file offset off 512
//   T4  throughput: sub-reads readv'd into slab rows vs read into a posix_memalign bounce, per drive
//
// Build: g++ -O2 -std=c++17 direct_read_probe.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -luring
// Run:   direct_read_probe <file on root A> <file on root B> <numa node> [rows=96] [qd=16]

#include <cuda_runtime.h>
#include <fcntl.h>
#include <liburing.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t kPage = 4096;
constexpr int kNames = 6;
// EXL3_STREAMED_NAMES row bytes on dsv41-full40 (w13_suh, w13_svh, w13_trellis, w2_suh, w2_svh, w2_trellis).
constexpr int64_t kRowBytes[kNames] = {20480, 9216, 8847360, 4608, 10240, 4423680};
constexpr int kCapacity = 8;  // rows per slab
constexpr int64_t kSubRead = 1703936;  // ~1.66 MB, page-aligned: production's piece size

int64_t image_bytes() {
  int64_t s = 0;
  for (int64_t b : kRowBytes) s += b;
  return s;
}

void die(const char* what) {
  std::perror(what);
  std::exit(2);
}

struct Slabs {
  uint8_t* base[kNames] = {};
  uint8_t* row(int name, int slot) const { return base[name] + static_cast<int64_t>(slot) * kRowBytes[name]; }
};

Slabs make_slabs(int node) {
  Slabs s;
  for (int n = 0; n < kNames; ++n) {
    const size_t bytes = static_cast<size_t>(kRowBytes[n] * kCapacity);
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) die("mmap");
    unsigned long mask = 1ul << node;
    if (syscall(SYS_mbind, p, bytes, 2 /* MPOL_BIND */, &mask, 64, 0) != 0) die("mbind");
    if (cudaHostRegister(p, bytes, cudaHostRegisterDefault) != cudaSuccess) {
      std::fprintf(stderr, "cudaHostRegister failed\n");
      std::exit(2);
    }
    s.base[n] = static_cast<uint8_t*>(p);
  }
  return s;
}

// The iovecs that land image bytes [lo, hi) of a row image in slot `slot`.
std::vector<iovec> iovecs_for(const Slabs& s, int slot, int64_t lo, int64_t hi) {
  std::vector<iovec> v;
  int64_t at = 0;
  for (int n = 0; n < kNames; ++n) {
    const int64_t a = std::max(lo, at), b = std::min(hi, at + kRowBytes[n]);
    if (b > a) v.push_back(iovec{s.row(n, slot) + (a - at), static_cast<size_t>(b - a)});
    at += kRowBytes[n];
  }
  return v;
}

struct Req {
  int fd;
  int64_t offset;
  std::vector<iovec> iov;
  int64_t bytes;
  int res = 0;
};

// Submit every request with at most `qd` in flight; fill res. Returns wall seconds.
double run(io_uring* ring, std::vector<Req>& reqs, unsigned qd, bool vectored) {
  const auto t0 = std::chrono::steady_clock::now();
  size_t next = 0, done = 0;
  unsigned inflight = 0;
  while (done < reqs.size()) {
    while (next < reqs.size() && inflight < qd) {
      io_uring_sqe* sqe = io_uring_get_sqe(ring);
      Req& r = reqs[next];
      if (vectored) {
        io_uring_prep_readv(sqe, r.fd, r.iov.data(), static_cast<unsigned>(r.iov.size()), static_cast<uint64_t>(r.offset));
      } else {
        io_uring_prep_read(sqe, r.fd, r.iov[0].iov_base, static_cast<unsigned>(r.iov[0].iov_len), static_cast<uint64_t>(r.offset));
      }
      io_uring_sqe_set_data64(sqe, next);
      ++next;
      ++inflight;
    }
    io_uring_submit_and_wait(ring, 1);
    io_uring_cqe* cqe;
    unsigned head, seen = 0;
    io_uring_for_each_cqe(ring, head, cqe) {
      reqs[io_uring_cqe_get_data64(cqe)].res = cqe->res;
      ++seen;
    }
    io_uring_cq_advance(ring, seen);
    inflight -= seen;
    done += seen;
  }
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Compare the slot's six rows with a buffered read of the image at `offset`.
bool verify(int buffered_fd, int64_t offset, const Slabs& s, int slot) {
  std::vector<uint8_t> want(static_cast<size_t>(image_bytes()));
  if (pread(buffered_fd, want.data(), want.size(), offset) != static_cast<ssize_t>(want.size())) die("pread");
  int64_t at = 0;
  for (int n = 0; n < kNames; ++n) {
    if (std::memcmp(s.row(n, slot), want.data() + at, static_cast<size_t>(kRowBytes[n])) != 0) {
      std::printf("    MISMATCH in name %d slot %d\n", n, slot);
      return false;
    }
    at += kRowBytes[n];
  }
  return true;
}

void poison(const Slabs& s) {
  for (int n = 0; n < kNames; ++n) std::memset(s.base[n], 0xA5, static_cast<size_t>(kRowBytes[n] * kCapacity));
}

int64_t pick_offset(std::mt19937_64& rng, int64_t file_size) {
  const int64_t pages = (file_size - image_bytes()) / kPage;
  return static_cast<int64_t>(rng() % static_cast<uint64_t>(pages)) * kPage;
}

const char* verdict(int res, int64_t want) {
  return res == want ? "ok" : res < 0 ? std::strerror(-res) : "SHORT";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <file A> <file B> <node> [rows] [qd]\n", argv[0]);
    return 2;
  }
  const char* path[2] = {argv[1], argv[2]};
  const int node = std::atoi(argv[3]);
  const int rows = argc > 4 ? std::atoi(argv[4]) : 96;
  const unsigned qd = argc > 5 ? static_cast<unsigned>(std::atoi(argv[5])) : 16;
  const int64_t image = image_bytes();
  std::printf("row image %lld B (%% 512 = %lld), node %d\n", (long long)image, (long long)(image % 512), node);
  if (image % 512) return 2;

  int direct[2], buffered[2];
  int64_t size[2];
  for (int d = 0; d < 2; ++d) {
    direct[d] = open(path[d], O_RDONLY | O_DIRECT | O_CLOEXEC);
    buffered[d] = open(path[d], O_RDONLY | O_CLOEXEC);
    if (direct[d] < 0 || buffered[d] < 0) die("open");
    struct statx sx;
    if (statx(AT_FDCWD, path[d], 0, STATX_DIOALIGN | STATX_SIZE, &sx) != 0) die("statx");
    size[d] = static_cast<int64_t>(sx.stx_size);
    std::printf("T0 %s: dio_mem_align %u dio_offset_align %u (%s)\n", path[d], sx.stx_dio_mem_align,
                sx.stx_dio_offset_align, (sx.stx_mask & STATX_DIOALIGN) ? "reported" : "NOT reported");
  }

  if (cudaSetDevice(0) != cudaSuccess) die("cudaSetDevice");
  Slabs s = make_slabs(node);
  for (int n = 0; n < kNames; ++n) {
    std::printf("  slab %d base %% 4096 = %lld, slot 3 row %% 4096 = %lld, %% 512 = %lld\n", n,
                (long long)(reinterpret_cast<uintptr_t>(s.base[n]) % kPage),
                (long long)(reinterpret_cast<uintptr_t>(s.row(n, 3)) % kPage),
                (long long)(reinterpret_cast<uintptr_t>(s.row(n, 3)) % 512));
  }

  io_uring ring;
  if (io_uring_queue_init(64, &ring, 0) != 0) die("io_uring_queue_init");
  std::mt19937_64 rng(1234);
  bool all_ok = true;

  // T1: one readv per row image, per drive, into slots 0..3.
  for (int d = 0; d < 2; ++d) {
    for (int slot = 0; slot < 4; ++slot) {
      poison(s);
      std::vector<Req> r(1);
      r[0].fd = direct[d];
      r[0].offset = pick_offset(rng, size[d]);
      r[0].iov = iovecs_for(s, slot, 0, image);
      r[0].bytes = image;
      run(&ring, r, 1, true);
      const bool ok = r[0].res == image && verify(buffered[d], r[0].offset, s, slot);
      all_ok &= ok;
      std::printf("T1 drive %d slot %d: %zu iovecs, res %s, bytes %s\n", d, slot, r[0].iov.size(),
                  verdict(r[0].res, image), ok ? "match" : "WRONG");
    }
  }

  // T2: production-shaped: the image cut at page-aligned file offsets into sub-reads; the first half of the
  // sub-reads from drive 0, the rest from drive 1 (the mirror split), all in one ring, into slot 5.
  {
    poison(s);
    const int64_t offset = pick_offset(rng, std::min(size[0], size[1]));
    std::vector<Req> r;
    const int subs = static_cast<int>((image + kSubRead - 1) / kSubRead);
    for (int k = 0; k < subs; ++k) {
      const int64_t lo = k * kSubRead, hi = std::min(image, lo + kSubRead);
      Req q;
      q.fd = direct[k < subs / 2 ? 0 : 1];
      q.offset = offset + lo;
      q.iov = iovecs_for(s, 5, lo, hi);
      q.bytes = hi - lo;
      r.push_back(q);
    }
    run(&ring, r, qd, true);
    bool ok = true;
    for (auto& q : r) ok &= q.res == q.bytes;
    ok = ok && verify(buffered[0], offset, s, 5);
    all_ok &= ok;
    std::printf("T2 %zu sub-reads across both drives into slot 5: %s\n", r.size(), ok ? "all landed, bytes match" : "FAILED");
    for (size_t k = 0; k < r.size(); ++k) std::printf("    sub %zu: %zu iovecs res %s\n", k, r[k].iov.size(), verdict(r[k].res, r[k].bytes));
  }

  // T3: negative controls (expected to be refused).
  {
    const int64_t offset = pick_offset(rng, size[0]);
    struct Case {
      const char* name;
      int64_t off;
      std::vector<iovec> iov;
    };
    std::vector<Case> cases = {
        {"segment length off 512 (4608+256, 4096-256)", offset,
         {iovec{s.row(3, 1), 4608 + 256}, iovec{s.row(3, 3), 4096 - 256}}},
        {"segment address off 4 (+2)", offset, {iovec{s.row(3, 1) + 2, 4096}}},
        {"segment address 4-aligned, not 512 (+4)", offset, {iovec{s.row(3, 1) + 4, 4096}}},
        {"file offset off 512 (+256)", offset + 256, {iovec{s.row(0, 1), 4096}}},
    };
    for (auto& c : cases) {
      std::vector<Req> r(1);
      r[0].fd = direct[0];
      r[0].offset = c.off;
      r[0].iov = c.iov;
      r[0].bytes = 0;
      for (auto& v : c.iov) r[0].bytes += static_cast<int64_t>(v.iov_len);
      run(&ring, r, 1, true);
      std::printf("T3 %-44s -> %s\n", c.name, r[0].res < 0 ? std::strerror(-r[0].res) : r[0].res == r[0].bytes ? "ACCEPTED" : "short");
    }
  }

  // T4: throughput per drive, `rows` row images as sub-reads, qd in flight: readv into slab slots vs read into a
  // bounce. Alternate the order to spread any drive warm-up.
  const int64_t stride = (image + kPage - 1) / kPage * kPage;  // bounce slots are page-aligned, as in production
  void* bounce = nullptr;
  if (posix_memalign(&bounce, kPage, static_cast<size_t>(kCapacity * stride)) != 0) die("posix_memalign");
  std::memset(bounce, 0, static_cast<size_t>(kCapacity * stride));
  for (int d = 0; d < 2; ++d) {
    std::vector<int64_t> offsets;
    for (int i = 0; i < rows; ++i) offsets.push_back(pick_offset(rng, size[d]));
    double gbs[2] = {0, 0};
    size_t iovs = 0;
    for (int pass = 0; pass < 4; ++pass) {
      const bool vectored = (pass + d) % 2 == 0;
      std::vector<Req> r;
      for (int i = 0; i < rows; ++i) {
        const int slot = i % kCapacity;
        for (int64_t lo = 0; lo < image; lo += kSubRead) {
          const int64_t hi = std::min(image, lo + kSubRead);
          Req q;
          q.fd = direct[d];
          q.offset = offsets[static_cast<size_t>(i)] + lo;
          q.bytes = hi - lo;
          if (vectored) {
            q.iov = iovecs_for(s, slot, lo, hi);
          } else {
            q.iov = {iovec{static_cast<uint8_t*>(bounce) + slot * stride + lo, static_cast<size_t>(hi - lo)}};
          }
          r.push_back(q);
        }
      }
      const double t = run(&ring, r, qd, vectored);
      int64_t bytes = 0;
      bool ok = true;
      for (auto& q : r) {
        ok &= q.res == q.bytes;
        bytes += q.bytes;
        if (vectored) iovs += q.iov.size();
      }
      if (!ok) {
        std::printf("T4 drive %d pass %d: a read failed\n", d, pass);
        all_ok = false;
      }
      gbs[vectored ? 0 : 1] += bytes / t / 1e9 / 2;
      if (vectored && pass >= 2) {  // spot-check the last pass's final rows
        for (int slot = 0; slot < kCapacity && slot < rows; ++slot) {
          const int i = rows - kCapacity + slot;
          if (i >= 0) all_ok &= verify(buffered[d], offsets[static_cast<size_t>(i)], s, i % kCapacity);
        }
      }
    }
    std::printf("T4 drive %d: %d rows x %lld B, qd %u: readv into slabs %.2f GB/s, read into bounce %.2f GB/s (%.2f iovecs/sub-read)\n",
                d, rows, (long long)image, qd, gbs[0], gbs[1], iovs / 2.0 / (rows * ((image + kSubRead - 1) / kSubRead)));
  }
  std::printf("RESULT %s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
