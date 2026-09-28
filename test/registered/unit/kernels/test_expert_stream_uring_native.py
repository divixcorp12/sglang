"""Exercise the production expert-stream reader against the real Linux io_uring API.

The default ring is required. Optional polling modes are skipped only after an
independent kernel probe and a check that UringReader refuses the unsupported
configuration; a supported mode must read correct bytes. Filesystem rejection of
IOPOLL is reported separately from ring/opcode support. No mocked completions.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_SOURCE = r'''
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "moe/expert_stream/host/uring_reader.h"

using sglang::expert_stream::ReadCompletion;
using sglang::expert_stream::UringReader;
constexpr size_t kPage = 4096;
constexpr size_t kArenaBytes = 16 * kPage;
constexpr size_t kFileBytes = 8 * kPage + 123;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
std::string setting(const char* key, const char* fallback) {
  const char* value = std::getenv(key);
  return value ? value : fallback;
}
bool optional_errno(int rc) {
  return rc == -EINVAL || rc == -EOPNOTSUPP || rc == -ENOSYS || rc == -EPERM || rc == -EACCES;
}
struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };

struct Arena {
  uint8_t* data = static_cast<uint8_t*>(
      mmap(nullptr, kArenaBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  Arena() { require(data != MAP_FAILED, "mmap aligned arena failed"); }
  ~Arena() { if (data != MAP_FAILED) munmap(data, kArenaBytes); }
};
struct DataFile {
  std::string path;
  std::vector<uint8_t> bytes;
  int fd = -1;
  DataFile(const std::string& directory, unsigned seed, bool direct) : bytes(kFileBytes) {
    std::string pattern = directory + "/uring-native-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const int writer = mkstemp(name.data());
    require(writer >= 0, "mkstemp failed");
    path = name.data();
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (i * 17 + (i / kPage) * 37 + seed) % 251;
    const ssize_t written = pwrite(writer, bytes.data(), bytes.size(), 0);
    const int synced = fsync(writer);
    ::close(writer);
    require(written == static_cast<ssize_t>(bytes.size()) && synced == 0, "fixture write/fsync failed");
    fd = ::open(path.c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
    require(fd >= 0, "opening fixture for reads failed: " + std::string(std::strerror(errno)));
  }
  ~DataFile() { if (fd >= 0) ::close(fd); if (!path.empty()) unlink(path.c_str()); }
};

// Probe the actual kernel separately: production may not quietly fall back to
// another setup mode or opcode when the caller explicitly requested one.
std::string unsupported_capability(const std::string& mode, const std::string& read_mode) {
  io_uring_params params{};
  if (mode == "iopoll" || mode == "sqpoll_iopoll") params.flags |= IORING_SETUP_IOPOLL;
  if (mode == "sqpoll" || mode == "sqpoll_iopoll") {
    params.flags |= IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 1000;
  }
  io_uring ring{};
  const int rc = io_uring_queue_init_params(8, &ring, &params);
  if (rc < 0) {
    require(mode != "default" || !optional_errno(rc),
            "required default io_uring setup unavailable: " + std::string(std::strerror(-rc)));
    require(optional_errno(rc), "unexpected kernel setup failure: " + std::to_string(rc));
    return "kernel setup " + mode + " refused: " + std::strerror(-rc);
  }
  std::string unsupported;
  if (read_mode == "readv_fixed") {
#if TEST_HAS_READV_FIXED
    io_uring_probe* probe = io_uring_get_probe_ring(&ring);
    const bool supported = probe && io_uring_opcode_supported(probe, IORING_OP_READV_FIXED);
    if (probe) io_uring_free_probe(probe);
    if (!supported) unsupported = "kernel does not advertise IORING_OP_READV_FIXED";
#else
    unsupported = "liburing headers do not expose IORING_OP_READV_FIXED";
#endif
  }
  io_uring_queue_exit(&ring);
  return unsupported;
}

std::vector<ReadCompletion> finish(UringReader& reader, unsigned count, bool allow_iopoll_unavailable = false) {
  std::vector<ReadCompletion> completions;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (completions.size() < count) {
    require(std::chrono::steady_clock::now() < deadline, "completion deadline exceeded");
    const int submitted = reader.submit(1);
    require(submitted >= 0, "submit failed: " + std::to_string(submitted));
    reader.reap(completions);
    for (const auto& completion : completions) {
      if (allow_iopoll_unavailable && (completion.res == -EOPNOTSUPP || completion.res == -EINVAL))
        throw Unsupported("filesystem/device rejected IOPOLL read: " + std::to_string(completion.res));
      require(completion.res >= 0, "read failed: " + std::to_string(completion.res));
    }
    std::this_thread::yield();
  }
  require(completions.size() == count, "unexpected extra completion");
  return completions;
}
void expect_completion(const std::vector<ReadCompletion>& completions, uint64_t tag, int result) {
  const auto match = std::find_if(completions.begin(), completions.end(),
                                 [=](const auto& completion) { return completion.data == tag; });
  require(match != completions.end(), "completion lost or changed its user tag");
  require(match->res == result, "wrong byte count for tag " + std::to_string(tag) + ": " + std::to_string(match->res));
}
void expect_bytes(const uint8_t* actual, const DataFile& file, size_t offset, size_t count) {
  require(std::memcmp(actual, file.bytes.data() + offset, count) == 0, "read bytes differ from file fixture");
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected temporary fixture directory");
    const std::string mode = setting("SGLANG_EXPERT_STREAM_URING_MODE", "default");
    const std::string read_mode = setting("SGLANG_EXPERT_STREAM_URING_READ_MODE", "normal");
    const bool iopoll = mode == "iopoll" || mode == "sqpoll_iopoll";
    const bool direct = setting("URING_TEST_DIRECT", "1") == "1";
    const unsigned configured_depth = std::stoul(setting("SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH", "0"));
    const unsigned depth = configured_depth == 0 ? 8 : configured_depth;
    const std::string unsupported = unsupported_capability(mode, read_mode);
    Arena arena;
    DataFile first(argv[1], 7, direct), second(argv[1], 113, direct);
    // Two nontrivial fd numbers prove fixed-file submission translates real fd
    // values to the registered table's indices. One arena encloses BOTH scatter
    // destinations: READV_FIXED has one buffer index, not an index per iovec.
    const std::vector<int> fds{second.fd, first.fd};
    const std::vector<iovec> buffers{{arena.data, kArenaBytes}};
    UringReader reader;
    if (!unsupported.empty()) {
      bool rejected = false;
      try {
        rejected = !reader.init(depth);
        if (!rejected) {
          reader.configure_resources(fds, buffers, direct);
          const iovec scatter[]{{arena.data + kPage, kPage}, {arena.data + 3 * kPage, kPage}};
          rejected = !reader.prep_readv(first.fd, scatter, 2, kPage, 1);
        }
      } catch (const std::exception& error) {
        rejected = true;
        std::cout << "PRODUCTION_REFUSAL " << error.what() << '\n';
      }
      reader.close();
      require(rejected, "unsupported explicit option was silently accepted: " + unsupported);
      throw Unsupported(unsupported + "; production refusal verified");
    }
    require(reader.init(depth), "production init failed despite successful independent setup probe");
    reader.configure_resources(fds, buffers, direct);
    std::memset(arena.data, 0xcd, kArenaBytes);
    const iovec scatter[]{{arena.data + kPage, kPage}, {arena.data + 3 * kPage, kPage}};
    const unsigned scatter_count = read_mode == "fixed" ? 1 : 2;
    if (read_mode == "fixed") {
      // READ_FIXED has one destination. The explicit mode must reject a
      // multi-iovec request instead of silently reverting to unregistered READV.
      bool refused = false;
      try { reader.prep_readv(first.fd, scatter, 2, kPage, 0xdead); }
      catch (const std::invalid_argument&) { refused = true; }
      require(refused, "fixed mode silently accepted multi-iovec READV");
    }
    // Production opens on the Python thread and submits on its service thread.
    // Do not accidentally add a setup flag that binds the ring to its creator.
    std::vector<ReadCompletion> completions;
    std::exception_ptr service_error;
    std::thread service([&] {
      try {
        require(reader.prep_readv(first.fd, scatter, scatter_count, kPage, 0x100000011ull), "prepare scatter read failed");
        require(reader.prep_read(second.fd, arena.data + 5 * kPage, kPage, 2 * kPage, 0x200000012ull),
                "prepare scalar read failed");
        require(reader.prep_read(first.fd, arena.data + 6 * kPage, kPage, 6 * kPage, 0x300000013ull),
                "prepare second scalar read failed");
        completions = finish(reader, 3, iopoll);
      } catch (...) { service_error = std::current_exception(); }
    });
    service.join();
    if (service_error) std::rethrow_exception(service_error);
    expect_completion(completions, 0x100000011ull, scatter_count * kPage);
    expect_completion(completions, 0x200000012ull, kPage);
    expect_completion(completions, 0x300000013ull, kPage);
    expect_bytes(arena.data + kPage, first, kPage, kPage);
    if (scatter_count == 2) expect_bytes(arena.data + 3 * kPage, first, 2 * kPage, kPage);
    expect_bytes(arena.data + 5 * kPage, second, 2 * kPage, kPage);
    expect_bytes(arena.data + 6 * kPage, first, 6 * kPage, kPage);
    require(std::all_of(arena.data + 2 * kPage, arena.data + 3 * kPage, [](uint8_t v) { return v == 0xcd; }),
            "scatter read overwrote the gap between destinations");

    // Discard a prepared, unsubmitted request. Ring recreation must preserve
    // fixed files and the common registered arena before the next read.
    require(reader.prep_read(second.fd, arena.data + 6 * kPage, kPage, 0, 21), "prepare before reset failed");
    reader.drain(1);
    require(reader.ready(), "drain left the reader unusable");
    require(reader.prep_readv(first.fd, scatter, scatter_count, 3 * kPage, 22), "prepare after reset failed");
    completions = finish(reader, 1);
    expect_completion(completions, 22, scatter_count * kPage);
    expect_bytes(arena.data + kPage, first, 3 * kPage, kPage);
    if (scatter_count == 2) expect_bytes(arena.data + 3 * kPage, first, 4 * kPage, kPage);

    // Drain submitted work before reusing the registered memory.
    require(reader.prep_read(second.fd, arena.data + 7 * kPage, kPage, 4 * kPage, 31), "prepare before drain failed");
    require(reader.submit(0) >= 0, "submit before drain failed");
    reader.drain(1);
    expect_bytes(arena.data + 7 * kPage, second, 4 * kPage, kPage);
    std::vector<ReadCompletion> empty;
    require(reader.reap(empty) == 0 && empty.empty(), "drain left a stale completion");

    require(reader.prep_read(first.fd, arena.data + 8 * kPage, kPage, 8 * kPage, 41), "prepare EOF read failed");
    completions = finish(reader, 1);
    expect_completion(completions, 41, 123);
    expect_bytes(arena.data + 8 * kPage, first, 8 * kPage, 123);

    // close() must settle/cancel writes before its caller reuses/frees buffers.
    require(reader.prep_read(second.fd, arena.data + 9 * kPage, kPage, 0, 51), "prepare before close failed");
    require(reader.submit(0) >= 0, "submit before close failed");
    reader.close();
    require(!reader.ready(), "close did not clear ready");
    reader.close();  // explicit close plus destructor must be safe
    std::memset(arena.data, 0xa7, kArenaBytes);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    require(std::all_of(arena.data, arena.data + kArenaBytes, [](uint8_t v) { return v == 0xa7; }),
            "kernel wrote into the arena after reader.close returned");

    require(reader.init(depth), "reinitialization failed");
    reader.configure_resources(fds, buffers, direct);
    require(reader.prep_read(first.fd, arena.data, kPage, 5 * kPage, 61), "prepare after reopen failed");
    completions = finish(reader, 1);
    expect_completion(completions, 61, kPage);
    expect_bytes(arena.data, first, 5 * kPage, kPage);
    reader.close();
    std::cout << "PASS real io_uring mode=" << mode << " read_mode=" << read_mode
              << " fixed_files=" << setting("SGLANG_EXPERT_STREAM_URING_FIXED_FILES", "0")
              << " wait_mode=" << setting("SGLANG_EXPERT_STREAM_URING_WAIT_MODE", "block")
              << " direct=" << direct << " scatter/reset/drain/short-read/close/reopen verified\n";
    return 0;
  } catch (const Unsupported& error) {
    std::cout << "CAPABILITY_UNSUPPORTED " << error.what() << '\n';
    return 77;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
'''


@pytest.fixture(scope="module")
def uring_native_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    assert compiler is not None, "native reader tests require a C++ compiler"
    root = Path(__file__).resolve().parents[4]
    build = tmp_path_factory.mktemp("expert-stream-uring-build")
    feature = build / "feature.cpp"
    feature.write_text("#include <liburing.h>\nint main() { return IORING_OP_READV_FIXED; }\n")
    probe = subprocess.run(
        [compiler, "-std=c++20", "-fsyntax-only", str(feature)], capture_output=True, text=True, check=False
    )
    source = build / "uring_native.cpp"
    source.write_text(_SOURCE)
    binary = build / "uring_native"
    compiled = subprocess.run(
        [
            compiler, "-std=c++20", "-O2", "-pthread", f"-DTEST_HAS_READV_FIXED={int(probe.returncode == 0)}",
            "-I", str(root / "python/sglang/kernels/jit/csrc"), str(source), "-luring", "-o", str(binary),
        ],
        capture_output=True, text=True, check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return binary


def _run_native(binary, tmp_path, *, mode, read_mode, fixed_files, wait_mode="block", direct=True):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SGLANG_EXPERT_STREAM_URING_")}
    env.update(
        SGLANG_EXPERT_STREAM_URING_MODE=mode,
        SGLANG_EXPERT_STREAM_URING_READ_MODE=read_mode,
        SGLANG_EXPERT_STREAM_URING_FIXED_FILES=str(int(fixed_files)),
        SGLANG_EXPERT_STREAM_URING_QUEUE_DEPTH="4" if wait_mode == "spin" else "0",
        SGLANG_EXPERT_STREAM_URING_WAIT_MODE=wait_mode,
        SGLANG_EXPERT_STREAM_URING_SQ_THREAD_IDLE_MS="1000",
        SGLANG_EXPERT_STREAM_URING_SQ_THREAD_CPU="-1",
        SGLANG_EXPERT_STREAM_URING_DIAGNOSTICS="1",
        URING_TEST_DIRECT=str(int(direct)),
    )
    completed = subprocess.run(
        [str(binary), str(tmp_path)], env=env, capture_output=True, text=True, timeout=20, check=False
    )
    output = completed.stdout + completed.stderr
    if completed.returncode == 77:
        assert "CAPABILITY_UNSUPPORTED" in output, output
        pytest.skip(output.strip())
    assert completed.returncode == 0, output
    assert "PASS real io_uring" in output, output


@pytest.mark.parametrize("mode", ["default", "iopoll", "sqpoll", "sqpoll_iopoll"])
@pytest.mark.parametrize("read_mode", ["normal", "fixed", "readv_fixed"])
@pytest.mark.parametrize("fixed_files", [False, True], ids=["raw-files", "fixed-files"])
def test_real_uring_read_and_resource_lifecycle(uring_native_binary, tmp_path, mode, read_mode, fixed_files):
    _run_native(uring_native_binary, tmp_path, mode=mode, read_mode=read_mode, fixed_files=fixed_files)


@pytest.mark.parametrize("mode", ["default", "iopoll", "sqpoll", "sqpoll_iopoll"])
@pytest.mark.parametrize("read_mode", ["normal", "fixed", "readv_fixed"])
def test_real_uring_spin_wait_and_explicit_depth(uring_native_binary, tmp_path, mode, read_mode):
    _run_native(
        uring_native_binary, tmp_path, mode=mode, read_mode=read_mode, fixed_files=True, wait_mode="spin"
    )


def test_default_reader_also_supports_buffered_files(uring_native_binary, tmp_path):
    _run_native(uring_native_binary, tmp_path, mode="default", read_mode="normal", fixed_files=False, direct=False)
