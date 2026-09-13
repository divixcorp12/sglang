#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>

#include <sys/stat.h>
#include <sys/uio.h>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <liburing.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace sglang {

namespace {

constexpr unsigned kRegisteredBufferSlots = 1024;
constexpr uint64_t kMaxRegisteredBufferBytes = 1ULL << 30;
constexpr uint64_t kMaxReadBytes = 1ULL << 30;

std::string errno_message(const std::string& what, int error) {
  return what + ": " + std::strerror(error);
}

pid_t current_thread_id() {
  thread_local const pid_t id = ::gettid();
  return id;
}

/// Prefer a ring only its creating thread may drive: the kernel then rejects
/// submissions and registrations from other threads, and runs completion work
/// inside that thread's own wait instead of interrupting it.
void init_ring(unsigned queue_depth, io_uring* ring) {
  int rc = -EINVAL;
#if defined(IORING_SETUP_SINGLE_ISSUER) && defined(IORING_SETUP_DEFER_TASKRUN)
  io_uring_params params{};
  params.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
  rc = io_uring_queue_init_params(queue_depth, ring, &params);
#endif
  if (rc == -EINVAL) {
    rc = io_uring_queue_init(queue_depth, ring, 0);
  }
  if (rc < 0) {
    throw std::runtime_error(errno_message("io_uring_queue_init", -rc));
  }
}

}  // namespace

/// Batched positional reads from a few files through one io_uring instance.
///
/// A read request is a list of ``(file, offset, destination address, length)``
/// extents. Every extent is submitted, short reads are resubmitted for their
/// remainder, and the call returns only after every in-flight completion has
/// been reaped, so no completion can outlive the destination memory the caller
/// guarantees for the duration of the call.
///
/// Registered buffers use ``IORING_OP_READ_FIXED`` when an extent's destination
/// lies entirely inside one of them, which avoids pinning destination pages on
/// every read. A registered buffer must stay allocated until it is unregistered
/// or the reader is closed: the ring holds its own page references, so reads
/// into a freed-then-reused address range would land in the old pages.
///
/// A reader belongs to the thread that constructs it, so the ring and its
/// tables need no lock: only that thread may open files, register buffers,
/// read, or close, and any other thread gets an error instead of waiting. The
/// one call allowed from any thread is ``request_unregister``, for garbage
/// collectors: it pushes the range onto a lock-free list that the owner applies
/// at the start of its next call. Memory freed before that call began cannot
/// still be named by a stale registration during it, and memory freed while
/// it runs cannot be one of its destinations.
struct UringFileReaderObj : public tvm::ffi::Object {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("sgl.UringFileReader", UringFileReaderObj, tvm::ffi::Object);
  static constexpr bool _type_mutable = true;

  explicit UringFileReaderObj(int64_t queue_depth) : owner_(current_thread_id()) {
    if (queue_depth < 1 || queue_depth > 32768) {
      throw std::invalid_argument("io_uring queue depth must be in [1, 32768]");
    }
    queue_depth_ = static_cast<unsigned>(queue_depth);
    init_ring(queue_depth_, &ring_);
    open_ = true;
    buffers_supported_ = io_uring_register_buffers_sparse(&ring_, kRegisteredBufferSlots) == 0;
    if (buffers_supported_) {
      slot_used_.assign(kRegisteredBufferSlots, false);
    }
  }

  ~UringFileReaderObj() {
    shutdown_();
    UnregisterRequest* request = unregister_requests_.exchange(nullptr);
    while (request != nullptr) {
      UnregisterRequest* next = request->next;
      delete request;
      request = next;
    }
  }

  int64_t open_file(const std::string& path, int64_t direct) {
    enter_();
    int flags = O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0);
    int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
      throw std::runtime_error(errno_message("open " + path, errno));
    }
    struct stat status;
    if (::fstat(fd, &status) != 0) {
      int error = errno;
      ::close(fd);
      throw std::runtime_error(errno_message("fstat " + path, error));
    }
    files_.push_back(File{fd, static_cast<uint64_t>(status.st_size)});
    return static_cast<int64_t>(files_.size() - 1);
  }

  int64_t file_size(int64_t file_id) {
    enter_();
    return static_cast<int64_t>(file_(file_id).size);
  }

  int64_t registered_buffers_supported() {
    return buffers_supported_ ? 1 : 0;
  }

  /// Register ``[address, address + nbytes)`` in chunks of at most 1 GiB.
  /// Returns the number of chunks registered, or zero when the ring does not
  /// support registration or rejects the memory; reads then pin per request.
  int64_t register_buffer(int64_t address, int64_t nbytes) {
    enter_();
    if (!buffers_supported_ || nbytes <= 0) {
      return 0;
    }
    uint64_t base = static_cast<uint64_t>(address);
    uint64_t remaining = static_cast<uint64_t>(nbytes);
    for (const Buffer& buffer : buffers_) {
      if (buffer.base < base + remaining && base < buffer.base + buffer.length) {
        return 0;
      }
    }
    std::vector<unsigned> added;
    while (remaining > 0) {
      auto free_slot = std::find(slot_used_.begin(), slot_used_.end(), false);
      if (free_slot == slot_used_.end()) {
        rollback_(added);
        return 0;
      }
      unsigned slot = static_cast<unsigned>(free_slot - slot_used_.begin());
      uint64_t length = std::min(remaining, kMaxRegisteredBufferBytes);
      struct iovec vector{reinterpret_cast<void*>(base), static_cast<size_t>(length)};
      __u64 tag = 0;
      if (io_uring_register_buffers_update_tag(&ring_, slot, &vector, &tag, 1) != 1) {
        rollback_(added);
        return 0;
      }
      slot_used_[slot] = true;
      buffers_.push_back(Buffer{base, length, slot});
      added.push_back(slot);
      base += length;
      remaining -= length;
    }
    sort_buffers_();
    return static_cast<int64_t>(added.size());
  }

  /// Unregister every chunk lying inside ``[address, address + nbytes)``.
  int64_t unregister_buffer(int64_t address, int64_t nbytes) {
    enter_();
    uint64_t low = static_cast<uint64_t>(address);
    return unregister_range_(low, low + static_cast<uint64_t>(std::max<int64_t>(nbytes, 0)));
  }

  /// Queue ``[address, address + nbytes)`` for unregistration; any thread may
  /// call this, and the owner applies it at the start of its next call.
  void request_unregister(int64_t address, int64_t nbytes) {
    uint64_t low = static_cast<uint64_t>(address);
    auto* request =
        new UnregisterRequest{low, low + static_cast<uint64_t>(std::max<int64_t>(nbytes, 0)), nullptr};
    request->next = unregister_requests_.load();
    while (!unregister_requests_.compare_exchange_weak(request->next, request)) {
    }
  }

  /// Read every extent completely. Returns the bytes read, which is less than
  /// the requested total only where an extent runs past the end of its file.
  int64_t read(
      const tvm::ffi::TensorView file_ids,
      const tvm::ffi::TensorView offsets,
      const tvm::ffi::TensorView destinations,
      const tvm::ffi::TensorView lengths) {
    enter_();
    const int64_t count = file_ids.size(0);
    if (offsets.size(0) != count || destinations.size(0) != count || lengths.size(0) != count) {
      throw std::invalid_argument("io_uring read extents must have matching lengths");
    }
    if (count == 0) {
      return 0;
    }
    const auto* file_id_values = static_cast<const int64_t*>(file_ids.data_ptr());
    const auto* offset_values = static_cast<const int64_t*>(offsets.data_ptr());
    const auto* destination_values = static_cast<const int64_t*>(destinations.data_ptr());
    const auto* length_values = static_cast<const int64_t*>(lengths.data_ptr());

    const size_t capacity = static_cast<size_t>(count);
    if (requests_.size() < capacity) {
      requests_.resize(capacity);
      pending_.resize(capacity);
    }
    // Each extent is pending or in flight at most once, so ``capacity`` slots
    // hold the whole FIFO without growing.
    size_t pending_head = 0;
    size_t pending_size = 0;
    for (int64_t index = 0; index < count; ++index) {
      if (offset_values[index] < 0 || length_values[index] < 0) {
        throw std::invalid_argument("io_uring read offsets and lengths must be non-negative");
      }
      const File& file = file_(file_id_values[index]);
      Request& request = requests_[static_cast<size_t>(index)];
      request.fd = file.fd;
      request.file_bytes = file.size;
      request.offset = static_cast<uint64_t>(offset_values[index]);
      request.destination = static_cast<uint64_t>(destination_values[index]);
      request.length = static_cast<uint64_t>(length_values[index]);
      request.done = 0;
      request.buffer_index = find_buffer_(request.destination, request.length);
      if (request.length > 0) {
        pending_[pending_size++] = static_cast<uint32_t>(index);
      }
    }

    unsigned inflight = 0;
    int first_error = 0;
    std::string error_context;
    while (pending_size > 0 || inflight > 0) {
      while (first_error == 0 && pending_size > 0 && inflight < queue_depth_) {
        io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (sqe == nullptr) {
          break;
        }
        uint32_t index = pending_[pending_head];
        pending_head = (pending_head + 1) % capacity;
        --pending_size;
        Request& request = requests_[index];
        uint64_t remaining = request.length - request.done;
        unsigned chunk = static_cast<unsigned>(std::min(remaining, kMaxReadBytes));
        auto* buffer = reinterpret_cast<void*>(request.destination + request.done);
        uint64_t position = request.offset + request.done;
        if (request.buffer_index >= 0) {
          io_uring_prep_read_fixed(sqe, request.fd, buffer, chunk, position, request.buffer_index);
        } else {
          io_uring_prep_read(sqe, request.fd, buffer, chunk, position);
        }
        sqe->user_data = index;
        ++inflight;
      }
      if (inflight == 0) {
        break;
      }
      int rc = io_uring_submit_and_wait(&ring_, 1);
      if (rc < 0 && rc != -EINTR && rc != -EAGAIN && rc != -EBUSY) {
        abandon_after_submit_failure_(inflight, -rc);
      }
      io_uring_cqe* cqe;
      unsigned head;
      unsigned seen = 0;
      io_uring_for_each_cqe(&ring_, head, cqe) {
        ++seen;
        uint32_t index = static_cast<uint32_t>(cqe->user_data);
        if (inflight == 0 || index >= capacity) {
          // A completion from outside this call would write through a
          // destination address that is no longer guaranteed to be alive.
          std::terminate();
        }
        --inflight;
        Request& request = requests_[index];
        int result = cqe->res;
        bool resubmit = false;
        if (result == -EINTR || result == -EAGAIN) {
          resubmit = true;
        } else if (result < 0) {
          if (first_error == 0) {
            first_error = -result;
            error_context = "io_uring read of " + std::to_string(request.length - request.done) + " bytes at offset " +
                            std::to_string(request.offset + request.done);
          }
        } else if (result == 0) {
          if (request.offset + request.done >= request.file_bytes) {
            request.length = request.done;
          } else if (first_error == 0) {
            first_error = EIO;
            error_context = "io_uring read returned no bytes before end of file at offset " +
                            std::to_string(request.offset + request.done);
          }
        } else {
          request.done += static_cast<uint64_t>(result);
          resubmit = request.done < request.length;
        }
        if (resubmit) {
          pending_[(pending_head + pending_size) % capacity] = index;
          ++pending_size;
        }
      }
      io_uring_cq_advance(&ring_, seen);
    }
    if (first_error != 0) {
      throw std::runtime_error(errno_message(error_context, first_error));
    }
    uint64_t total = 0;
    for (size_t index = 0; index < capacity; ++index) {
      total += requests_[index].done;
    }
    return static_cast<int64_t>(total);
  }

  void close() {
    check_owner_();
    shutdown_();
  }

 private:
  /// A failed submit may leave reads running in the kernel and unsubmitted
  /// SQEs in the ring. Wait out every read the kernel took, then tear the ring
  /// down so the unsubmitted SQEs can never run, and only then throw: no read
  /// may complete after the caller has regained ownership of its buffers.
  [[noreturn]] void abandon_after_submit_failure_(unsigned inflight, int error) {
    unsigned unsubmitted = std::min(inflight, io_uring_sq_ready(&ring_));
    unsigned in_kernel = inflight - unsubmitted;
    while (in_kernel > 0) {
      io_uring_cqe* cqe = nullptr;
      int rc = io_uring_wait_cqe(&ring_, &cqe);
      if (rc == -EINTR) {
        continue;
      }
      if (rc < 0) {
        std::terminate();
      }
      io_uring_cqe_seen(&ring_, cqe);
      --in_kernel;
    }
    shutdown_();
    throw std::runtime_error(errno_message("io_uring_submit_and_wait failed; reader closed", error));
  }

  void shutdown_() {
    if (!open_) {
      return;
    }
    open_ = false;
    for (const File& file : files_) {
      ::close(file.fd);
    }
    files_.clear();
    buffers_.clear();
    io_uring_queue_exit(&ring_);
  }

 private:
  struct File {
    int fd;
    uint64_t size;
  };

  struct Buffer {
    uint64_t base;
    uint64_t length;
    unsigned slot;
  };

  struct Request {
    int fd = -1;
    uint64_t file_bytes = 0;
    uint64_t offset = 0;
    uint64_t destination = 0;
    uint64_t length = 0;
    uint64_t done = 0;
    int buffer_index = -1;
  };

  struct UnregisterRequest {
    uint64_t low;
    uint64_t high;
    UnregisterRequest* next;
  };

  void check_owner_() const {
    const pid_t caller = current_thread_id();
    if (caller != owner_) {
      throw std::runtime_error(
          "io_uring file reader belongs to thread " + std::to_string(owner_) + " but was called from thread " +
          std::to_string(caller));
    }
  }

  /// Start an owner call: reject other threads and closed readers, then apply
  /// unregistrations queued since the previous call.
  void enter_() {
    check_owner_();
    ensure_open_();
    if (unregister_requests_.load() != nullptr) {
      UnregisterRequest* request = unregister_requests_.exchange(nullptr);
      while (request != nullptr) {
        UnregisterRequest* next = request->next;
        unregister_range_(request->low, request->high);
        delete request;
        request = next;
      }
    }
  }

  void ensure_open_() const {
    if (!open_) {
      throw std::runtime_error("io_uring file reader is closed");
    }
  }

  const File& file_(int64_t file_id) const {
    if (file_id < 0 || static_cast<size_t>(file_id) >= files_.size()) {
      throw std::out_of_range("io_uring file id " + std::to_string(file_id) + " is not open");
    }
    return files_[static_cast<size_t>(file_id)];
  }

  int find_buffer_(uint64_t destination, uint64_t length) const {
    auto after =
        std::upper_bound(buffers_.begin(), buffers_.end(), destination, [](uint64_t value, const Buffer& buffer) {
          return value < buffer.base;
        });
    if (after == buffers_.begin()) {
      return -1;
    }
    const Buffer& buffer = *(after - 1);
    if (destination + length <= buffer.base + buffer.length) {
      return static_cast<int>(buffer.slot);
    }
    return -1;
  }

  int64_t unregister_range_(uint64_t low, uint64_t high) {
    std::vector<unsigned> removed;
    for (const Buffer& buffer : buffers_) {
      if (buffer.base >= low && buffer.base + buffer.length <= high) {
        removed.push_back(buffer.slot);
      }
    }
    rollback_(removed);
    return static_cast<int64_t>(removed.size());
  }

  void rollback_(const std::vector<unsigned>& slots) {
    for (unsigned slot : slots) {
      struct iovec empty{nullptr, 0};
      __u64 tag = 0;
      io_uring_register_buffers_update_tag(&ring_, slot, &empty, &tag, 1);
      slot_used_[slot] = false;
    }
    buffers_.erase(
        std::remove_if(
            buffers_.begin(),
            buffers_.end(),
            [&](const Buffer& buffer) { return std::find(slots.begin(), slots.end(), buffer.slot) != slots.end(); }),
        buffers_.end());
  }

  void sort_buffers_() {
    std::sort(buffers_.begin(), buffers_.end(), [](const Buffer& left, const Buffer& right) {
      return left.base < right.base;
    });
  }

  const pid_t owner_;
  io_uring ring_{};
  unsigned queue_depth_ = 0;
  bool open_ = false;
  bool buffers_supported_ = false;
  std::vector<bool> slot_used_;
  std::vector<File> files_;
  std::vector<Buffer> buffers_;
  std::vector<Request> requests_;
  std::vector<uint32_t> pending_;
  std::atomic<UnregisterRequest*> unregister_requests_{nullptr};
};

void register_uring_file_reader() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<UringFileReaderObj>()
      .def(refl::init<int64_t>(), "__init__")
      .def("open_file", &UringFileReaderObj::open_file)
      .def("file_size", &UringFileReaderObj::file_size)
      .def("registered_buffers_supported", &UringFileReaderObj::registered_buffers_supported)
      .def("register_buffer", &UringFileReaderObj::register_buffer)
      .def("unregister_buffer", &UringFileReaderObj::unregister_buffer)
      .def("request_unregister", &UringFileReaderObj::request_unregister)
      .def("read", &UringFileReaderObj::read)
      .def("close", &UringFileReaderObj::close);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(register_once, register_uring_file_reader);

}  // namespace sglang
