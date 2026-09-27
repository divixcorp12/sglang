// Test-only submit faults (ReadFault's submit_* words) around any reader; zero faults forward every call unchanged.
#pragma once

#include "file_reader.h"

namespace sglang {
namespace expert_stream {

template <AsyncFileReader Inner>
class FaultyReader {
 public:
  void set_submit_fault(const SubmitFault& fault) {
    fault_ = fault;
  }
  bool init(unsigned depth) {
    return inner_.init(depth);
  }
  bool ready() const {
    return inner_.ready();
  }
  bool prep_read(int fd, void* buf, unsigned len, uint64_t off, uint64_t tag) {
    return inner_.prep_read(fd, buf, len, off, tag);
  }
  bool prep_readv(int fd, const iovec* iov, unsigned count, uint64_t off, uint64_t tag) {
    return inner_.prep_readv(fd, iov, count, off, tag);
  }
  int submit(unsigned wait_nr) {
    ++submits_;
    if (fault_.error != 0 && submits_ == fault_.call) {
      if (fault_.submit_first) inner_.submit(0);
      return -fault_.error;
    }
    // Fault: the kernel consumed none of the prepared SQEs and reported success. They stay prepared and
    // are counted in `pending`, so the next submit must send them; nothing may wait on them meanwhile.
    if (fault_.short_call != 0 && submits_ == fault_.short_call) return 0;
    return inner_.submit(wait_nr);
  }
  unsigned reap(std::vector<ReadCompletion>& out) {
    return inner_.reap(out);
  }
  void drain(unsigned pending) {
    inner_.drain(pending);
  }

 private:
  Inner inner_;
  SubmitFault fault_{};
  int64_t submits_ = 0;
};

}  // namespace expert_stream
}  // namespace sglang
