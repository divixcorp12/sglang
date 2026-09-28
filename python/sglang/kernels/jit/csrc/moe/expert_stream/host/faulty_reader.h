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
  void configure_resources(const std::vector<int>& files, const std::vector<RegisteredRegion>& buffers, bool direct)
    requires requires(Inner& reader) { reader.configure_resources(files, buffers, direct); }
  {
    inner_.configure_resources(files, buffers, direct);
  }
  void set_fixed_chunk_cap(size_t cap)
    requires requires(Inner& reader) { reader.set_fixed_chunk_cap(cap); }
  {
    inner_.set_fixed_chunk_cap(cap);
  }
  bool fixed_reads() const
    requires requires(const Inner& reader) { reader.fixed_reads(); }
  {
    return inner_.fixed_reads();
  }
  unsigned sq_space() const
    requires requires(const Inner& reader) { reader.sq_space(); }
  {
    return inner_.sq_space();
  }
  unsigned fixed_legs(const iovec* iov, unsigned count, FixedLeg* out) const
    requires requires(const Inner& reader) { reader.fixed_legs(iov, count, out); }
  {
    return inner_.fixed_legs(iov, count, out);
  }
  bool prep_readv_fixed(int fd, const iovec* iov, unsigned count, uint64_t off, int buffer, uint64_t tag)
    requires requires(Inner& reader) { reader.prep_readv_fixed(fd, iov, count, off, buffer, tag); }
  {
    return inner_.prep_readv_fixed(fd, iov, count, off, buffer, tag);
  }
  void note_fanout(unsigned legs)
    requires requires(Inner& reader) { reader.note_fanout(legs); }
  {
    inner_.note_fanout(legs);
  }
  uint64_t fixed_cuts() const
    requires requires(const Inner& reader) { reader.fixed_cuts(); }
  {
    return inner_.fixed_cuts();
  }
  uint64_t fanout_sqes() const
    requires requires(const Inner& reader) { reader.fanout_sqes(); }
  {
    return inner_.fanout_sqes();
  }
  void close()
    requires requires(Inner& reader) { reader.close(); }
  {
    inner_.close();
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
