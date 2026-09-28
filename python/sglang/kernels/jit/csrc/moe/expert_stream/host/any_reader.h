// The tier's reader: a PackReader (bounce and pack) or a RowReader (direct row images), chosen once by
// Tables::images. RamTier and HostExports hold one Source type; the variant keeps them unchanged.
#pragma once

#include "pack_reader.h"
#include "row_reader.h"
#include <variant>

namespace sglang::expert_stream {

template <ExpertRowLayout Layout, AsyncFileReader Reader>
class AnyReader {
 public:
  using LayoutType = Layout;
  using SqeRecord = expert_stream::SqeRecord;

  AnyReader(Tables tables, bool direct, int64_t pack_workers = 0, int64_t pack_split = 0) {
    if (tables.images) {
      impl_.template emplace<RowReader<Layout, Reader>>(std::move(tables), direct, pack_workers, pack_split);
    } else {
      impl_.template emplace<PackReader<Layout, Reader>>(std::move(tables), direct, pack_workers, pack_split);
    }
  }
  AnyReader(const AnyReader&) = delete;
  AnyReader& operator=(const AnyReader&) = delete;

  bool is_pack() const {
    return std::holds_alternative<PackReader<Layout, Reader>>(impl_);
  }
  const Tables& tables() const {
    return visit([](auto& r) -> const Tables& { return r.tables(); });
  }
  void set_pack(int64_t w, int64_t s) {
    visit([&](auto& r) { r.set_pack(w, s); });
  }
  unsigned pack_workers() const {
    return visit([](auto& r) { return r.pack_workers(); });
  }
  unsigned pack_split() const {
    return visit([](auto& r) { return r.pack_split(); });
  }
  std::vector<int> packing_cpus() const {
    return visit([](auto& r) { return r.packing_cpus(); });
  }
  void set_piece_stream(bool on) {
    visit([&](auto& r) { r.set_piece_stream(on); });
  }
  bool piece_stream() const {
    return visit([](auto& r) { return r.piece_stream(); });
  }
  int64_t publish_refused() const {
    return visit([](auto& r) { return r.publish_refused(); });
  }
  size_t descriptors() const {
    return visit([](auto& r) { return r.descriptors(); });
  }
  unsigned credit() const {
    return visit([](auto& r) { return r.credit(); });
  }
  void set_sqe_log(std::vector<SqeRecord>* log) {
    visit([&](auto& r) { r.set_sqe_log(log); });
  }
  void set_owner_core(int64_t core) {
    visit([&](auto& r) { r.set_owner_core(core); });
  }
  int64_t unfinished_jobs() const {
    return visit([](auto& r) { return r.unfinished_jobs(); });
  }
  void set_fault(const ReadFault& f) {
    visit([&](auto& r) { r.set_fault(f); });
  }
  int64_t cqes() const {
    return visit([](auto& r) { return r.cqes(); });
  }
  int64_t stale_cqes() const {
    return visit([](auto& r) { return r.stale_cqes(); });
  }
  int64_t generation_wraps() const {
    return visit([](auto& r) { return r.generation_wraps(); });
  }
  void set_fixed_chunk_cap(int64_t cap) {
    visit([&](auto& r) { r.set_fixed_chunk_cap(cap); });
  }
  int64_t fixed_cuts() const {
    return visit([](auto& r) { return r.fixed_cuts(); });
  }
  int64_t fanout_sqes() const {
    return visit([](auto& r) { return r.fanout_sqes(); });
  }
  bool open() {
    return visit([](auto& r) { return r.open(); });
  }
  template <class... Args>
  int read(Args&&... args) {
    return visit([&](auto& r) { return r.read(std::forward<Args>(args)...); });
  }

 private:
  template <class F>
  decltype(auto) visit(F&& f) {
    if (auto* p = std::get_if<PackReader<Layout, Reader>>(&impl_)) return f(*p);
    return f(std::get<RowReader<Layout, Reader>>(impl_));
  }
  template <class F>
  decltype(auto) visit(F&& f) const {
    if (auto* p = std::get_if<PackReader<Layout, Reader>>(&impl_)) return f(*p);
    return f(std::get<RowReader<Layout, Reader>>(impl_));
  }
  std::variant<std::monostate, PackReader<Layout, Reader>, RowReader<Layout, Reader>> impl_;
};

}  // namespace sglang::expert_stream
