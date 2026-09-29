// A fixed-capacity vector for the service's per-request state (plan 2026-09-29-hotpath-zero-overhead Task 11): the
// wire format bounds every per-request list (kMaxIds need and protect ids, kLeaseLanes lanes), so nothing on the
// request path needs the heap. Overflow throws: it means a caller broke that bound, never a valid request.
#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <span>
#include <stdexcept>

namespace sglang::expert_stream {

template <class T, size_t N>
class FixedVec {
 public:
  void push_back(const T& value) {
    if (n_ == N) overflow();
    data_[n_++] = value;
  }
  // Replaces the contents. A range past N throws before anything is changed.
  template <class It>
  void assign(It first, It last) {
    if (std::distance(first, last) > static_cast<std::ptrdiff_t>(N)) overflow();
    n_ = 0;
    for (; first != last; ++first)
      data_[n_++] = *first;
  }
  void clear() {
    n_ = 0;
  }
  size_t size() const {
    return n_;
  }
  bool empty() const {
    return n_ == 0;
  }
  T* begin() {
    return data_;
  }
  T* end() {
    return data_ + n_;
  }
  const T* begin() const {
    return data_;
  }
  const T* end() const {
    return data_ + n_;
  }
  T& operator[](size_t i) {
    return data_[i];
  }
  const T& operator[](size_t i) const {
    return data_[i];
  }
  T& back() {
    return data_[n_ - 1];
  }
  std::span<const T> span() const {
    return {data_, n_};
  }
  operator std::span<const T>() const {
    return span();
  }

 private:
  [[noreturn]] static void overflow() {
    throw std::logic_error("expert stream: a per-request list exceeded its wire-format bound");
  }
  T data_[N]{};
  size_t n_ = 0;
};

// Membership in any contiguous id list (FixedVec, std::vector, std::span).
template <class Ids, class Id>
bool listed(const Ids& ids, Id id) {
  return std::find(std::begin(ids), std::end(ids), id) != std::end(ids);
}

}  // namespace sglang::expert_stream
