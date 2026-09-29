// Row tables: the per-expert segment map and the tables_from build of it.
#pragma once

#include "../row_layout.h"
#include "file_reader.h"  // RegisteredRegion
#include "reader_base.h"
#include <sys/uio.h>

namespace sglang {
namespace expert_stream {

struct Segment {
  int64_t name;
  int64_t dst;
  int64_t src;
  int64_t bytes;
};

// One part of a row's aligned read: `length` bytes of `file` at `offset`, into the row's bounce
// slot at `dest`. A row is `parts` extents (one per mirror root); a zero-length one reads nothing.
struct Read {
  int64_t file;
  int64_t offset;
  int64_t length;
  int64_t dest;
};

struct Tables {
  int64_t layers = 0;
  int64_t experts = 0;
  int64_t parts = 1;
  int64_t slot_bytes = 0;
  std::vector<std::string> paths;
  std::vector<std::string> source_paths;  // paths[f]'s source shard: a mirror is a copy of it
  std::vector<int64_t> file_sizes;        // the SOURCE size of every file, mirrors included
  std::vector<Read> extents;              // [layers][experts][parts]
  std::vector<int64_t> starts;            // [layers][experts]: where the row starts in its aligned superset
  std::vector<Segment> segments;
  int64_t need_end = 0;  // the row's last needed byte + 1, from its start: max(src + bytes) over segments
  std::vector<std::vector<uint8_t*>> slabs;
  std::vector<int64_t> row_bytes;
  // One per named slab tensor retained by Python tables.keepalive, with its row size: registered as row-aligned chunks.
  std::vector<RegisteredRegion> buffer_regions;
  // Row images (plan 2026-09-24-dsv41-row-images; always on since the packed path's removal): the files are
  // exl3_row_image layer files, not checkpoint shards. A row's needed bytes are then its image, [0, need_end) of
  // the extents' destination coordinates, and the segments tile it in source order, so every image byte has
  // exactly one slab destination. The reader reads straight into the slab rows (RowReader, direct mode): no bounce.
  // RowReader is the only reader and refuses tables without it (plan 2026-09-29-hotpath-zero-overhead D4); "bounce
  // slot" in the checks below names a row's pipeline slot, not memory.
  bool images = false;
};

inline std::vector<int32_t> ids_of(TensorView tensor) {
  const auto* data = static_cast<const int64_t*>(tensor.data_ptr());
  return std::vector<int32_t>(data, data + tensor.size(0));
}

// Row images: the reader scatters each read straight into slab rows (RowReader::image_iovecs), which is only right
// when every byte a read returns has exactly one destination and the row's reads return exactly its image. So: the
// row starts at 0 of its reads, the segments tile [0, need_end) in source order inside their names' slab rows, and
// each row's reading parts tile [0, need_end) in part order. The 512-byte alignment O_DIRECT needs is
// RowReader::check_image_alignment's, at open.
template <ExpertRowLayout Layout>
inline void check_image_tables(const Tables& t) {
  const std::string prefix = error_prefix<Layout>();
  for (int64_t start : t.starts) {
    if (start != 0) throw std::runtime_error(prefix + "row images start every row at 0 of its reads");
  }
  int64_t cursor = 0;
  for (const Segment& s : t.segments) {
    if (s.name < 0 || s.name >= static_cast<int64_t>(t.row_bytes.size()) || s.src != cursor || s.bytes <= 0 ||
        s.dst < 0 || s.dst + s.bytes > t.row_bytes[s.name]) {
      throw std::runtime_error(
          prefix + "row-image segments must tile the image in source order, each inside its slab row");
    }
    cursor += s.bytes;
  }
  // ... and on the destination side, each name's segments tile its slab row: two segments landing on the same slab
  // bytes would pass the source check and silently keep whichever read landed last.
  for (int64_t name = 0; name < static_cast<int64_t>(t.row_bytes.size()); ++name) {
    std::vector<std::pair<int64_t, int64_t>> spans;
    for (const Segment& s : t.segments) {
      if (s.name == name) spans.emplace_back(s.dst, s.dst + s.bytes);
    }
    std::sort(spans.begin(), spans.end());
    int64_t at = 0;
    for (const auto& span : spans) {
      if (span.first != at) throw std::runtime_error(prefix + "row-image segments must tile each slab row once");
      at = span.second;
    }
    if (at != t.row_bytes[name]) throw std::runtime_error(prefix + "row-image segments must tile each slab row once");
  }
  const size_t parts = static_cast<size_t>(t.parts);
  for (size_t base = 0; base < t.extents.size(); base += parts) {
    int64_t at = 0;
    for (size_t p = 0; p < parts; ++p) {
      const Read& e = t.extents[base + p];
      if (e.length <= 0) continue;
      if (e.dest != at) throw std::runtime_error(prefix + "a row image's parts must tile it in part order");
      at += e.length;
    }
    if (at != t.need_end) {
      throw std::runtime_error(prefix + "a row image's parts must read exactly its image, never its padding");
    }
  }
}

template <ExpertRowLayout Layout>
inline Tables tables_from(
    TensorView extents,
    TensorView starts,
    TensorView file_sizes,
    TensorView segments,
    TensorView slabs,
    TensorView row_bytes,
    TensorView buffer_regions,
    const std::string& paths,
    const std::string& source_paths,
    int64_t slot_bytes,
    int64_t row_images) {
  const std::string prefix = error_prefix<Layout>();
  if (slabs.size(1) != kNumNames<Layout> || row_bytes.size(0) != kNumNames<Layout>) {
    throw std::runtime_error(
        prefix + "slabs and row_bytes must have " + std::to_string(kNumNames<Layout>) + " names (the layout's), got " +
        std::to_string(slabs.size(1)) + " and " + std::to_string(row_bytes.size(0)));
  }
  Tables t;
  const auto* regions = static_cast<const int64_t*>(buffer_regions.data_ptr());
  for (int64_t i = 0; i < buffer_regions.size(0); ++i) {
    const int64_t base = regions[3 * i], bytes = regions[3 * i + 1], row = regions[3 * i + 2];
    if (base <= 0 || bytes <= 0 || row <= 0 || bytes % row != 0 ||
        static_cast<uint64_t>(base) + static_cast<uint64_t>(bytes) < static_cast<uint64_t>(base))
      throw std::runtime_error(prefix + "invalid I/O registration buffer region");
    t.buffer_regions.push_back(
        {reinterpret_cast<void*>(static_cast<uintptr_t>(base)), static_cast<size_t>(bytes), static_cast<size_t>(row)});
  }
  t.images = row_images != 0;
  t.layers = extents.size(0);
  t.experts = extents.size(1);
  t.parts = extents.size(2);
  t.slot_bytes = slot_bytes;
  const auto split_lines = [](const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (true) {
      const size_t end = text.find('\n', start);
      lines.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
      if (end == std::string::npos) break;
      start = end + 1;
    }
    return lines;
  };
  t.paths = split_lines(paths);
  t.source_paths = split_lines(source_paths);
  if (t.source_paths.size() != t.paths.size() || static_cast<size_t>(file_sizes.size(0)) != t.paths.size()) {
    throw std::runtime_error(prefix + "every file needs a size and the path of the source shard it copies");
  }
  const auto* sizes = static_cast<const int64_t*>(file_sizes.data_ptr());
  t.file_sizes.assign(sizes, sizes + file_sizes.size(0));
  const auto* extent_data = static_cast<const int64_t*>(extents.data_ptr());
  t.extents.resize(static_cast<size_t>(t.layers * t.experts * t.parts));
  for (size_t i = 0; i < t.extents.size(); ++i) {
    t.extents[i] = Read{extent_data[4 * i], extent_data[4 * i + 1], extent_data[4 * i + 2], extent_data[4 * i + 3]};
    // The reader writes each extent into its row's bounce slot and reads its file without
    // checking again, so a table that would write outside the slot or name no file is refused here.
    const Read& e = t.extents[i];
    if (e.file < 0 || e.file >= static_cast<int64_t>(t.paths.size()) ||
        e.file >= static_cast<int64_t>(t.file_sizes.size()) || e.offset < 0 || e.length < 0 || e.dest < 0 ||
        e.dest + e.length > slot_bytes) {
      throw std::runtime_error(prefix + "an extent names no file or falls outside its bounce slot");
    }
  }
  // The EOF guard (ReaderCore::admit_batch) decides a whole row from ONE of its parts: it reads that
  // part's `offset - dest` as the row's aligned base and its file size as the row's file size. Both are
  // true by construction of today's builder - exl3_ram_miss.py repeats one source size across all the
  // parts of a shard, and the mirror layout puts two files under a row only as two copies of the SAME
  // shard - but nothing in this file pinned either, and a builder that ever gave a row parts from
  // genuinely different files would arm the guard to clear a row against the wrong size. Checked here,
  // once per table, rather than per read.
  for (size_t base = 0; base + static_cast<size_t>(t.parts) <= t.extents.size(); base += static_cast<size_t>(t.parts)) {
    const Read* head = nullptr;
    for (int64_t p = 0; p < t.parts && head == nullptr; ++p) {
      if (t.extents[base + static_cast<size_t>(p)].length > 0) head = &t.extents[base + static_cast<size_t>(p)];
    }
    if (head == nullptr) continue;  // a row nothing reads never reaches the guard
    for (int64_t p = 0; p < t.parts; ++p) {
      const Read& e = t.extents[base + static_cast<size_t>(p)];
      // Only the reading parts: a zero-length part is never submitted and its fields are unused, so
      // requiring anything of them would over-constrain the builder for no gain.
      if (e.length <= 0) continue;
      if (t.file_sizes[e.file] != t.file_sizes[head->file] || e.offset - e.dest != head->offset - head->dest) {
        throw std::runtime_error(
            prefix +
            "a row's parts disagree on their aligned base or their file size, so the "
            "EOF guard cannot decide the row from part 0");
      }
    }
  }
  const auto* start_data = static_cast<const int64_t*>(starts.data_ptr());
  t.starts.assign(start_data, start_data + t.layers * t.experts);
  const auto* segment_data = static_cast<const int64_t*>(segments.data_ptr());
  t.segments.resize(static_cast<size_t>(segments.size(0)));
  for (size_t i = 0; i < t.segments.size(); ++i) {
    t.segments[i] =
        Segment{segment_data[4 * i], segment_data[4 * i + 1], segment_data[4 * i + 2], segment_data[4 * i + 3]};
    if (t.segments[i].name < 0 || t.segments[i].name >= kNumNames<Layout>) {
      throw std::runtime_error(
          prefix + "a segment names a tensor outside the layout's " + std::to_string(kNumNames<Layout>) + " names");
    }
    t.need_end = std::max(t.need_end, t.segments[i].src + t.segments[i].bytes);
  }
  const auto* slab_data = static_cast<const int64_t*>(slabs.data_ptr());
  const int64_t names = slabs.size(1);
  t.slabs.resize(static_cast<size_t>(t.layers));
  for (int64_t row = 0; row < t.layers; ++row) {
    for (int64_t name = 0; name < names; ++name) {
      t.slabs[row].push_back(reinterpret_cast<uint8_t*>(static_cast<intptr_t>(slab_data[row * names + name])));
    }
  }
  const auto* rows = static_cast<const int64_t*>(row_bytes.data_ptr());
  t.row_bytes.assign(rows, rows + row_bytes.size(0));
  if (t.images) check_image_tables<Layout>(t);
  return t;
}

}  // namespace expert_stream
}  // namespace sglang
