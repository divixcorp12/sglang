"""Aligned matrix partitions must cover every output tile exactly once."""

import os
import subprocess

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import MOE

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_row_weighted_tile_ownership_and_known_skew(tmp_path):
    header = MOE.parent / "exl3/optimized/tile_assignment.hpp"
    source = tmp_path / "ownership.cpp"
    source.write_text(
        r'''
#include "tile_assignment.hpp"
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
using namespace sglang::exl3_cpu;

template<int Unit>
std::vector<int> check(const std::vector<int>& rows, int tiles, int workers, bool weighted) {
  std::vector<int> owners(rows.size() * tiles, -1), work(workers);
  for (int w = 0; w < workers; ++w) {
    auto visit = [&](int j, int a, int b) {
      assert(j >= 0 && j < int(rows.size()));
      // The original partition can issue an empty range when workers exceed groups.
      assert(a >= 0 && a <= b && b <= tiles && a % Unit == 0 && b % Unit == 0);
      for (int t = a; t < b; ++t) {
        assert(owners[j * tiles + t] == -1);
        owners[j * tiles + t] = w;
        work[w] += rows[j];
      }
    };
    if (weighted) assign_row_weighted_gemvs<Unit>(w, workers, rows.size(), tiles,
                                                [&](int j) { return rows[j]; }, visit);
    else assign_gemvs<Unit>(w, workers, rows.size(), tiles, visit);
  }
  for (int owner : owners) assert(owner >= 0 && owner < workers);
  assert(std::accumulate(work.begin(), work.end(), 0) ==
         tiles * std::accumulate(rows.begin(), rows.end(), 0));
  // Uniform shapes preserve the original ownership, including whole-GEMV striding.
  if (weighted && !rows.empty() && std::all_of(rows.begin(), rows.end(), [&](int m) { return m == rows[0]; })) {
    for (int w = 0; w < workers; ++w)
      assign_gemvs<Unit>(w, workers, rows.size(), tiles, [&](int j, int a, int b) {
        for (int t = a; t < b; ++t) assert(owners[j * tiles + t] == w);
      });
  }
  return work;
}

template<int Unit> void exercise() {
  std::mt19937 rng(20261007);
  for (int workers : {1, 3, 10, 16, 64})
    for (int tiles : {8, 144, 320})
      for (int chunks : {0, 1, 2, 10, 17, 65, 257}) {
        for (int uniform : {0, 1, 4}) {
          std::vector<int> rows(chunks);
          for (int& m : rows) m = uniform ? uniform : 1 + rng() % 8;
          check<Unit>(rows, tiles, workers, false);
          auto work = check<Unit>(rows, tiles, workers, true);
          if (uniform == 0 && !rows.empty()) {
            const double mean = double(std::accumulate(work.begin(), work.end(), 0)) / workers;
            const int atom = Unit * *std::max_element(rows.begin(), rows.end());
            for (int cost : work) assert(std::abs(cost - mean) <= atom + 1);
          }
        }
      }
  // Captured six-row / ten-chunk shape: one expert serves two rows, nine serve one.
  // Gate and up are independent GEMVs with the same chunk multiplicities.
  std::vector<int> rows;
  for (int j = 0; j < 10; ++j) for (int gu = 0; gu < 2; ++gu) rows.push_back(j == 0 ? 2 : 1);
  auto a = check<Unit>(rows, 144, 10, false);
  auto b = check<Unit>(rows, 144, 10, true);
  assert(*std::max_element(a.begin(), a.end()) == 576);
  assert(*std::max_element(b.begin(), b.end()) <= 336);
}
int main() { exercise<2>(); exercise<8>(); }
'''
    )
    executable = tmp_path / "ownership"
    subprocess.run(
        [os.environ.get("CXX", "c++"), "-std=c++20", "-O2", "-I", str(header.parent), str(source), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_plan_assignment_is_chosen_per_call(tmp_path):
    """The plan's entry takes the layer's choice at run time: false keeps the original mapping, true balances by rows.

    One kernel serves the target's and the draft's layers, so the choice travels with the layer (its params), not
    the build."""
    header = MOE.parent / "exl3/optimized/tile_assignment.hpp"
    source = tmp_path / "per_call.cpp"
    source.write_text(
        r'''
#include "tile_assignment.hpp"
#include <algorithm>
#include <cassert>
#include <vector>
using namespace sglang::exl3_cpu;

// Per-worker work units and ownership for the captured ten-chunk shape (chunk 0 holds two rows), gate and up.
static std::vector<int> run(bool weighted, std::vector<int>* owners_out) {
  std::vector<int> rows;
  for (int j = 0; j < 10; ++j) for (int gu = 0; gu < 2; ++gu) rows.push_back(j == 0 ? 2 : 1);
  const int tiles = 144, workers = 10;
  std::vector<int> owners(rows.size() * tiles, -1), work(workers);
  for (int w = 0; w < workers; ++w)
    assign_plan_gemvs<8>(w, workers, rows.size(), tiles, [&](int j) { return rows[j]; },
                         [&](int j, int a, int b) { for (int t = a; t < b; ++t) { owners[j * tiles + t] = w; work[w] += rows[j]; } },
                         weighted);
  for (int owner : owners) assert(owner >= 0);
  if (owners_out) *owners_out = owners;
  return work;
}

int main() {
  std::vector<int> original_owners, weighted_owners;
  auto a = run(false, &original_owners);
  auto b = run(true, &weighted_owners);
  assert(*std::max_element(a.begin(), a.end()) == 576);
  assert(*std::max_element(b.begin(), b.end()) <= 336);
  assert(original_owners != weighted_owners);
  // false is the original partition, tile for tile.
  std::vector<int> owners(original_owners.size(), -1);
  for (int w = 0; w < 10; ++w)
    assign_gemvs<8>(w, 10, 20, 144, [&](int j, int a, int b) { for (int t = a; t < b; ++t) owners[j * 144 + t] = w; });
  assert(owners == original_owners);
  return 0;
}
'''
    )
    executable = tmp_path / "per_call"
    subprocess.run(
        [os.environ.get("CXX", "c++"), "-std=c++20", "-O2", "-I", str(header.parent), str(source), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
