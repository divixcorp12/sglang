"""FixedVec, the service's per-request container (plan 2026-09-29-hotpath-zero-overhead Task 11): bounded by the wire
format, never allocating, loud on overflow."""

import subprocess
import textwrap

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.expert_stream_sources import MOE

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_fixed_vec_pushes_assigns_clears_and_throws_on_overflow(tmp_path):
    src = tmp_path / "t.cpp"
    src.write_text(textwrap.dedent(f'''
        #include "{MOE}/expert_stream/host/fixed_vec.h"
        #include <cassert>
        #include <cstdint>
        #include <stdexcept>
        #include <vector>
        using sglang::expert_stream::FixedVec;
        using sglang::expert_stream::listed;
        int main() {{
          FixedVec<int, 3> v; v.push_back(1); v.push_back(2);
          assert(v.size() == 2 && v[1] == 2 && v.back() == 2 && !v.empty());
          int src[] = {{7, 8, 9}}; v.assign(src, src + 3); assert(v.size() == 3 && v[2] == 9);
          std::span<const int> s = v; assert(s.size() == 3 && s[0] == 7);
          assert(listed(v, 8) && !listed(v, 4) && listed(s, 9) && listed(std::vector<int>{{4}}, 4));
          bool threw = false; try {{ v.push_back(4); }} catch (const std::logic_error&) {{ threw = true; }}
          assert(threw && v.size() == 3);
          threw = false; int four[] = {{1, 2, 3, 4}};
          try {{ v.assign(four, four + 4); }} catch (const std::logic_error&) {{ threw = true; }}
          assert(threw && v.size() == 3);
          v.clear(); assert(v.empty() && v.span().empty());
          // Storage and a count, nothing else: no pointer to a heap buffer (plan preflight F12's compile-able bound).
          static_assert(sizeof(FixedVec<int64_t, 3>) == 4 * sizeof(int64_t));
          return 0;
        }}
    '''))
    exe = tmp_path / "t"
    subprocess.run(["c++", "-std=c++20", "-O1", "-o", str(exe), str(src)], check=True)
    subprocess.run([str(exe)], check=True)
