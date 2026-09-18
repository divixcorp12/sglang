"""Pure helpers of the exllamav3 extension loader (no build, no GPU)."""

import os

import pytest

from sglang.srt.layers.quantization import exl3_ext
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_sources_are_sorted_c_cpp_cu_only(tmp_path):
    for rel in ("b.cu", "a.cpp", "sub/c.c", "sub/d.cuh", "e.h", "sub/f.py"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    got = [os.path.relpath(p, tmp_path) for p in exl3_ext.extension_sources(str(tmp_path))]
    assert got == ["a.cpp", "b.cu", os.path.join("sub", "c.c")]


def test_wrong_commit_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(exl3_ext, "_checkout_commit", lambda src: "deadbeef")
    with pytest.raises(RuntimeError, match="expected exllamav3"):
        exl3_ext._checked_ext_dir(str(tmp_path))


def test_missing_src_raises(monkeypatch):
    monkeypatch.setattr(exl3_ext.envs.SGLANG_EXL3_SRC, "get", lambda: "")
    exl3_ext.exl3_ext.cache_clear()
    with pytest.raises(RuntimeError, match="SGLANG_EXL3_SRC"):
        exl3_ext.exl3_ext()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
