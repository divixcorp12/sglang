"""CPU-only tests for the JIT build pipeline: ninja generation and the cache.

Everything here runs against synthetic files, so the invariants that matter — a
bad recorded dependency list never causes reuse, differing flags never share a
directory, a moved clone still hits — are checked without a GPU or a compiler.
"""

from __future__ import annotations

import os
import pathlib
import sys

import msgspec
import pytest

from sglang.kernels.jit.utils.compile import cache, ninja
from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
from sglang.kernels.jit.utils.compile.spec import BuildSpec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _fresh_digests():
    cache.clear_digest_cache()
    yield
    cache.clear_digest_cache()


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _spec(**overrides) -> BuildSpec:
    base = dict(
        module_args=("activation", "bf16_t"),
        cpp_files=(),
        cuda_files=(),
        cpp_wrappers=(("run", "Kernel::run"),),
        cuda_wrappers=(),
        cflags=("-O3",),
        cuda_cflags=("-O3",),
        ldflags=(),
        include_paths=(),
        header_only=True,
    )
    base.update(overrides)
    return BuildSpec(**base)


def _build_key(**overrides) -> str:
    spec = _spec(**overrides)
    return cache.compute_build_key(spec, build_file=ninja.generate(spec))


def _entries(paths) -> list:
    out = []
    for path in paths:
        root, relpath = cache._normalize_path(path)
        out.append(
            cache._DepEntry(root=root, relpath=relpath, digest=cache._file_digest(path))
        )
    return out


def _publish_leaf(scope: pathlib.Path, paths, *, module_name="m") -> pathlib.Path:
    """Create a leaf the way commit_build would: name derived from its own list."""
    entries = _entries(paths)
    leaf = scope / f"{cache._DEPS_KEY_PREFIX}{cache._deps_key(entries)}"
    leaf.mkdir(parents=True)
    (leaf / cache._DEPS_FILE).write_bytes(msgspec.json.encode(entries))
    (leaf / f"{module_name}.so").write_bytes(b"")
    return leaf


# --------------------------------------------------------------------------
# Anchor roots
# --------------------------------------------------------------------------


def test_in_tree_paths_normalize_to_an_anchor():
    header = KERNEL_PATH / "include" / "sgl_kernel" / "utils.cuh"
    root, relpath = cache._normalize_path(header)
    assert root == "kernels"
    assert relpath == "include/sgl_kernel/utils.cuh"
    assert cache._resolve_path(root=root, relpath=relpath) == header


def test_unknown_paths_fall_back_to_absolute(tmp_path):
    root, relpath = cache._normalize_path(tmp_path / "elsewhere.h")
    assert root == "abs"
    assert cache._resolve_path(root=root, relpath=relpath) == tmp_path / "elsewhere.h"


def test_unresolvable_anchor_is_a_miss_not_a_crash():
    assert cache._resolve_path(root="pkg:does-not-exist", relpath="x.h") is None


def test_anchor_roots_are_symlink_resolved(tmp_path, monkeypatch):
    """Anchors must be symlink-resolved, since the paths matched against them are.

    `/usr/local/cuda` is a symlink to `/usr/local/cuda-<version>`; an unresolved
    anchor makes every toolkit header miss it and fall through to `sys`, whose
    relpath then carries the CUDA version and breaks reuse across upgrades.

    The toolkit stands in for all of them — one comprehension resolves every
    anchor — and it is faked rather than read off the machine so this still
    guards on the CPU-only runners, which have no toolkit at all.
    """
    from sglang.kernels.jit.utils.compile import toolchain

    versioned = tmp_path / "cuda-12.9"
    versioned.mkdir()
    (tmp_path / "cuda").symlink_to(versioned)
    monkeypatch.setattr(toolchain, "toolkit_home", lambda: tmp_path / "cuda")

    # `__wrapped__` is the undecorated function: the anchors are memoized for
    # the process, and the real ones were already computed by an earlier test.
    roots = dict(cache._anchor_roots.__wrapped__())
    assert roots["toolkit"] == versioned


# --------------------------------------------------------------------------
# build_key — what must and must not change it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"module_args": ("activation", "fp16_t")},
        {"cuda_cflags": ("-O3", "--use_fast_math")},
        {"cflags": ("-O2",)},
        {"ldflags": ("-lfoo",)},
        {"cpp_wrappers": (("run", "Other::run"),)},
        {"include_paths": ("/opt/extra",)},
        {"header_only": False, "cpp_files": ("/tmp/x.cpp",), "cpp_wrappers": ()},
    ],
    ids=["args", "cuda_cflags", "cflags", "ldflags", "wrappers", "includes", "mode"],
)
def test_build_key_separates_every_build_input(overrides):
    """Two builds differing in any of these must never share a directory.

    Sharing one would let a lookup select a leaf produced under different flags,
    which is the failure the whole two-key split exists to prevent.
    """
    assert _build_key(**overrides) != _build_key()


def test_build_key_covers_the_whole_ninja_file():
    """Every flag reaching the compiler reaches the key, because the key is
    taken over the generated build file itself rather than over a hand-listed
    subset of inputs."""
    spec = _spec()
    baseline = cache.compute_build_key(spec, build_file=ninja.generate(spec))
    tampered = ninja.generate(spec).replace("-O3", "-O0")
    assert cache.compute_build_key(spec, build_file=tampered) != baseline


def test_build_key_tracks_direct_source_contents(tmp_path):
    source = _write(tmp_path / "a.cu", "// v1")
    before = _build_key(cuda_files=(str(source),))
    source.write_text("// v2")
    cache.clear_digest_cache()
    assert _build_key(cuda_files=(str(source),)) != before


def test_no_unordered_container_reaches_the_key(monkeypatch):
    """Nothing hashed into a key may iterate in `PYTHONHASHSEED` order.

    A set or dict among the hashed parts would make the same tree key
    differently in two processes: no error, no wrong result, the cache simply
    never hits again. Verified end-to-end by running the key computation under
    several hash seeds; this pins the property cheaply.
    """
    recorded = []
    original = cache._hash_parts
    monkeypatch.setattr(
        cache,
        "_hash_parts",
        lambda parts: recorded.append(list(parts)) or original(recorded[-1]),
    )
    _build_key()

    def walk(value, path="parts"):
        assert not isinstance(value, (set, frozenset, dict)), (
            f"unordered container at {path}: {type(value).__name__}"
        )
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(recorded)


def test_build_key_is_independent_of_install_location():
    """Absolute paths are anchor-normalized, which is what lets a second clone
    of the same tree reuse the first clone's builds."""
    text = ninja.generate(_spec())
    assert str(KERNEL_PATH) not in cache._normalize_text(text)


# --------------------------------------------------------------------------
# deps_key — a leaf that reproduces its own name
# --------------------------------------------------------------------------


def test_leaf_is_found_when_nothing_changed(tmp_path):
    dep = _write(tmp_path / "dep.h", "// v1")
    leaf = _publish_leaf(tmp_path, [dep])
    assert cache.find_prebuilt(scope=tmp_path, module_name="m") == leaf / "m.so"


def test_leaf_is_skipped_when_a_dependency_changed(tmp_path):
    dep = _write(tmp_path / "dep.h", "// v1")
    _publish_leaf(tmp_path, [dep])
    dep.write_text("// v2")
    cache.clear_digest_cache()
    assert cache.find_prebuilt(scope=tmp_path, module_name="m") is None


def test_leaf_survives_an_mtime_only_change(tmp_path):
    dep = _write(tmp_path / "dep.h", "// stable")
    leaf = _publish_leaf(tmp_path, [dep])
    os.utime(dep, (0, 0))
    cache.clear_digest_cache()
    assert cache.find_prebuilt(scope=tmp_path, module_name="m") == leaf / "m.so"


def test_leaf_is_skipped_when_a_dependency_disappeared(tmp_path):
    dep = _write(tmp_path / "dep.h", "// here")
    _publish_leaf(tmp_path, [dep])
    dep.unlink()
    cache.clear_digest_cache()
    assert cache.find_prebuilt(scope=tmp_path, module_name="m") is None


def test_a_leaf_that_does_not_match_its_own_name_is_skipped(tmp_path):
    """The recorded list is verified, not trusted.

    This is what replaces a shared manifest plus a format-version check: a
    truncated, tampered, or foreign list simply fails to reproduce the leaf's
    own name, so no schema field has to be believed.
    """
    dep = _write(tmp_path / "dep.h", "// v1")
    leaf = _publish_leaf(tmp_path, [dep])
    other = _write(tmp_path / "other.h", "// x")
    (leaf / cache._DEPS_FILE).write_bytes(msgspec.json.encode(_entries([dep, other])))
    assert cache.find_prebuilt(scope=tmp_path, module_name="m") is None


@pytest.mark.parametrize(
    "payload",
    [b"", b"not json", b'[["kernels", "a.h"]]'],
    ids=["empty", "garbage", "wrong-shape"],
)
def test_an_unreadable_dependency_list_is_a_miss(tmp_path, payload):
    dep = _write(tmp_path / "dep.h", "// v1")
    leaf = _publish_leaf(tmp_path, [dep])
    (leaf / cache._DEPS_FILE).write_bytes(payload)
    assert cache.find_prebuilt(scope=tmp_path, module_name="m") is None


def test_a_foreign_leaf_does_not_block_a_valid_one(tmp_path):
    """A leaf naming a file that does not exist here is skipped, not fatal.

    A shared, merged manifest could not do this: one unresolvable entry would
    make every lookup fail permanently on this machine.
    """
    dep = _write(tmp_path / "dep.h", "// v1")
    good = _publish_leaf(tmp_path, [dep])

    foreign = tmp_path / f"{cache._DEPS_KEY_PREFIX}{'0' * 16}"
    foreign.mkdir()
    (foreign / cache._DEPS_FILE).write_bytes(
        msgspec.json.encode(
            [cache._DepEntry(root="sys", relpath="include/c++/99/absent.h", digest="x")]
        )
    )
    (foreign / "m.so").write_bytes(b"")
    os.utime(foreign, None)  # make the foreign leaf the newest

    assert cache.find_prebuilt(scope=tmp_path, module_name="m") == good / "m.so"


def test_a_hit_survives_an_unwritable_cache(tmp_path, monkeypatch):
    """Touching the leaf is bookkeeping; it must never turn a hit into a crash.

    The cache root can be a read-only mount, and a prune racing the lookup
    leaves nothing to touch -- either way `os.utime` raises, and before this it
    escaped `find_prebuilt` and took `load_jit` down on an otherwise good hit.
    """
    header = _write(tmp_path / "a.h", "x")
    scope = tmp_path / "scope"
    _publish_leaf(scope, [header], module_name="m")

    def deny(*args, **kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(cache.os, "utime", deny)
    assert cache.find_prebuilt(scope=scope, module_name="m") is not None


def test_missing_library_is_not_a_hit(tmp_path):
    dep = _write(tmp_path / "dep.h", "// v1")
    leaf = _publish_leaf(tmp_path, [dep])
    (leaf / "m.so").unlink()
    assert cache.find_prebuilt(scope=tmp_path, module_name="m") is None


# --------------------------------------------------------------------------
# Commit-side guards
# --------------------------------------------------------------------------


def test_empty_dependency_scan_is_rejected():
    """An empty scan must not be recorded: it would narrow the checked set to
    nothing, the one shape of bad recorded data that could cause reuse."""
    assert not cache._covers_direct_sources(entries=[], direct_sources=["/x/a.cuh"])


def test_scan_missing_a_direct_source_is_rejected(tmp_path):
    other = _write(tmp_path / "other.h", "// x")
    entries = _entries([other])
    assert not cache._covers_direct_sources(
        entries=entries, direct_sources=[str(tmp_path / "a.cuh")]
    )
    assert cache._covers_direct_sources(entries=entries, direct_sources=[str(other)])


def test_build_directory_entries_are_dropped(tmp_path):
    """The generated translation unit is not a dependency of itself.

    Its path is unstable and its contents are already a function of inputs the
    build key covers, so recording it would defeat reuse across clones.
    """
    build_dir = (tmp_path / "build").resolve()
    generated = _write(build_dir / "cuda.cu", "// generated")
    outside = _write(tmp_path / "real.h", "// real")
    entries = cache._to_entries(dependencies=[generated, outside], build_dir=build_dir)
    assert [entry.relpath for entry in entries] == [str(outside)]


def test_publish_loses_the_race_gracefully(tmp_path):
    """Two processes building identical content: the loser adopts the winner's leaf."""
    winner = tmp_path / "leaf"
    winner.mkdir()
    (winner / "m.so").write_bytes(b"winner")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "m.so").write_bytes(b"loser")

    assert cache._publish(staging=staging, leaf=winner) == winner
    assert (winner / "m.so").read_bytes() == b"winner"


def test_build_lock_excludes_a_second_holder(tmp_path):
    """One compile per module variant per node, not one per process.

    Every tensor-parallel rank hits the same cold cache at the same instant.
    Without exclusion each runs a full compile — measured with 8 ranks, all
    eight compiled; with it, one compiled and seven took the cache. The lock
    saves duplicated work only; the atomic rename is what makes publication
    safe, so a missing lock is slow rather than wrong.
    """
    import threading

    from sglang.kernels.jit.utils.compile import loader

    held, release, contender_entered = (threading.Event() for _ in range(3))

    def holder():
        with loader._build_lock(tmp_path):
            held.set()
            release.wait(5)

    def contender():
        with loader._build_lock(tmp_path):
            contender_entered.set()

    first = threading.Thread(target=holder)
    first.start()
    assert held.wait(5)

    second = threading.Thread(target=contender)
    second.start()
    assert not contender_entered.wait(0.3), "entered while the lock was held"

    release.set()
    first.join(5)
    second.join(5)
    assert contender_entered.is_set(), "never entered after the lock was released"


# --------------------------------------------------------------------------
# ninja generation
# --------------------------------------------------------------------------


def test_generated_ninja_keeps_depfiles_on_disk():
    """`deps = gcc` must not be emitted.

    That setting folds each depfile into ninja's binary log and deletes it,
    leaving the cache with nothing to record the dependency closure from.
    """
    text = ninja.generate(_spec(cuda_wrappers=(("run", "K::run"),)))
    assert "deps = gcc" not in text
    assert "depfile = $out.d" in text


def test_generated_ninja_asks_both_compilers_for_dependencies():
    """Both rules must write a depfile on every backend.

    tvm-ffi's HIP branch declared `depfile =` while running a command that never
    produced one, so ROCm builds silently carried no header dependencies.
    """
    text = ninja.generate(_spec(cuda_wrappers=(("run", "K::run"),)))
    compile_commands = [
        line
        for line in text.splitlines()
        if line.startswith("  command = ") and ' -c "$in"' in line
    ]
    assert len(compile_commands) == 2
    assert all('-MD -MF "$out.d"' in line for line in compile_commands)


def test_pure_cpp_module_does_not_link_the_gpu_runtime():
    """A module with no `.cu` sources must not ask the linker for libcudart.

    `ngram_corpus` is five .cpp files and no device code, and it is built on
    CPU-only CI runners that have no CUDA toolkit — linking it there fails with
    `cannot find -lcudart`. tvm-ffi keyed the runtime flags off the presence of
    `.cu` sources for exactly this reason.
    """
    cpu_only = _spec(cpp_files=("/tmp/a.cpp",), cpp_wrappers=(), header_only=False)
    ldflags = next(
        line
        for line in ninja.generate(cpu_only).splitlines()
        if line.startswith("ldflags = ")
    )
    assert "cudart" not in ldflags and "amdhip" not in ldflags

    with_device = _spec(cuda_files=("/tmp/a.cu",), cpp_wrappers=(), header_only=False)
    ldflags = next(
        line
        for line in ninja.generate(with_device).splitlines()
        if line.startswith("ldflags = ")
    )
    assert "cudart" in ldflags or "amdhip" in ldflags


def test_generated_ninja_is_deterministic():
    spec = _spec(cuda_wrappers=(("run", "K::run"),))
    assert ninja.generate(spec) == ninja.generate(spec)


def test_header_only_module_compiles_through_a_generated_wrapper(tmp_path):
    source = str(tmp_path / "kernel.cuh")
    units = _spec(
        cuda_files=(source,), cuda_wrappers=(("run", "K::run"),)
    ).translation_units()
    assert [unit.filename for unit in units] == ["main.cpp", "cuda.cu"]
    generated = next(unit for unit in units if unit.filename == "cuda.cu").source
    assert f'#include "{source}"' in generated
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, (K::run));" in generated
    # The wrapper must include the header defining the macro it uses. Every
    # kernel in tree happens to drag it in transitively, so dropping this would
    # only break whichever future kernel does not.
    assert "#include <tvm/ffi/function.h>" in generated


def test_non_header_only_module_compiles_its_sources_in_place(tmp_path):
    source = str(tmp_path / "kernel.cu")
    units = _spec(
        cuda_files=(source,), cpp_wrappers=(), header_only=False
    ).translation_units()
    assert [(unit.filename, unit.source, unit.is_cuda) for unit in units] == [
        (source, None, True)
    ]


# --------------------------------------------------------------------------
# Per-source flags (load_jit's source_cflags)
# --------------------------------------------------------------------------


def _two_sources(**overrides) -> BuildSpec:
    return _spec(
        cpp_files=("/src/host.cpp", "/src/nvfp4.c"),
        cpp_wrappers=(),
        header_only=False,
        **overrides,
    )


def _compile_flags(text: str) -> dict:
    """Each object's cxxflags: its own `cxxflags` binding when it has one, else the module's."""
    module = next(
        line.split(" = ", 1)[1] for line in text.splitlines() if line.startswith("cxxflags = ")
    )
    flags, current = {}, None
    for line in text.splitlines():
        if line.startswith("build ") and ".o:" in line:
            current = line.split(":")[0].removeprefix("build ")
            flags[current] = module
        elif current and line.startswith("  cxxflags = "):
            flags[current] = line.removeprefix("  cxxflags = ")
        elif not line.startswith("  "):
            current = None
    return flags


def test_source_cflags_replace_the_module_flags_for_that_file_only():
    """A C file gets `-x c -std=c11` and none of the module's `-O3`; the other file and the toolchain's own flags are
    untouched."""
    spec = _two_sources(source_cflags=(("/src/nvfp4.c", ("-x", "c", "-std=c11", "-Ofast")),))
    flags = _compile_flags(ninja.generate(spec))
    host, c_file = flags["host_0.o"].split(), flags["nvfp4_1.o"].split()
    assert "-O3" in host and "-x" not in host
    assert c_file[c_file.index("-x") : c_file.index("-x") + 4] == ["-x", "c", "-std=c11", "-Ofast"]
    assert "-O3" not in c_file
    assert "-fPIC" in c_file


def test_source_cflags_reach_the_build_key():
    plain = _build_key(cpp_files=("/src/host.cpp", "/src/nvfp4.c"), cpp_wrappers=(), header_only=False)
    keys = {
        plain,
        *(
            cache.compute_build_key(spec, build_file=ninja.generate(spec))
            for spec in (
                _two_sources(source_cflags=(("/src/nvfp4.c", ("-Ofast",)),)),
                _two_sources(source_cflags=(("/src/nvfp4.c", ("-Ofast", "-DX=1")),)),
                _two_sources(source_cflags=(("/src/host.cpp", ("-Ofast",)),)),
            )
        ),
    }
    assert len(keys) == 4


def test_source_cflags_are_refused_where_they_cannot_apply():
    from sglang.kernels.jit.utils.compile.loader import _resolved_source_cflags

    flags = {"a.cpp": ["-O2"]}
    with pytest.raises(ValueError, match="header_only=False"):
        _resolved_source_cflags(flags, cpp_files=["a.cpp"], header_only=True)
    with pytest.raises(ValueError, match="not in cpp_files"):
        _resolved_source_cflags(flags, cpp_files=["b.cpp"], header_only=False)


# --------------------------------------------------------------------------
# Compiler launcher (SGLANG_JIT_COMPILER_LAUNCHER)
# --------------------------------------------------------------------------


def _written_build_file(spec, tmp_path, monkeypatch, launcher):
    """What `ninja.build` writes for *spec* under *launcher*, ninja itself faked."""
    import subprocess

    from sglang.srt.environ import envs

    monkeypatch.setattr(
        ninja.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""),
    )
    with envs.SGLANG_JIT_COMPILER_LAUNCHER.override(launcher):
        ninja.build(spec=spec, build_dir=tmp_path, build_file=ninja.generate(spec))
    return (tmp_path / "build.ninja").read_text()


def test_launcher_does_not_change_the_build_key():
    """ccache changes how an object is made, not what it is: turning it on must
    not cold-start the JIT cache, nor give its builds their own directories."""
    from sglang.srt.environ import envs

    spec = _spec(cuda_wrappers=(("run", "K::run"),))
    baseline_text = ninja.generate(spec)
    baseline = cache.compute_build_key(spec, build_file=baseline_text)
    with envs.SGLANG_JIT_COMPILER_LAUNCHER.override("ccache"):
        launched_text = ninja.generate(spec)
        launched = cache.compute_build_key(spec, build_file=launched_text)
    assert launched_text == baseline_text
    assert launched == baseline


def test_unset_or_empty_launcher_writes_the_generated_file(tmp_path, monkeypatch):
    spec = _spec(cuda_wrappers=(("run", "K::run"),))
    for launcher, build_dir in ((None, tmp_path / "unset"), ("", tmp_path / "empty")):
        assert _written_build_file(spec, build_dir, monkeypatch, launcher) == (
            ninja.generate(spec)
        )
        assert not (build_dir / ninja._SOURCE_LINK).exists()


def test_launcher_prefixes_both_compiles_and_not_the_link(tmp_path, monkeypatch):
    spec = _spec(cuda_wrappers=(("run", "K::run"),))
    text = _written_build_file(spec, tmp_path, monkeypatch, "ccache --flag")
    assert "\nlauncher = ccache --flag\n" in text
    commands = [line for line in text.splitlines() if line.startswith("  command = ")]
    compiles = [line for line in commands if ' -c "$in"' in line]
    assert len(compiles) == 2
    assert all(line.startswith("  command = $launcher $") for line in compiles)
    (link,) = [line for line in commands if line not in compiles]
    assert "$launcher" not in link


def test_launched_build_reaches_the_tree_through_a_link(tmp_path, monkeypatch):
    """The same module built from two clones must give ccache the same command
    and the same preprocessed text, so no path into the clone may survive."""
    source = str(KERNEL_PATH / "csrc" / "elementwise" / "activation.cuh")
    spec = _spec(
        cuda_files=(source,),
        cuda_wrappers=(("run", "K::run"),),
        include_paths=(str(KERNEL_PATH / "include"),),
    )
    text = _written_build_file(spec, tmp_path, monkeypatch, "ccache")
    wrapper = (tmp_path / "cuda.cu").read_text()
    root = str(ninja._SOURCE_ROOT)
    assert root not in text and root not in wrapper
    assert f"-I{ninja._SOURCE_LINK}/sglang/kernels/jit/include" in text
    assert f'#include "{ninja._SOURCE_LINK}/sglang/kernels/jit/csrc/' in wrapper
    assert (tmp_path / ninja._SOURCE_LINK).resolve() == ninja._SOURCE_ROOT.resolve()


def test_dependencies_listed_through_the_link_resolve_to_the_tree(tmp_path):
    """A launched build's depfiles name files relative to the build directory;
    the dependency list must still record the real files, or the leaf would
    never cover its direct sources and would not be published."""
    build_dir = tmp_path / ".staging-x"
    build_dir.mkdir()
    (build_dir / ninja._SOURCE_LINK).symlink_to(ninja._SOURCE_ROOT)
    header = KERNEL_PATH / "include" / "sgl_kernel" / "utils.cuh"
    relative = f"{ninja._SOURCE_LINK}/{header.relative_to(ninja._SOURCE_ROOT)}"
    (build_dir / "main_0.o.d").write_text(f"main_0.o: main.cpp {relative}\n")
    entries = cache._to_entries(
        dependencies=ninja.scan_dependencies(build_dir), build_dir=build_dir
    )
    assert [(e.root, e.relpath) for e in entries] == [
        ("kernels", "include/sgl_kernel/utils.cuh")
    ]


# --------------------------------------------------------------------------
# depfile parsing
# --------------------------------------------------------------------------


def test_depfile_parsing_handles_continuations_and_escaped_spaces():
    text = "cuda_0.o: /a/cuda.cu \\\n  /a/with\\ space.cuh \\\n  /b/plain.h\n"
    assert ninja._parse_depfile(text) == [
        "/a/cuda.cu",
        "/a/with space.cuh",
        "/b/plain.h",
    ]


def test_depfile_parsing_ignores_a_target_with_no_prerequisites():
    assert ninja._parse_depfile("a.o:\n") == []


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def test_layout_is_readable_and_scoped_by_build_key():
    scope = cache.build_key_dir(
        module_name="sgl_kernel_jit_activation_bf16_t", build_key="abc123"
    )
    assert scope.name == "build-abc123"
    assert scope.parent.name == "sgl_kernel_jit_activation_bf16_t"
    assert scope.parent.parent.name == cache._target_tag()


def test_module_name_is_derived_from_the_args():
    assert _spec().module_name == "sgl_kernel_jit_activation_bf16_t"


def test_relative_sources_resolve_against_csrc():
    from sglang.kernels.jit.utils.compile.spec import resolve_sources

    assert resolve_sources(["elementwise/activation.cuh"]) == (
        str(KERNEL_PATH / "csrc" / "elementwise" / "activation.cuh"),
    )
    assert resolve_sources(["/usr/include/stdio.h"]) == ("/usr/include/stdio.h",)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
