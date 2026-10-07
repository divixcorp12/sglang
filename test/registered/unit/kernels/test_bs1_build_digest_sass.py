"""The BS1 digest hashes what a kernel executes, not how cuobjdump pads it: it pads each line's trailing encoding comment
to the widest instruction in the module, so a change to one kernel re-pads all the others (CPU only, no build)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41"))

from bs1_build_digest import _sass_digests  # noqa: E402

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _listing(*instructions: tuple[str, str], pad: int) -> str:
    """A cuobjdump -sass listing of one function; every instruction's encoding comment starts at column `pad`."""
    body = "\n".join(f"        /*{addr}*/  {text}".ljust(pad) + f" /* 0x00000000deadbeef */" for addr, text in instructions)
    return f"\tFunction : _Z6kernelv\n\t.headerflags\t@\"EF_CUDA_SM120\"\n{body}\n"


LDG = ("0010", "LDG.E.128.STRONG.SYS R32, desc[UR12][R8.64+0x4190] ;")
LDC = ("0000", "LDC R1, c[0x0][0x37c] ;")


def test_comment_padding_does_not_change_the_digest():
    assert _sass_digests(_listing(LDC, LDG, pad=70)) == _sass_digests(_listing(LDC, LDG, pad=71))


def test_runs_of_spaces_inside_an_instruction_do_not_change_the_digest():
    spaced = ("0010", "LDG.E.128.STRONG.SYS   R32,  desc[UR12][R8.64+0x4190] ;")
    assert _sass_digests(_listing(LDC, LDG, pad=70)) == _sass_digests(_listing(LDC, spaced, pad=70))


def test_an_operand_change_changes_the_digest():
    wider = ("0010", "LDG.E.128.STRONG.SYS R32, desc[UR12][R16.64+0x4190] ;")
    assert _sass_digests(_listing(LDC, LDG, pad=70)) != _sass_digests(_listing(LDC, wider, pad=70))
