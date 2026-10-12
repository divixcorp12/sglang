"""b0_guards.py: each launch refusal fires and a clean launch passes. CPU only."""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b0_guards  # noqa: E402


def _launch(tmp_path, **over):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "exl3"}})
    )
    args = dict(
        out_dir="/mnt/nvme1/b0/run",
        env={"SGLANG_DSV41_RAM_PREFETCH": "0"},
        arm="dspark-both-misscut",
        model_path=str(model),
        ps_lines=["/usr/bin/python3 capture.py", "bash"],
    )
    args.update(over)
    return args


def test_a_clean_launch_passes(tmp_path):
    b0_guards.refuse_launch(**_launch(tmp_path))


@pytest.mark.parametrize(
    "path", ["/data/models/b0", "/mnt/nvme0/b0", "/mnt/nvme2/b0", "/mnt/nvme1x/b0", "/mnt/nvme1/../nvme0/b0", "b0"]
)
def test_an_output_path_outside_nvme1_is_refused(tmp_path, path):
    with pytest.raises(RuntimeError, match="output path"):
        b0_guards.refuse_launch(**_launch(tmp_path, out_dir=path))


@pytest.mark.parametrize("env", [{"SGLANG_DSV41_RAM_PREFETCH": "1"}, {}])
def test_prefetch_not_off_is_refused(tmp_path, env):
    with pytest.raises(RuntimeError, match="SGLANG_DSV41_RAM_PREFETCH"):
        b0_guards.refuse_launch(**_launch(tmp_path, env=env))


def test_another_arm_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="arm"):
        b0_guards.refuse_launch(**_launch(tmp_path, arm="dspark-both-prefetch"))


@pytest.mark.parametrize(
    "line",
    [
        "python -m sglang.launch_server --model-path x",
        "python analysis/dsv41-drive/dspark/both_cpu_ab.py",
        "/bin/bash analysis/run_arm.sh arm",
    ],
)
def test_a_running_server_or_ab_is_refused(tmp_path, line):
    with pytest.raises(RuntimeError, match="running"):
        b0_guards.refuse_launch(**_launch(tmp_path, ps_lines=["bash", line]))


def test_the_ps_scan_ignores_its_own_grep(tmp_path):
    ps = ['grep -E "launch_server|both_cpu_ab|run_arm.sh"']
    b0_guards.refuse_launch(**_launch(tmp_path, ps_lines=ps))


def test_a_non_directory_model_resolving_to_nvfp4_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="not EXL3"):
        b0_guards.refuse_launch(**_launch(tmp_path, model_path=str(tmp_path / "missing")))
