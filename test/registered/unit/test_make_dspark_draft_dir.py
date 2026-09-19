import json
import torch
from safetensors.torch import save_file, load_file
from scripts.dsv41.make_dspark_draft_dir import make_draft_dir


def _write(tmp_path):
    src, tgt, out = tmp_path / "src", tmp_path / "tgt", tmp_path / "out"
    src.mkdir(); tgt.mkdir()
    save_file({"layers.0.x": torch.zeros(2), "mtp.0.a": torch.ones(3)}, src / "s1.safetensors")
    save_file({"mtp.1.b": torch.full((4,), 2.0)}, src / "s2.safetensors")
    json.dump({"weight_map": {"layers.0.x": "s1.safetensors", "mtp.0.a": "s1.safetensors",
                              "mtp.1.b": "s2.safetensors"}},
              open(src / "model.safetensors.index.json", "w"))
    json.dump({"num_nextn_predict_layers": 0, "dspark_block_size": 5, "quantization_config": {"quant_method": "exl3"}},
              open(tgt / "config.json", "w"))
    json.dump({"text_config": {"num_nextn_predict_layers": 3}}, open(src / "config.json", "w"))
    return src, tgt, out


def test_copies_only_mtp_tensors_and_restores_nextn(tmp_path):
    src, tgt, out = _write(tmp_path)
    assert make_draft_dir(str(src), str(tgt), str(out)) == 2
    t = load_file(out / "model.safetensors")
    assert set(t) == {"mtp.0.a", "mtp.1.b"} and torch.equal(t["mtp.1.b"], torch.full((4,), 2.0))
    cfg = json.load(open(out / "config.json"))
    assert cfg["num_nextn_predict_layers"] == 3 and cfg["dspark_block_size"] == 5
    idx = json.load(open(out / "model.safetensors.index.json"))["weight_map"]
    assert idx == {"mtp.0.a": "model.safetensors", "mtp.1.b": "model.safetensors"}
