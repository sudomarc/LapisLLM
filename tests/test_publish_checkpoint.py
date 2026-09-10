from pathlib import Path

import torch

from scripts import publish_checkpoint


def test_publish_copies_verified_checkpoint_and_tokenizer(tmp_path, monkeypatch) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    source = checkpoint_root / "colab-runs" / "run-001" / "checkpoint.pt"
    source.parent.mkdir(parents=True)
    tokenizer = source.parent / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    torch.save({"model_state_dict": {}, "config": {}}, source)

    monkeypatch.setattr(publish_checkpoint, "CHECKPOINT_ROOT", checkpoint_root)
    monkeypatch.setattr(
        publish_checkpoint,
        "LATEST_CHECKPOINT",
        checkpoint_root / "latest.pt",
    )
    monkeypatch.setattr(
        publish_checkpoint,
        "LATEST_TOKENIZER",
        checkpoint_root / "tokenizer",
    )

    publish_checkpoint.publish(source)

    assert (checkpoint_root / "latest.pt").is_file()
    assert (checkpoint_root / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8") == "{}\n"
    assert Path(source).read_bytes() == (checkpoint_root / "latest.pt").read_bytes()
