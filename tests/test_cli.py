from __future__ import annotations

import typer
from typer.testing import CliRunner

from darija_eval import cli


def test_dotenv_loads_before_dataset_access(monkeypatch) -> None:
    events = []

    def load_environment() -> None:
        events.append("dotenv")

    def load_dataset():
        assert events == ["dotenv"]
        raise typer.Exit()

    monkeypatch.setattr(cli, "load_dotenv", load_environment)
    monkeypatch.setattr(cli, "load_source_dataset", load_dataset)
    result = CliRunner().invoke(cli.app, ["inspect"])
    assert result.exit_code == 0


def test_inspect_prints_raw_statistics_before_rejecting_unknown_label(monkeypatch):
    from datasets import Dataset
    dataset = Dataset.from_dict({"review": ["text"], "label": ["unknown"], "writing_style": ["Arabizi"], "topic": ["it"]})
    monkeypatch.setattr(cli, "load_source_dataset", lambda: dataset)
    result = CliRunner().invoke(cli.app, ["inspect"])
    assert result.exit_code == 2
    assert "Dataset size: 1" in result.output
    assert "'unknown': 1" in result.output
    assert "INVALID: no mapping" in result.output
