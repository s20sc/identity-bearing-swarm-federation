"""Plan the portable runner without executing experiments or changing traces."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"
    spec = importlib.util.spec_from_file_location("portable_run_all", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(module, "_await_stable_machine", lambda: ([], 1.0))
    calls = []
    monkeypatch.setattr(module, "_run", lambda step: (calls.append(step) or str(step), 0.0))
    return module, calls


def test_standalone_default_skips_private_manuscript_gate(runner, caplog, monkeypatch):
    module, calls = runner
    monkeypatch.setattr(module.sys, "argv", ["run_all.py"])
    caplog.set_level("INFO")
    module.main()
    assert calls == list(module.EXPERIMENTS + module.AGGREGATORS)
    assert not any("check_manuscript_numbers.py" in step for step in calls)
    assert "Skipping manuscript" in caplog.text


def test_explicit_manuscript_is_last_gate(runner, tmp_path):
    module, calls = runner
    manuscript = tmp_path / "paper with spaces.tex"
    manuscript.write_text("test manuscript")
    module.main(["--manuscript", str(manuscript)])
    assert calls[:-1] == list(module.EXPERIMENTS + module.AGGREGATORS)
    assert calls[-1] == ("check_manuscript_numbers.py", str(manuscript.resolve()))


@pytest.mark.parametrize("kind", ["missing", "directory", "wrong_suffix"])
def test_invalid_manuscript_stops_before_work(runner, tmp_path, monkeypatch, kind):
    module, calls = runner
    path = tmp_path / ("paper.txt" if kind == "wrong_suffix" else "paper.tex")
    if kind == "directory":
        path.mkdir()
    elif kind == "wrong_suffix":
        path.write_text("not TeX")
    monkeypatch.setattr(module, "_await_stable_machine", lambda: pytest.fail("probe started"))
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("subprocess started"))
    with pytest.raises(SystemExit) as error:
        module.main(["--manuscript", str(path)])
    assert error.value.code != 0
    assert calls == []
    assert not module.REPORTS.exists()
