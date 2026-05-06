from pathlib import Path


def test_scaffold_layout_exists() -> None:
    base = Path(__file__).resolve().parents[1]
    assert (base / "orchestrator" / "main.py").exists()
    assert (base / "config" / "modes" / "container.yaml").exists()
    assert (base / "config" / "modes" / "native.yaml").exists()
    assert (base / "scenarios" / "sepal-sum" / "scenario.yaml").exists()
