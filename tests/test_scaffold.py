from pathlib import Path


def test_project_scaffold_exists() -> None:
    root = Path(__file__).resolve().parents[1]

    expected_paths = [
        root / "data" / "raw",
        root / "data" / "processed",
        root / "data" / "squads",
        root / "src" / "models",
        root / "src" / "engine",
        root / "src" / "viz",
        root / "src" / "data",
        root / "app" / "streamlit_app.py",
        root / "requirements.txt",
    ]

    missing = [str(path.relative_to(root)) for path in expected_paths if not path.exists()]
    assert missing == []
