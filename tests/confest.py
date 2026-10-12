from pathlib import Path

import pytest

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hr.db"


@pytest.fixture(scope="session", autouse=True)
def hr_db():
    """The tests read the synthetic HR database; build it once if it is missing (seed 42 = same data)."""
    if not DB_PATH.exists():
        import build_db
        build_db.main()
    return DB_PATH