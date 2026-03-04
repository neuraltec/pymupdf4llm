from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def _load_jubilant_config():
    tables_dir = Path(__file__).parent
    if str(tables_dir) not in sys.path:
        sys.path.insert(0, str(tables_dir))

    from Jubilant_tests import TEST_CONFIGURATIONS, PDF_ENV_VAR, SUPPLIER_ID
    return SUPPLIER_ID, PDF_ENV_VAR, TEST_CONFIGURATIONS


def _extract_ascii_tables(pdf_env_var: str, test_configurations):
    from test_table_runner import _generate_fixture

    tables_dir = Path(__file__).parent
    fixtures_dir = tables_dir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixtures_dir / "tmp-fixture.json"
    _generate_fixture(fixture_path, pdf_env_var, test_configurations)
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def main() -> int:
    _maybe_load_dotenv()
    supplier = sys.argv[1] if len(sys.argv) > 1 else "jubilant"
    if supplier != "jubilant":
        print("Only 'jubilant' is supported right now.")
        return 2

    supplier_id, pdf_env_var, test_configurations = _load_jubilant_config()
    fixtures = _extract_ascii_tables(pdf_env_var, test_configurations)

    tables_dir = Path(__file__).parent
    fixtures_dir = tables_dir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixtures_dir / f"{supplier_id}.json"
    fixture_path.write_text(
        json.dumps(fixtures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote fixtures to {fixture_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
