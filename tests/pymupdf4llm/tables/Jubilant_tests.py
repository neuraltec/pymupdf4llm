"""
Tables for the Jubilant supplier tests.
"""

from pathlib import Path
import sys

import pytest

TABLES_DIR = Path(__file__).parent
if str(TABLES_DIR) not in sys.path:
    sys.path.insert(0, str(TABLES_DIR))

from test_table_runner import run_table_test

SUPPLIER_ID = "jubilant"
PDF_ENV_VAR = "JUBILANT_PDF_PATH"
FIXTURE_PATH = TABLES_DIR / "fixtures" / f"{SUPPLIER_ID}.json"

# (test_id, page, table_index, expected_ascii_matrix)
TEST_CONFIGURATIONS = [
    ("table1", None, 0, None),
    ("table5", 14, 0, None),
    ("table11", 21, 0, None),
    ("table13", 23, 0, None),
    ("table14", 23, 1, None),
    ("table15", 24, 0, None),
    ("table16", 25, 0, None),
    ("table17", 25, 1, None),
    ("table18", 26, 0, None),
    ("table19", 27, 0, None),
]


@pytest.mark.parametrize(
    "test_id,page,table_index,expected_ascii_matrix",
    TEST_CONFIGURATIONS,
    ids=[config[0] for config in TEST_CONFIGURATIONS]
)
def test_ascii_matrix_comparison(test_id, page, table_index, expected_ascii_matrix):
    run_table_test(
        PDF_ENV_VAR,
        test_id,
        page,
        table_index,
        expected_ascii_matrix,
        fixture_path=FIXTURE_PATH,
        test_configurations=TEST_CONFIGURATIONS,
    )

