"""
Tables for the Finerenona HINYE supplier tests.
"""

from pathlib import Path
import sys

import pytest

TABLES_DIR = Path(__file__).parent
if str(TABLES_DIR) not in sys.path:
    sys.path.insert(0, str(TABLES_DIR))

from test_table_runner import run_table_test

SUPPLIER_ID = "finerenona_hinye"
PDF_ENV_VAR = "FINERENONA_HINYE_PDF_PATH"
FIXTURE_PATH = TABLES_DIR / "fixtures" / f"{SUPPLIER_ID}.json"

# (test_id, page, table_index, expected_ascii_matrix)
TEST_CONFIGURATIONS = [
    ("table2_page9", 9, 1, None),
    ("table_page10", 10, 0, None),
    ("table_page15", 15, 0, None),
    ("table_page16", 16, 0, None),
    ("table_page226", 226, 1, None),
    ("table_page226_alt", 226, 2, None),
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
"""
Tables for the Finerenona HINYE supplier tests.
"""

from pathlib import Path
import sys

import pytest

TABLES_DIR = Path(__file__).parent
if str(TABLES_DIR) not in sys.path:
    sys.path.insert(0, str(TABLES_DIR))

from test_table_runner import run_table_test

SUPPLIER_ID = "finerenona_hinye"
PDF_ENV_VAR = "FINERENONA_HINYE_PDF_PATH"

# (test_id, page, table_index, expected_ascii_matrix)
TEST_CONFIGURATIONS = [
    (
        "table2_page9",
        9,
        1,
        """--------------------------------------------------------------------------------------------------------------------------
|Material Code|Chemical Name                                                                                   |CAS No.  |
|-------------|------------------------------------------------------------------------------------------------|---------|
|CDI          |N. N-carbonyldiimidazole                                                                        |530-62-1 |
|-------------|------------------------------------------------------------------------------------------------|---------|
|DMAP         |4-Dimethylaminopyridine                                                                         |1122-58-3|
|-------------|------------------------------------------------------------------------------------------------|---------|
|HMDS         |Hexamethyldisilazane                                                                            |999-97-3 |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- 1    |4-cyano-2-methoxybenzaldehyde                                                                   |21962-45-|
|             |                                                                                                |8        |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- 2    |Ethyl 2-cyanoacetoacetate                                                                       |65193-87-|
|             |                                                                                                |5        |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- 4    |4-amino-5-methyl-2-hydroxypyridine                                                              |95306-64-|
|             |                                                                                                |2        |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- 10   |D - (+) - Dibenzoyl Tartaric Acid                                                               |17026-42-|
|             |                                                                                                |5        |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- 3    |2-cyanoethyl 2- (4-cyano-2-methoxybenzylidene) -3-oxobutyric acid ester                         |1050477-3|
|             |                                                                                                |9-8      |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- 5    |4- (4-cyano-2-methoxyphenyl) -2,8-dimethyl-5-oxo-1,4,5,6-tetrahydro-1,6-naphthalene-3-carboxylic|1050477-4|
|             |acid 2-cyanoethyl ester                                                                         |3-4      |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- 6    |4- (4-cyano-2-methoxyphenyl) -5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3- carboxylic   |1050477-4|
|             |acid 2-cyanoethyl ester                                                                         |4-5      |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- 7    |4- (4-cyano-2-methoxyphenyl) -5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3- carboxylic   |1050477-4|
|             |acid                                                                                            |5-6      |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- 8    |4- (4-cyano-2-methoxyphenyl) -5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3- carboxamide  |1050477-2|
|             |                                                                                                |7-4      |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- 9    |Non maleketoneD - (+) - dibenzoyl tartrate                                                      |N/A      |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304- CP   |(4S) -4- (4-cyano-2-methoxyphenyl) -5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3-        |1050477-3|
|             |carboxamide                                                                                     |1-0      |
|-------------|------------------------------------------------------------------------------------------------|---------|
|YA2304       |(4S) -4- (4-cyano-2-methoxyphenyl) -5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3-        |1050477-3|
|             |carboxamide                                                                                     |1-0      |
--------------------------------------------------------------------------------------------------------------------------"""
    ),
    (
        "table_page10",
        10,
        0,
        """---------------------------------------------------------------------------------------
|Materials         |Name                                         |Grade               |
|------------------|---------------------------------------------|--------------------|
|Starting materials|4-cyano-2-methoxybenzaldehyde (YA2304-1)     |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Starting materials|Ethyl 2-cyanoacetoacetate (YA2304-2)         |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Starting materials|4-amino-5-methyl-2-hydroxypyridine (YA2304-4)|Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |piperidine                                   |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |Glacial acetic Acid                          |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |Concentrated sulfuric acid                   |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |Triethyl orthoacetate                        |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |Sodium Hydroxide                             |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |Anhydrous sodium acetate                     |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |Concentrated hydrochloric Acid               |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |N, N-carbonyl diimidazole                    |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |4-dimethylaminopyridine                      |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |Hexamethyldisilazane                         |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |D-(+) -dibenzoyltartaric acid                |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Reagents          |Sodium phosphate                             |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Solvents          |Isopropyl alcohol                            |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Solvents          |2-butanol                                    |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Solvents          |N-methylpyrrolidone                          |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Solvents          |tetrahydrofuran                              |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Solvents          |Toluene                                      |Industrial chemicals|
|------------------|---------------------------------------------|--------------------|
|Solvents          |Anhydrous ethanol                            |Industrial chemicals|
---------------------------------------------------------------------------------------"""
    ),
    (
        "table_page15",
        15,
        0,
        """--------------------------------------------------------------------------------
|Batch No.|Accurate measured mass value|Theoretical value|Elemental composition|
|---------|----------------------------|-----------------|---------------------|
|231101   |379.18                      |378.43           |[C21H22N4O3+H]+      |
--------------------------------------------------------------------------------"""
    ),
    (
        "table_page16",
        16,
        0,
        """------------------------------------------------------------------------------------------------------------------------
|2974.23, 2953.02, 28 35.36         |υCH       |-CH -CH -C 3, 2, H         |Carbon hydrogen stretching vibration       |
|-----------------------------------|----------|---------------------------|-------------------------------------------|
|2229.71                            |υC≡N      |-CN                        |Carbon nitrogen triple bond stretching     |
|                                   |          |                           |vibration                                  |
|-----------------------------------|----------|---------------------------|-------------------------------------------|
|1683.86                            |υC=O      |-CONH2                     |Carbon oxygen double bond stretching       |
|                                   |          |                           |vibration                                  |
|-----------------------------------|----------|---------------------------|-------------------------------------------|
|1660.71                            |υC=C      |-C=C                       |Carbon carbon double bond stretching       |
|                                   |          |                           |vibration                                  |
|-----------------------------------|----------|---------------------------|-------------------------------------------|
|1606.70, 1573.91, 14 89.05         |υC=N，υC=C|Pyridine ring, benzene ring|Expansion and contraction vibrations of    |
|                                   |          |                           |carbon nitro gen and carbon carbon do uble |
|                                   |          |                           |bonds                                      |
|-----------------------------------|----------|---------------------------|-------------------------------------------|
|1463.97                            |δCH       |-CH2                       |In-plane shear vibration                   |
|-----------------------------------|----------|---------------------------|-------------------------------------------|
|1454.33, 1431.18, 14 08.04, 1381.03|δCH       |-CH3                       |Out of plane deformation vibration         |
|-----------------------------------|----------|---------------------------|-------------------------------------------|
|1267.23, 1257.59                   |υ=C-O-C   |-OCH3,-OCH2                |Asymmetric stretching vibr ation of ether  |
|                                   |          |                           |bond                                       |
|-----------------------------------|----------|---------------------------|-------------------------------------------|
|1138.00, 1031.92                   |υ=C-O-C   |-OCH3,-OCH2                |Symmetric stretching vibrat ion of ether   |
|                                   |          |                           |bond                                       |
------------------------------------------------------------------------------------------------------------------------"""
    ),

     (
        "table_page226",
        226,
        1,
        """------------------------------------------------------------------------------------------------------------------------
|Main peak    |6709         |6535         |6563         |6771         |6836         |6606         |6732        |1.7    |
|area         |             |             |             |             |             |             |            |       |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|------------|-------|
|Principal    |75.4         |65.3         |92.2         |89.8         |54.0         |103.0        |92.8        |/      |
|peak         |             |             |             |             |             |             |            |       |
|signal-to-   |             |             |             |             |             |             |            |       |
|noise ratio  |             |             |             |             |             |             |            |       |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|------------|-------|
|Conclusion: When the control solution was placed at 5℃for 100h, the chromatogram of the control solution (251nm)      |
|showed that the main peak signal-to-noise ratio (S/N) was NLT 54.0 (required to be NLT 20), and the RSD of the peak   |
|area was 1.7% (required to be no more than 10%). All of the above meet the verification requirements, indicating that |
|the control solution is stable within 100hunder 5℃.                                                                   |
------------------------------------------------------------------------------------------------------------------------"""
    ),
    (
    "table_page226",
        226,
        2,
        """------------------------------------------------------------------------------------------------------------------------
|Main peak    |6709         |6535         |6563         |6771         |6836         |6606         |6732        |1.7    |
|area         |             |             |             |             |             |             |            |       |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|------------|-------|
|Principal    |75.4         |65.3         |92.2         |89.8         |54.0         |103.0        |92.8        |/      |
|peak         |             |             |             |             |             |             |            |       |
|signal-to-   |             |             |             |             |             |             |            |       |
|noise ratio  |             |             |             |             |             |             |            |       |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|------------|-------|
|Conclusion: When the control solution was placed at 5℃for 100h, the chromatogram of the control solution (251nm)      |
|showed that the main peak signal-to-noise ratio (S/N) was NLT 54.0 (required to be NLT 20), and the RSD of the peak   |
|area was 1.7% (required to be no more than 10%). All of the above meet the verification requirements, indicating that |
|the control solution is stable within 100hunder 5℃.                                                                   |
------------------------------------------------------------------------------------------------------------------------"""
    ),
]


@pytest.mark.parametrize(
    "test_id,page,table_index,expected_ascii_matrix",
    TEST_CONFIGURATIONS,
    ids=[config[0] for config in TEST_CONFIGURATIONS]
)
def test_ascii_matrix_comparison(test_id, page, table_index, expected_ascii_matrix):
    run_table_test(PDF_ENV_VAR, test_id, page, table_index, expected_ascii_matrix)

