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
        """----------------------------------------------------------------------------
|Materia l |Chemical Name                                        |CAS No.  |
|Code      |                                                     |         |
|----------|-----------------------------------------------------|---------|
|CDI       |N. N-carbonyldiimidazole                             |530-62-1 |
|----------|-----------------------------------------------------|---------|
|DMAP      |4-Dimethylaminopyridine                              |1122-58-3|
|----------|-----------------------------------------------------|---------|
|HMDS      |Hexamethyldisilazane                                 |999-97-3 |
|----------|-----------------------------------------------------|---------|
|YA2304- 1 |4-cyano-2-methoxybenzaldehyde                        |21962-45-|
|          |                                                     |8        |
|----------|-----------------------------------------------------|---------|
|YA2304- 2 |Ethyl 2-cyanoacetoacetate                            |65193-87-|
|          |                                                     |5        |
|----------|-----------------------------------------------------|---------|
|YA2304- 4 |4-amino-5-methyl-2-hydroxypyridine                   |95306-64-|
|          |                                                     |2        |
|----------|-----------------------------------------------------|---------|
|YA2304- 10|D - (+) - Dibenzoyl Tartaric Acid                    |17026-42-|
|          |                                                     |5        |
|----------|-----------------------------------------------------|---------|
|YA2304- 3 |2-cyanoethyl 2- (4-cyano-2-methoxybenzylidene)       |1050477-3|
|          |-3-oxobutyric acid ester                             |9-8      |
|----------|-----------------------------------------------------|---------|
|YA2304- 5 |4- (4-cyano-2-methoxyphenyl)                         |1050477-4|
|          |-2,8-dimethyl-5-oxo-1,4,5,6-tetrahydro-1,6-naphthalen|3-4      |
|          |e-3-carboxylic acid 2-cyanoethyl ester               |         |
|----------|-----------------------------------------------------|---------|
|YA2304- 6 |4- (4-cyano-2-methoxyphenyl)                         |1050477-4|
|          |-5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3-|4-5      |
|          |carboxylic acid 2-cyanoethyl ester                   |         |
|----------|-----------------------------------------------------|---------|
|YA2304- 7 |4- (4-cyano-2-methoxyphenyl)                         |1050477-4|
|          |-5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3-|5-6      |
|          |carboxylic acid                                      |         |
|----------|-----------------------------------------------------|---------|
|YA2304- 8 |4- (4-cyano-2-methoxyphenyl)                         |1050477-2|
|          |-5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3-|7-4      |
|          |carboxamide                                          |         |
|----------|-----------------------------------------------------|---------|
|YA2304- 9 |Non maleketone D - (+) - dibenzoyl tartrate          |N/A      |
|----------|-----------------------------------------------------|---------|
|YA2304- CP|(4S) -4- (4-cyano-2-methoxyphenyl)                   |1050477-3|
|          |-5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3-|1-0      |
|          |carboxamide                                          |         |
|----------|-----------------------------------------------------|---------|
|YA2304    |(4S) -4- (4-cyano-2-methoxyphenyl)                   |1050477-3|
|          |-5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3-|1-0      |
|          |carboxamide                                          |         |
----------------------------------------------------------------------------"""
    ),
    (
        "table_page10",
        10,
        0,
        """---------------------------------------------------------------------
|Materials|Name                                |Grade               |
|---------|------------------------------------|--------------------|
|Starting |4-cyano-2-methoxybenzaldehyde       |Industrial chemicals|
|materials|(YA2304-1)                          |                    |
|         |------------------------------------|--------------------|
|         |Ethyl 2-cyanoacetoacetate (YA2304-2)|Industrial chemicals|
|         |------------------------------------|--------------------|
|         |4-amino-5-methyl-2-hydroxypyridine  |Industrial chemicals|
|         |(YA2304-4)                          |                    |
|---------|------------------------------------|--------------------|
|Reagents |piperidine                          |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |Glacial acetic Acid                 |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |Concentrated sulfuric acid          |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |Triethyl orthoacetate               |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |Sodium Hydroxide                    |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |Anhydrous sodium acetate            |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |Concentrated hydrochloric Acid      |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |N, N-carbonyl diimidazole           |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |4-dimethylaminopyridine             |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |Hexamethyldisilazane                |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |D-(+) -dibenzoyltartaric acid       |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |Sodium phosphate                    |Industrial chemicals|
|---------|------------------------------------|--------------------|
|Solvents |Isopropyl alcohol                   |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |2-butanol                           |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |N-methylpyrrolidone                 |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |tetrahydrofuran                     |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |Toluene                             |Industrial chemicals|
|         |------------------------------------|--------------------|
|         |Anhydrous ethanol                   |Industrial chemicals|
---------------------------------------------------------------------"""
    ),
    (
        "table_page15",
        15,
        0,
        """--------------------------------------------------------------
|Batch No.|Accurate measured mass|Theoretical|Elemental      |
|         |value                 |value      |composition    |
|---------|----------------------|-----------|---------------|
|231101   |379.18                |378.43     |[C21H22N4O3+H]+|
--------------------------------------------------------------"""
    ),
    (
        "table_page16",
        16,
        0,
        """-----------------------------------------------------------------------
|2974.23, 2953.02,   |υCH      |-CH3,-CH2,  |Carbon hydrogen 	      |
|2835.36             |         |-CH         |stretching vibration     |
|--------------------|---------|------------|-------------------------|
|2229.71             |υC≡N     |-CN         |Carbon nitrogen triple   |
|                    |         |            |bond stretching          |
|                    |         |            |vibration                |
|--------------------|---------|------------|-------------------------|
|1683.86             |υC=O     |-CONH2      |Carbon oxygen double     |
|                    |         |            |bond stretching vibration|
|--------------------|---------|------------|-------------------------|
|1660.71             |υC=C     |-C=C        |Carbon carbon double     |
|                    |         |            |bond stretching vibration|
|--------------------|---------|------------|-------------------------|
|1606.70, 1573.91,   |υC=N,    |Pyridine    |Expansion and contraction|
|1489.05             |υC=C     |ring,       |vibrations of carbon     |
|                    |         |benzene     |nitro gen and carbon     |
|                    |         |ring        |carbon double bonds      |
|--------------------|---------|------------|-------------------------|
|1463.97             |δCH      |-CH2        |In-plane shear vibration |
|--------------------|---------|------------|-------------------------|
|1454.33, 1431.18,   |δCH      |-CH3        |Out of plane deformation |
|1408.04, 1381.03    |         |            |vibration                |
|--------------------|---------|------------|-------------------------|
|1267.23, 1257.59    |υ=C-O-C  |-OCH3,-OCH2 |Asymmetric stretching    |
|                    |         |            |vibration of ether bond  |
|--------------------|---------|------------|-------------------------|
|1138.00, 1031.92    |υ=C-O-C  |-OCH3,-OCH2 |Symmetric stretching     |
|                    |         |            |vibration of ether bond  |
-----------------------------------------------------------------------"""
    ),
]


@pytest.mark.parametrize(
    "test_id,page,table_index,expected_ascii_matrix",
    TEST_CONFIGURATIONS,
    ids=[config[0] for config in TEST_CONFIGURATIONS]
)
def test_ascii_matrix_comparison(test_id, page, table_index, expected_ascii_matrix):
    run_table_test(PDF_ENV_VAR, test_id, page, table_index, expected_ascii_matrix)

