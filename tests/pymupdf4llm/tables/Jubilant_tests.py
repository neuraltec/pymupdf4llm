"""
Tables for the Jubilant supplier tests.
"""

from pathlib import Path
import sys

import pytest

TABLES_DIR = Path(__file__).parent
if str(TABLES_DIR) not in sys.path:
    sys.path.insert(0, str(TABLES_DIR))

from table_extractor import run_table_test

SUPPLIER_ID = "jubilant"
PDF_ENV_VAR = "JUBILANT_PDF_PATH"

# (test_id, page, table_index, expected_ascii_matrix)
TEST_CONFIGURATIONS = [
    (
        "table1",
        None,
        0,
        """-----------------
|STAGE : ARP-3  |
|---------------|
|Input  |Output |
|batch  |batch  |
|size   |size   |
|-------|-------|
|55 – 60|43.18  |
|Kg of  |to 57.6|
|ARP2   |       |
-----------------"""
    ),
    (
        "table5",
        14,
        0,
        """----------------------
|Name of     |Limit  |
|Solvents    |       |
|------------|-------|
|Acetonitrile|Not    |
|            |more   |
|            |than   |
|            |200 ppm|
|------------|-------|
|Isopropyl   |Not    |
|alcohol     |more   |
|            |than   |
|            |2000   |
|            |ppm    |
|------------|-------|
|Cyclohexane |Not    |
|            |more   |
|            |than   |
|            |1000   |
|            |ppm    |
----------------------"""
    ),
    (
        "table11",
        21,
        0,
        """------------------------------------------
|Source       |Are there   |Remarks      |
|             |any direct  |             |
|             |source of   |             |
|             |nitrosamines|             |
|             |(Like sodium|             |
|             |nitrites and|             |
|             |amines)     |             |
|             |(Yes/No)    |             |
|-------------|------------|-------------|
|Solvents used|No          |Risk of      |
|in key       |            |formation of |
|starting     |            |nitroso      |
|materials and|            |impurities   |
|drug         |            |due to       |
|substance    |            |solvents is  |
|manufacturing|            |eliminated.  |
|-------------|------------|-------------|
|Reagents used|No          |Risk of      |
|in key       |            |formation of |
|starting     |            |nitroso      |
|materials and|            |impurities   |
|drug         |            |due to       |
|substance    |            |reagents is  |
|manufacturing|            |eliminated.  |
|             |            |There isa    |
|             |            |possibility  |
|             |            |for carryover|
|             |            |of secondary |
|             |            |amines (DBA) |
|             |            |& Tetra Butyl|
|             |            |Ammonium     |
|             |            |Iodide       |
|             |            |(TBAI).      |
|             |            |Since, there |
|             |            |is no source |
|             |            |of nitrite is|
|             |            |used during  |
|             |            |the          |
|             |            |manufacturing|
|             |            |process of   |
|             |            |drug         |
|             |            |substance,   |
|             |            |risk of      |
|             |            |formation of |
|             |            |nitrosamine  |
|             |            |impurities   |
|             |            |due to       |
|             |            |secondary    |
|             |            |amines from  |
|             |            |DBA and TBAI |
|             |            |is ruled out.|
|-------------|------------|-------------|
|All the      |            |Risk of      |
|possible     |            |formation of |
|process and  |            |nitroso      |
|degradation  |            |impurities   |
|             |            |due to the   |
|             |            |possible     |
|             |            |process and  |
------------------------------------------"""
    ),
    (
        "table13",
        23,
        0,
        """--------------------------------------------------------------
|Impurity    |Specifications|Batch No.                       |
|            |              |--------------------------------|
|            |              |3ARP321002|3ARP321003|3ARP321004|
|------------|--------------|----------|----------|----------|
|Aripiprazole|Not more than |0.03%     |0.04%     |0.02%     |
|Related     |0.10%         |          |          |          |
|compound-G  |              |          |          |          |
|------------|--------------|----------|----------|----------|
|Aripiprazole|Not more than |Not       |Not       |Not       |
|Related     |0.10%         |detected  |detected  |detected  |
|compound-F  |              |          |          |          |
|------------|--------------|----------|----------|----------|
|Aripiprazole|Not more than |0.02%     |0.02%     |0.02%     |
|4, 4’-dimer |0.10%         |          |          |          |
|------------|--------------|----------|----------|----------|
|Any other   |Not more than |0.06%     |0.05%     |0.05%     |
|individual  |0.10%         |          |          |          |
|impurity    |              |          |          |          |
|------------|--------------|----------|----------|----------|
|Total       |Not more than |0.013%    |0.13%     |0.11%     |
|impurities  |0.50%         |          |          |          |
--------------------------------------------------------------"""
    ),
    (
        "table14",
        23,
        1,
        """----------------------------------------------------------------
|Solvents    |Specifications|Batch No.                         |
|            |              |----------------------------------|
|            |              |3ARP321002|3ARP321003|3ARP321004  |
|------------|--------------|----------|----------|------------|
|Acetonitrile|Not more than |Below     |Below     |Below       |
|[LOD:15 ppm;|200 ppm       |Detection |Detection |quantitation|
|LOQ: 52 ppm]|              |Limit     |Limit     |limit       |
|------------|--------------|----------|----------|------------|
|Isopropyl   |Not more than |116 ppm   |141 ppm   |140 ppm     |
|alcohol     |2000 ppm      |          |          |            |
|[LOD:30 ppm;|              |          |          |            |
|LOQ: 75 ppm]|              |          |          |            |
|------------|--------------|----------|----------|------------|
|Cyclohexane |Not more than |Below     |Below     |Below       |
|[LOD:1 ppm; |1000 ppm      |Detection |Detection |Detection   |
|LOQ: 4 ppm] |              |Limit     |Limit     |Limit       |
----------------------------------------------------------------"""
    ),
    (
        "table15",
        24,
        0,
        """-----------------------------
|S.No   |Batch. No |Residue |
|       |          |on      |
|       |          |Ignition|
|-------|----------|--------|
|1      |3ARP321002|0.03%   |
|-------|----------|--------|
|2      |3ARP321003|0.03%   |
|-------|----------|--------|
|3      |3ARP321004|0.03%   |
|-------|----------|--------|
|Specification     |Not more|
|Limit             |than    |
|                  |0.1%    |
-----------------------------"""
    ),
    (
        "table16",
        25,
        0,
        """-------------------------------------------------------------------------
|Element|Class  |Detection|Quantitation|Batch #                         |
|       |       |Limit    |Limit       |                                |
|       |       |         |            |--------------------------------|
|       |       |         |            |3ARP317001|3ARP317002|3ARP318001|
|-------|-------|---------|------------|----------|----------|----------|
|Cd     |1      |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|Pb     |1      |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|As     |1      |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|Hg     |1      |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|Co     |2A     |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|V      |2A     |0.03ppm  |0.10 ppm    |BQL       |BQL       |BQL       |
|-------|-------|---------|------------|----------|----------|----------|
|Ni     |2A     |0.03ppm  |0.10 ppm    |BQL       |BQL       |0.17      |
-------------------------------------------------------------------------"""
    ),
    (
        "table17",
        25,
        1,
        """-----------------------------------------------------------------------------
|Element|Class   |Detection|Quantitation|Batch #                            |
|       |        |Limit    |Limit       |                                   |
|       |        |         |            |-----------------------------------|
|       |        |         |            |3APR3/12001|3APR3/12002|3APR3/12003|
|-------|--------|---------|------------|-----------|-----------|-----------|
|Al     |Other   |2.2 ppm  |6.5 ppm     |BDL        |BDL        |BDL        |
|       |elements|         |            |           |           |           |
-----------------------------------------------------------------------------"""
    ),
    (
        "table18",
        26,
        0,
        """------------------------------------------------------
|Intended route of administration/Use of the         |
|substance: Oral                                     |
|----------------------------------------------------|
|Element|Class  |Intentionally|Considered |Conclusion|
|       |       |added?       |in risk    |          |
|       |       |             |management?|          |
|-------|-------|-------------|-----------|----------|
|Cd     |1      |No           |Yes        |Absent    |
|-------|-------|-------------|-----------|----------|
|Pb     |1      |No           |Yes        |Absent    |
|-------|-------|-------------|-----------|----------|
|As     |1      |No           |Yes        |Absent    |
|-------|-------|-------------|-----------|----------|
|Hg     |1      |No           |Yes        |Absent    |
|-------|-------|-------------|-----------|----------|
|Co     |2A     |No           |Yes        |Absent    |
|-------|-------|-------------|-----------|----------|
|V      |2A     |No           |Yes        |Absent    |
|-------|-------|-------------|-----------|----------|
|Ni     |2A     |No           |Yes        |Absent    |
|-------|-------|-------------|-----------|----------|
|Tl     |2B     |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Au     |2B     |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Pd     |2B     |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Ir     |2B     |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Os     |2B     |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Rh     |2B     |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Ru     |2B     |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Se     |2B     |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Ag     |2B     |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Pt     |2B     |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Li     |3      |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Sb     |3      |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Ba     |3      |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Mo     |3      |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Cu     |3      |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Sn     |3      |No           |No         |*         |
|-------|-------|-------------|-----------|----------|
|Cr     |3      |No           |No         |*         |
------------------------------------------------------"""
    ),
    (
        "table19",
        27,
        0,
        """-----------------------------------------------------------------------------------------------------------------------
|Compound |Source     |Limit  |Class    |Batch results in the final API     |Methodology|LOD (%)|LOQ (%)|Remark       |
|/impurity|           |(ppm)  |         |                                   |used       |       |       |             |
|         |           |       |         |-----------------------------------|           |       |       |             |
|         |           |       |         |3APR3/12001|3APR3/12002|3APR3/12003|           |       |       |             |
|---------|-----------|-------|---------|-----------|-----------|-----------|-----------|-------|-------|-------------|
|Benzene  |Isopropyl  |NMT 2  |ICH,     |Below      |Below      |Below      |HS-GC      |0.2    |0.7    |No carry-over|
|         |alcohol and|ppm    |Class- 1.|detection  |detection  |detection  |           |       |       |to the       |
|         |Cyclohexane|       |         |limit      |limit      |limit      |           |       |       |Aripiprazole.|
|         |           |       |         |           |           |           |           |       |       |Hence control|
|         |           |       |         |           |           |           |           |       |       |in the       |
|         |           |       |         |           |           |           |           |       |       |Aripiprazole |
|         |           |       |         |           |           |           |           |       |       |is not       |
|         |           |       |         |           |           |           |           |       |       |proposed.    |
|---------|-----------|-------|---------|-----------|-----------|-----------|-----------|-------|-------|             |
|Aluminium|7-HDQ      |NMT 50 |Non-     |Below      |Below      |Below      |ICP-MS     |2.2    |6.5    |             |
|         |           |ppm    |Genotoxic|detection  |detection  |detection  |           |       |       |             |
|         |           |       |         |limit      |limit      |limit      |           |       |       |             |
-----------------------------------------------------------------------------------------------------------------------"""
    ),
]


@pytest.mark.parametrize(
    "test_id,page,table_index,expected_ascii_matrix",
    TEST_CONFIGURATIONS,
    ids=[config[0] for config in TEST_CONFIGURATIONS]
)
def test_ascii_matrix_comparison(test_id, page, table_index, expected_ascii_matrix):
    run_table_test(PDF_ENV_VAR, test_id, page, table_index, expected_ascii_matrix)

