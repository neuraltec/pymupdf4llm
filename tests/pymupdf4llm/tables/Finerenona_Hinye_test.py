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
        0,
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
        "table_page15",
        15,
        1,
        """------------------------------------------------------------------------------------------------------------------------
|Absorption peak posit ion (cm-1)|Type of vibr ation|Possible Gro ups           |Remarks                               |
|--------------------------------|------------------|---------------------------|--------------------------------------|
|3475.73                         |υNH2              |-CONH2                     |Asymmetric stretching vibr ation of   |
|                                |                  |                           |nitrogen and hydr ogen                |
|--------------------------------|------------------|---------------------------|--------------------------------------|
|3415.93                         |υNH               |-CONH2                     |Nitrogen hydrogen symmetr ic          |
|                                |                  |                           |stretching vibration                  |
|--------------------------------|------------------|---------------------------|--------------------------------------|
|3365.78                         |υNH               |-NH                        |Nitrogen hydrogen stretchi ng         |
|                                |                  |                           |vibration                             |
|--------------------------------|------------------|---------------------------|--------------------------------------|
|3115.04, 3079.53                |υCH               |Pyridine ring, benzene ring|Carbon hydrogen stretching vibration  |
------------------------------------------------------------------------------------------------------------------------"""
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
        0,
        """------------------------------------------------------------------------------------------------------------------------
|Main peak area                                        |6709   |6535   |6563   |6771   |6836   |6606   |6732   |1.7    |
|------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
|Principal peak signal-to-noise ratio                  |75.4   |65.3   |92.2   |89.8   |54.0   |103.0  |92.8   |/      |
|------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
|Conclusion: When the control solution was placed at 5℃for 100h, the chromatogram of the control solution (251nm)      |
|showed that the main peak signal-to-noise ratio (S/N) was NLT 54.0 (required to be NLT 20), and the RSD of the peak   |
|area was 1.7% (required to be no more than 10%). All of the above meet the verification requirements, indicating that |
|the control solution is stable within 100hunder 5℃.                                                                   |
|                                                                                                                      |
|                                                                                                                      |
|                                                                                                                      |
|                                                                                                                      |
------------------------------------------------------------------------------------------------------------------------"""
    ),
     (
        "table_page226",
        226,
        1,
        """---------------------------------------------------------------------------------------------------------------------------------------------------------------------
|Investigati on Items                          |Impurity    |Reference   |Col4        |Col5        |Col6        |Col7        |Col8        |Col9        |RSD %       |
|                                              |names       |solution    |            |            |            |            |            |            |            |
|----------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
|Investigati on Items                          |Impurity    |0h          |8h          |19h         |39h         |46h         |59h         |100h        |100h        |
|                                              |names       |            |            |            |            |            |            |            |            |
|----------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
|Peak area                                     |YA2304-     |1103 7      |1099 0      |1104 2      |1109 6      |1100 6      |1123 5      |1103 1      |0.8         |
|                                              |17          |            |            |            |            |            |            |            |            |
|----------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
|Peak area                                     |YA2304-     |8322        |8233        |8327        |8411        |8336        |8361        |8219        |0.9         |
|                                              |10          |            |            |            |            |            |            |            |            |
|----------------------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
|Conclusion: When the control solution was     |Conclusion: |Conclusion: |Conclusion: |Conclusion: |Conclusion: |Conclusion: |Conclusion: |Conclusion: |Conclusion: |
|placed at 5℃for 100h, the chromatogram of the |When the    |When the    |When        |When        |When        |When        |When        |When        |When        |
|control solution (230nm) was followed by      |control     |control     |the         |the         |the         |the         |the         |the         |the         |
|YA2304-17 and YA2304-10, and the RSD of the   |solution    |solution    |control     |control     |control     |control     |control     |control     |control     |
|peak-peak area of each impurity in the        |was         |was         |solution    |solution    |solution    |solution    |solution    |solution    |solution    |
|chromatogram was no more than 0.9% (required  |placed      |placed at   |was         |was         |was         |was         |was         |was         |was         |
|to be no more than 10%). The verification     |at 5℃for    |5℃for       |placed      |placed      |placed      |placed      |placed      |placed      |placed      |
|requirements were met, indicating that the    |100h,       |100h, the   |at          |at          |at          |at          |at          |at          |at          |
|control solution was stable within 100hunder  |the         |chromatogram|5℃for       |5℃for       |5℃for       |5℃for       |5℃for       |5℃for       |5℃for       |
|5℃.                                           |chromatogram|of the      |100h,       |100h,       |100h,       |100h,       |100h,       |100h,       |100h,       |
|                                              |of the      |control     |the         |the         |the         |the         |the         |the         |the         |
|                                              |control     |solution    |chromatogram|chromatogram|chromatogram|chromatogram|chromatogram|chromatogram|chromatogram|
|                                              |solution    |(230nm)     |of the      |of the      |of the      |of the      |of the      |of the      |of the      |
|                                              |(230nm)     |was         |control     |control     |control     |control     |control     |control     |control     |
|                                              |was         |followed    |solution    |solution    |solution    |solution    |solution    |solution    |solution    |
|                                              |followed    |by          |(230nm)     |(230nm)     |(230nm)     |(230nm)     |(230nm)     |(230nm)     |(230nm)     |
|                                              |by          |YA2304-17   |was         |was         |was         |was         |was         |was         |was         |
|                                              |YA2304-17   |and         |followed    |followed    |followed    |followed    |followed    |followed    |followed    |
|                                              |and         |YA2304-10,  |by          |by          |by          |by          |by          |by          |by          |
|                                              |YA2304-10,  |and the     |YA2304-17   |YA2304-17   |YA2304-17   |YA2304-17   |YA2304-17   |YA2304-17   |YA2304-17   |
|                                              |and the     |RSD of      |and         |and         |and         |and         |and         |and         |and         |
|                                              |RSD of      |the         |YA2304-10,  |YA2304-10,  |YA2304-10,  |YA2304-10,  |YA2304-10,  |YA2304-10,  |YA2304-10,  |
|                                              |the         |peak-peak   |and the     |and the     |and the     |and the     |and the     |and the     |and the     |
|                                              |peak-peak   |area of     |RSD of      |RSD of      |RSD of      |RSD of      |RSD of      |RSD of      |RSD of      |
|                                              |area of     |each        |the         |the         |the         |the         |the         |the         |the         |
|                                              |each        |impurity    |peak-peak   |peak-peak   |peak-peak   |peak-peak   |peak-peak   |peak-peak   |peak-peak   |
|                                              |impurity    |in the      |area of     |area of     |area of     |area of     |area of     |area of     |area of     |
|                                              |in the      |chromatogram|each        |each        |each        |each        |each        |each        |each        |
|                                              |chromatogram|was no      |impurity    |impurity    |impurity    |impurity    |impurity    |impurity    |impurity    |
|                                              |was no      |more than   |in the      |in the      |in the      |in the      |in the      |in the      |in the      |
|                                              |more        |0.9%        |chromatogram|chromatogram|chromatogram|chromatogram|chromatogram|chromatogram|chromatogram|
|                                              |than        |(required   |was no      |was no      |was no      |was no      |was no      |was no      |was no      |
|                                              |0.9%        |to be no    |more        |more        |more        |more        |more        |more        |more        |
|                                              |(required   |more than   |than        |than        |than        |than        |than        |than        |than        |
|                                              |to be no    |10%). The   |0.9%        |0.9%        |0.9%        |0.9%        |0.9%        |0.9%        |0.9%        |
|                                              |more        |verification|(required   |(required   |(required   |(required   |(required   |(required   |(required   |
|                                              |than        |requirements|to be       |to be       |to be       |to be       |to be       |to be       |to be       |
|                                              |10%).       |were met,   |no more     |no more     |no more     |no more     |no more     |no more     |no more     |
|                                              |The         |indicating  |than        |than        |than        |than        |than        |than        |than        |
|                                              |verification|that the    |10%).       |10%).       |10%).       |10%).       |10%).       |10%).       |10%).       |
|                                              |requirements|control     |The         |The         |The         |The         |The         |The         |The         |
|                                              |were        |solution    |verification|verification|verification|verification|verification|verification|verification|
|                                              |met,        |was         |requirements|requirements|requirements|requirements|requirements|requirements|requirements|
|                                              |indicating  |stable      |were        |were        |were        |were        |were        |were        |were        |
|                                              |that the    |within      |met,        |met,        |met,        |met,        |met,        |met,        |met,        |
|                                              |control     |100hunder   |indicating  |indicating  |indicating  |indicating  |indicating  |indicating  |indicating  |
|                                              |solution    |5℃.         |that        |that        |that        |that        |that        |that        |that        |
|                                              |was         |            |the         |the         |the         |the         |the         |the         |the         |
|                                              |stable      |            |control     |control     |control     |control     |control     |control     |control     |
|                                              |within      |            |solution    |solution    |solution    |solution    |solution    |solution    |solution    |
|                                              |100hunder   |            |was         |was         |was         |was         |was         |was         |was         |
|                                              |5℃.         |            |stable      |stable      |stable      |stable      |stable      |stable      |stable      |
|                                              |            |            |within      |within      |within      |within      |within      |within      |within      |
|                                              |            |            |100hunder   |100hunder   |100hunder   |100hunder   |100hunder   |100hunder   |100hunder   |
|                                              |            |            |5℃.         |5℃.         |5℃.         |5℃.         |5℃.         |5℃.         |5℃.         |
---------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
    ),
     (
        "table_page226",
        226,
        2,
        """------------------------------------------------------------------------------------------------------------------------
|Investigation Items                                           |Time of visit                                          |
|                                                              |                                                       |
|--------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|
|Investigation Items                                           |0h     |8h     |19h    |26.5 h.|46h    |59h    |100h   |
|--------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|
|Degree of separation between YA2304 and YA2304-14             |5.9    |6.0    |5.9    |5.9    |5.9    |5.8    |5.8    |
|--------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|
|Degree of separation between YA2304 and YA2304-15             |6.8    |6.9    |6.8    |6.8    |6.8    |6.8    |6.7    |
|--------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|
|Minimum separation between YA2304-17 and adjacent peaks       |1.6    |1.6    |1.6    |1.6    |1.6    |2.2    |2.2    |
|--------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|
|Minimum separation degree between YA2304-16 and adjacent peaks|1.6    |1.6    |1.6    |1.6    |1.6    |1.6    |1.6    |
|--------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|
|Verdict: The system suitable solution was placed at 5℃ for 100h, and the                                              |
|                                                                                                                      |
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

