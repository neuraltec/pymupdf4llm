"""
Tables for the Finerenona HINYE supplier tests.
"""

from pathlib import Path
import sys

import pytest

TABLES_DIR = Path(__file__).parent
if str(TABLES_DIR) not in sys.path:
    sys.path.insert(0, str(TABLES_DIR))

from table_extractor import run_table_test

SUPPLIER_ID = "finerenona_hinye"
PDF_ENV_VAR = "FINERENONA_HINYE_PDF_PATH"

# (test_id, page, table_index, expected_ascii_matrix)
TEST_CONFIGURATIONS = [
    (
        "table2_page9",
        9,
        0,
        """----------------------------------------------------------------------------------------
|Material|Chemical Name                                                      |CAS No.  |
|Code    |                                                                   |         |
|--------|-------------------------------------------------------------------|---------|
|CDI     |N. N-carbonyldiimidazole                                           |530-62-1 |
|--------|-------------------------------------------------------------------|---------|
|DMAP    |4-Dimethylaminopyridine                                            |1122-58-3|
|--------|-------------------------------------------------------------------|---------|
|HMDS    |Hexamethyldisilazane                                               |999-97-3 |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |4-cyano-2-methoxybenzaldehyde                                      |21962-45-|
|1       |                                                                   |8        |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |Ethyl 2-cyanoacetoacetate                                          |65193-87-|
|2       |                                                                   |5        |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |4-amino-5-methyl-2-hydroxypyridine                                 |95306-64-|
|4       |                                                                   |2        |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |D - (+) - Dibenzoyl Tartaric Acid                                  |17026-42-|
|10      |                                                                   |5        |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |2-cyanoethyl 2- (4-cyano-2-methoxybenzylidene) -3-oxobutyric acid  |1050477-3|
|3       |ester                                                              |9-8      |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |4- (4-cyano-2-methoxyphenyl)                                       |1050477-4|
|5       |-2,8-dimethyl-5-oxo-1,4,5,6-tetrahydro-1,6-naphthalene-3-carboxylic|3-4      |
|        |acid 2-cyanoethyl ester                                            |         |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |4- (4-cyano-2-methoxyphenyl)                                       |1050477-4|
|6       |-5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3- carboxylic   |4-5      |
|        |acid 2-cyanoethyl ester                                            |         |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |4- (4-cyano-2-methoxyphenyl)                                       |1050477-4|
|7       |-5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3- carboxylic   |5-6      |
|        |acid                                                               |         |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |4- (4-cyano-2-methoxyphenyl)                                       |1050477-2|
|8       |-5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3- carboxamide  |7-4      |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |Non maleketoneD - (+) - dibenzoyl tartrate                         |N/A      |
|9       |                                                                   |         |
|--------|-------------------------------------------------------------------|---------|
|YA2304- |(4S) -4- (4-cyano-2-methoxyphenyl)                                 |1050477-3|
|CP      |-5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3- carboxamide  |1-0      |
|--------|-------------------------------------------------------------------|---------|
|YA2304  |(4S) -4- (4-cyano-2-methoxyphenyl)                                 |1050477-3|
|        |-5-ethoxy-2,8-dimethyl-1,4-dihydro-1,6-naphthalene-3- carboxamide  |1-0      |
----------------------------------------------------------------------------------------"""
    ),
    (
        "table_page10",
        10,
        0,
        """---------------------------------------------------------
|Materials|Name                              |Grade     |
|---------|----------------------------------|----------|
|Starting |4-cyano-2-methoxybenzaldehyde     |Industrial|
|materials|(YA2304-1)                        |chemicals |
|         |----------------------------------|----------|
|         |Ethyl 2-cyanoacetoacetate         |Industrial|
|         |(YA2304-2)                        |chemicals |
|         |----------------------------------|----------|
|         |4-amino-5-methyl-2-hydroxypyridine|Industrial|
|         |(YA2304-4)                        |chemicals |
|---------|----------------------------------|----------|
|Reagents |piperidine                        |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |Glacial acetic Acid               |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |Concentrated sulfuric acid        |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |Triethyl orthoacetate             |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |Sodium Hydroxide                  |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |Anhydrous sodium acetate          |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |Concentrated hydrochloric Acid    |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |N, N-carbonyl diimidazole         |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |4-dimethylaminopyridine           |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |Hexamethyldisilazane              |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |D-(+) -dibenzoyltartaric acid     |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |Sodium phosphate                  |Industrial|
|         |                                  |chemicals |
|---------|----------------------------------|----------|
|Solvents |Isopropyl alcohol                 |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |2-butanol                         |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |N-methylpyrrolidone               |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |tetrahydrofuran                   |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |Toluene                           |Industrial|
|         |                                  |chemicals |
|         |----------------------------------|----------|
|         |Anhydrous ethanol                 |Industrial|
|         |                                  |chemicals |
---------------------------------------------------------"""
    ),
     (
        "table_page15",
        15,
        1,
        """----------------------------------------
|Absorption|Type of|Possible|Remarks   |
|peak posit|vibr   |Gro ups |          |
|ion (cm-1)|ation  |        |          |
|----------|-------|--------|----------|
|3475.73   |υNH2   |-CONH 2 |Asymmetric|
|          |       |        |stretching|
|          |       |        |vibr ation|
|          |       |        |of        |
|          |       |        |nitrogen  |
|          |       |        |and hydr  |
|          |       |        |ogen      |
|----------|-------|--------|----------|
|3415.93   |υNH    |-CONH 2 |Nitrogen  |
|          |       |        |hydrogen  |
|          |       |        |symmetr ic|
|          |       |        |stretching|
|          |       |        |vibration |
|----------|-------|--------|----------|
|3365.78   |υNH    |-NH     |Nitrogen  |
|          |       |        |hydrogen  |
|          |       |        |stretchi  |
|          |       |        |ng        |
|          |       |        |vibration |
|----------|-------|--------|----------|
|3115.04,  |υCH    |Pyridine|Carbon    |
|3079.53   |       |ring,   |hydrogen  |
|          |       |benzener|stretching|
|          |       |ing     |vibration |
----------------------------------------"""
    ),
    (
        "table_page230",
        230,
        0,
        """-------------------------------------------------------------------------------------------
|       |Impurity |       |9107   |9027   |8791   |8755   |8574   |8452   |8111   |4      |
|       |YA2304-18|       |       |       |       |       |       |       |       |       |
|       |---------|       |-------|-------|-------|-------|-------|-------|-------|-------|
|       |Impurity |       |15639  |15602  |15665  |15644  |15942  |15998  |15712  |1.1    |
|       |YA2304-19|       |       |       |       |       |       |       |       |       |
|       |---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|       |Impurity |230nm  |16023  |15969  |15952  |15986  |16017  |16400  |16199  |1.1    |
|       |YA2304-17|       |       |       |       |       |       |       |       |       |
|       |---------|       |-------|-------|-------|-------|-------|-------|-------|-------|
|       |Impurity |       |12233  |12086  |12173  |12076  |12218  |12178  |11975  |0.8    |
|       |YA2304-10|       |       |       |       |       |       |       |       |       |
|-------|---------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|Conclusions: The impurity peak less than 0.5 times the main peak area of the control     |
|solution was ignored in the chromatogram of the impurity solution added at 5℃for 100h. At|
|230mwavelength, the RSD of the known impurity YA2304-17 and YA2304-10 was not greater    |
|than 1.1% (10% is not required). At 251nm wavelength, the RSD of known impurities        |
|YA2304-12, YA2304-14, YA2304-15, YA2304-16, YA2304-18 and YA2304-19 are not greater than |
|4% (10% is not required); The RSD of other single unknown impurity is less than 0.05%,   |
|the RSD of the single unknown impurity is not counted (not more than 10%), and no new    |
|impurity interferes with the detection of related substances; The above are in line with |
|the verification requirements, indicating that the solution of added impurity is stable  |
|within 100hat 5℃.                                                                        |
-------------------------------------------------------------------------------------------"""
    ),
    (
        "table_page229",
        229,
        0,
        """-----------------------------------------------------------------------------------------------------
|       |Impurity |230nm  |Not     |Not     |Not     |Not     |Not     |Not     |Not     |Content < |
|       |YA2304-17|       |detected|detected|detected|detected|detected|detected|detected|0.05%, RSD|
|       |         |       |        |        |        |        |        |        |        |does not  |
|       |         |       |        |        |        |        |        |        |        |do        |
|       |         |       |        |        |        |        |        |        |        |statistics|
|       |---------|       |--------|--------|--------|--------|--------|--------|--------|----------|
|       |Impurity |       |Not     |Not     |Not     |Not     |Not     |Not     |Not     |Content < |
|       |YA2304-10|       |detected|detected|detected|detected|detected|detected|detected|0.05%, RSD|
|       |         |       |        |        |        |        |        |        |        |does not  |
|       |         |       |        |        |        |        |        |        |        |do        |
|       |         |       |        |        |        |        |        |        |        |statistics|
|-------|---------|-------|--------|--------|--------|--------|--------|--------|--------|----------|
|Conclusion: When the test solution was placed at 5℃ for 100h, the impurity peak smaller than 0.5   |
|times of the main peak area of the control solution was ignored in the chromatogram of the test    |
|solution. The known impurities were detected except YA2304-12 and YA2304-19, and both were less    |
|than 0.05%, and the rest were not detected. Other single unknown impurities were all less than     |
|0.05%, the RSD of each impurity peak area was not counted (not more than 10%), and no new          |
|impurities were added to interfere with the detection of related substances. The above are in line |
|with the verification requirements, indicating that the test solution is stable within 100hunder   |
|5℃.                                                                                                |
-----------------------------------------------------------------------------------------------------"""
    ),
    (
        "table_page229",
        229,
        1,
        """--------------------------------------------------------------------------------------------------
|Time Point        |Detection |0h     |8h     |19h    |25h    |46h    |59h    |100h   |RSD/%     |
|                  |wavelength|       |       |       |       |       |       |       |          |
|------------------|----------|-------|-------|-------|-------|-------|-------|-------|----------|
|Im pur |Impurity  |251nm     |8738   |8628   |8837   |8744   |8810   |8825   |8731   |0.9       |
|ity    |YA2304-12 |          |       |       |       |       |       |       |       |          |
|peak   |          |          |       |       |       |       |       |       |       |          |
|area   |          |          |       |       |       |       |       |       |       |          |
|       |----------|          |-------|-------|-------|-------|-------|-------|-------|----------|
|       |Unknown   |          |/      |945    |1233   |1061   |1491   |1223   |1666   |Content < |
|       |impurity  |          |       |       |       |       |       |       |       |0.05%, RSD|
|       |(RRT≈0.64)|          |       |       |       |       |       |       |       |does not  |
|       |          |          |       |       |       |       |       |       |       |do        |
|       |          |          |       |       |       |       |       |       |       |statistics|
|       |----------|          |-------|-------|-------|-------|-------|-------|-------|----------|
|       |Impurity  |          |10267  |9809   |9895   |10078  |10015  |10050  |9889   |1.6       |
|       |YA2304-14 |          |       |       |       |       |       |       |       |          |
|       |----------|          |-------|-------|-------|-------|-------|-------|-------|----------|
|       |Impurity  |          |8793   |8747   |8744   |8807   |8888   |8899   |8669   |1.0       |
|       |YA2304-15 |          |       |       |       |       |       |       |       |          |
|       |----------|          |-------|-------|-------|-------|-------|-------|-------|----------|
|       |Unknown   |          |1082   |1049   |1104   |1109   |1096   |1102   |1107   |Content < |
|       |impurity  |          |       |       |       |       |       |       |       |0.05%, RSD|
|       |(RRT≈1.27)|          |       |       |       |       |       |       |       |did not do|
|       |          |          |       |       |       |       |       |       |       |statistics|
|       |----------|          |-------|-------|-------|-------|-------|-------|-------|----------|
|       |Impurity  |          |9394   |9292   |9227   |9074   |8798   |8843   |8449   |4         |
|       |YA2304-16 |          |       |       |       |       |       |       |       |          |
--------------------------------------------------------------------------------------------------"""
    ),
    (
        "table_page228",
        228,
        0,
        """---------------------------------------------------------------------------------------------------------
|Time Point        |Detection |0h      |8h      |19h     |38.5 h. |46h     |59h     |100h    |RSD/%     |
|                  |wavelength|        |        |        |        |        |        |        |          |
|------------------|----------|--------|--------|--------|--------|--------|--------|--------|----------|
|Impuri |Impurity  |251nm     |382     |333     |343     |386     |467     |337     |356     |Content < |
|typ    |YA2304-12 |          |        |        |        |        |        |        |        |0.05%, RSD|
|eakar e|          |          |        |        |        |        |        |        |        |does not  |
|a      |          |          |        |        |        |        |        |        |        |do        |
|       |          |          |        |        |        |        |        |        |        |statistics|
|       |----------|          |--------|--------|--------|--------|--------|--------|--------|----------|
|       |Unknown   |          |1182    |1090    |1360    |1327    |1308    |1166    |458     |Content < |
|       |impurity  |          |        |        |        |        |        |        |        |0.05%, RSD|
|       |(RRT≈0.64)|          |        |        |        |        |        |        |        |did not do|
|       |          |          |        |        |        |        |        |        |        |statistics|
|       |----------|          |--------|--------|--------|--------|--------|--------|--------|----------|
|       |Impurity  |          |Not     |Not     |Not     |Not     |Not     |Not     |Not     |Content < |
|       |YA2304-14 |          |detected|detected|detected|detected|detected|detected|detected|0.05%, RSD|
|       |          |          |        |        |        |        |        |        |        |does not  |
|       |          |          |        |        |        |        |        |        |        |do        |
|       |          |          |        |        |        |        |        |        |        |statistics|
|       |----------|          |--------|--------|--------|--------|--------|--------|--------|----------|
|       |Impurity  |          |Not     |Not     |Not     |Not     |Not     |Not     |Not     |Content < |
|       |YA2304-15 |          |detected|detected|detected|detected|detected|detected|detected|0.05%, RSD|
|       |          |          |        |        |        |        |        |        |        |does not  |
|       |          |          |        |        |        |        |        |        |        |do        |
|       |          |          |        |        |        |        |        |        |        |statistics|
|       |----------|          |--------|--------|--------|--------|--------|--------|--------|----------|
|       |Unknown   |          |1220    |1198    |1161    |1174    |1266    |1201    |1181    |Content < |
|       |impurity  |          |        |        |        |        |        |        |        |0.05%, RSD|
|       |(RRT≈1.27)|          |        |        |        |        |        |        |        |does not  |
|       |          |          |        |        |        |        |        |        |        |do        |
|       |          |          |        |        |        |        |        |        |        |statistics|
|       |----------|          |--------|--------|--------|--------|--------|--------|--------|----------|
|       |Impurity  |          |Not     |Not     |Not     |Not     |Not     |Not     |Not     |Content < |
|       |YA2304-16 |          |detected|detected|detected|detected|detected|detected|detected|0.05%, RSD|
|       |          |          |        |        |        |        |        |        |        |does not  |
|       |          |          |        |        |        |        |        |        |        |do        |
|       |          |          |        |        |        |        |        |        |        |statistics|
|       |----------|          |--------|--------|--------|--------|--------|--------|--------|----------|
|       |Impurity  |          |Not     |Not     |Not     |Not     |Not     |Not     |Not     |Content < |
|       |YA2304-18 |          |detected|detected|detected|detected|detected|detected|detected|0.05%, RSD|
|       |          |          |        |        |        |        |        |        |        |does not  |
|       |          |          |        |        |        |        |        |        |        |do        |
|       |          |          |        |        |        |        |        |        |        |statistics|
|       |----------|          |--------|--------|--------|--------|--------|--------|--------|----------|
|       |Impurity  |          |560     |481     |575     |611     |588     |767     |409     |Content < |
|       |YA2304-19 |          |        |        |        |        |        |        |        |0.05%, RSD|
|       |          |          |        |        |        |        |        |        |        |does not  |
|       |          |          |        |        |        |        |        |        |        |do        |
|       |          |          |        |        |        |        |        |        |        |statistics|
---------------------------------------------------------------------------------------------------------"""
    ),
    (
        "table_page226",
        226,
        0,
        """---------------------------------------------------------------------------------
|Main peak area |6709   |6535   |6563   |6771   |6836   |6606   |6732   |1.7    |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|
|Principal peak |75.4   |65.3   |92.2   |89.8   |54.0   |103.0  |92.8   |/      |
|signal-to-noise|       |       |       |       |       |       |       |       |
|ratio          |       |       |       |       |       |       |       |       |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|
|Conclusion: When the control solution was placed at 5℃ for 100h, the           |
|chromatogram of the control solution (251nm) showed that the main peak         |
|signal-to-noise ratio (S/N) was NLT 54.0 (required to be NLT 20), and the RSD  |
|of the peak area was 1.7% (required to be no more than 10%). All of the above  |
|meet the verification requirements, indicating that the control solution is    |
|stable within 100hunder 5℃.                                                    |
---------------------------------------------------------------------------------"""
    ),
     (
        "table_page226",
        226,
        1,
        """--------------------------------------------------------------------------------------
|Investigati|Impurity|Reference solution                                     |RSD %  |
|on Items   |names   |                                                       |       |
|           |        |-------------------------------------------------------|       |
|           |        |0h     |8h     |19h    |39h    |46h    |59h    |100h   |       |
|-----------|--------|-------|-------|-------|-------|-------|-------|-------|-------|
|Peak area  |YA2304- |1103 7 |1099 0 |1104 2 |1109 6 |1100 6 |1123 5 |1103 1 |0.8    |
|           |17      |       |       |       |       |       |       |       |       |
|           |--------|-------|-------|-------|-------|-------|-------|-------|-------|
|           |YA2304- |8322   |8233   |8327   |8411   |8336   |8361   |8219   |0.9    |
|           |10      |       |       |       |       |       |       |       |       |
|-----------|--------|-------|-------|-------|-------|-------|-------|-------|-------|
|Conclusion: When the control solution was placed at 5℃ for 100h, the chromatogram of|
|the control solution (230nm) was followed by YA2304-17 and YA2304-10, and the RSD of|
|the peak-peak area of each impurity in the chromatogram was no more than 0.9%       |
|(required to be no more than 10%). The verification requirements were met,          |
|indicating that the control solution was stable within 100hunder 5℃.                |
--------------------------------------------------------------------------------------"""
    ),
     (
        "table_page226",
        226,
        2,
        """-----------------------------------------------------------------------
|Investigation|Time of visit                                          |
|Items        |                                                       |
|             |-------------------------------------------------------|
|             |0h     |8h     |19h    |26.5 h.|46h    |59h    |100h   |
|-------------|-------|-------|-------|-------|-------|-------|-------|
|Degree of    |5.9    |6.0    |5.9    |5.9    |5.9    |5.8    |5.8    |
|separation   |       |       |       |       |       |       |       |
|between      |       |       |       |       |       |       |       |
|YA2304 and   |       |       |       |       |       |       |       |
|YA2304-14    |       |       |       |       |       |       |       |
|-------------|-------|-------|-------|-------|-------|-------|-------|
|Degree of    |6.8    |6.9    |6.8    |6.8    |6.8    |6.8    |6.7    |
|separation   |       |       |       |       |       |       |       |
|between      |       |       |       |       |       |       |       |
|YA2304 and   |       |       |       |       |       |       |       |
|YA2304-15    |       |       |       |       |       |       |       |
|-------------|-------|-------|-------|-------|-------|-------|-------|
|Minimum      |1.6    |1.6    |1.6    |1.6    |1.6    |2.2    |2.2    |
|separation   |       |       |       |       |       |       |       |
|between      |       |       |       |       |       |       |       |
|YA2304-17 and|       |       |       |       |       |       |       |
|adjacent     |       |       |       |       |       |       |       |
|peaks        |       |       |       |       |       |       |       |
|-------------|-------|-------|-------|-------|-------|-------|-------|
|Minimum      |1.6    |1.6    |1.6    |1.6    |1.6    |1.6    |1.6    |
|separation   |       |       |       |       |       |       |       |
|degree       |       |       |       |       |       |       |       |
|between      |       |       |       |       |       |       |       |
|YA2304-16 and|       |       |       |       |       |       |       |
|adjacent     |       |       |       |       |       |       |       |
|peaks        |       |       |       |       |       |       |       |
|-------------|-------|-------|-------|-------|-------|-------|-------|
|Verdict: The system suitable solution was placed at 5℃ for 100h, and |
|the                                                                  |
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

