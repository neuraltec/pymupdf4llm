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

    # teste 1
    (
        "table_page6",
        6,
        0,
        """---------------------------------------
|Items         |Characteristics or    |
|              |properties            |
|--------------|----------------------|
|Appearance    |White to yellow powder|
|--------------|----------------------|
|Melting point |Approximately 252 ℃   |
|--------------|----------------------|
|Solubility    |Dissolved in methanol,|
|              |slightly soluble in   |
|              |ethanol, acetonitrile,|
|              |and acetone, slightly |
|              |soluble in            |
|              |isopropanol, almost   |
|              |insoluble in water    |
|--------------|----------------------|
|Hygroscopicity|It is non-hygroscopic.|
|--------------|----------------------|
|hydrate       |This product does not |
|              |contain crystal water.|
|--------------|----------------------|
|Dissociation  |4.39                  |
|constant      |                      |
|[(pKa)]       |                      |
|--------------|----------------------|
|partition     |LogD（1-Octanol/buffer|
|coefficient   |solution pH 2.4）=0.4 |
|              |LogD（1-Octanol/buffer|
|              |solution pH 7.4）=2.8 |
|--------------|----------------------|
|BCS           |ClassⅡ                |
|classification|                      |
---------------------------------------"""
    ),
    # teste 2
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
    # teste 3
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
    # teste 4
    (
        "table_page14",
        14,
        0,
        """--------------------------
|Structure|Sample        |
|         |information   |
|---------|--------------|
|Molecular|BatchNo.:     |
|formula: |231101 Assay: |
|C H N O  |100.1% Source |
|21 22 4 3|of sample:    |
|Molecular|Hinye         |
|weight:  |Pharmaceutical|
|378.43   |Co., Ltd. Test|
|g/mol    |items:        |
|         |LC-MS/MS, IR, |
|         |NMR, XRD, TGA |
|         |and DSC.      |
--------------------------"""
    ),
    # teste 5
    (
        "table_page15",
        15,
        0,
        """----------------------------------------------
|Batch  |Accurate|Theoretical|Elemental      |
|No.    |measured|value      |composition    |
|       |mass    |           |               |
|       |value   |           |               |
|-------|--------|-----------|---------------|
|231101 |379.18  |378.43     |[C21 H22 N4 O3 |
|       |        |           |+H]+           |
|       |        |           |               |
----------------------------------------------"""
    ),
     # teste 6
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
    # teste 7
    (
        "table_page16",
        16,
        0,
        """----------------------------------------
|2974.23,|υCH    |-CH3    |Carbon      |
|2953.02,|       |-CH2,	  |hydrogen    |
|28 35.36|       | -C, H  |stretching  |
|        |       |        |vibration   |
|--------|-------|--------|------------|
|2229.71 |υ C≡N  |-CN     |Carbon      |
|        |       |        |nitrogen    |
|        |       |        |triple bond |
|        |       |        |stretching  |
|        |       |        |vibration   |
|--------|-------|--------|------------|
|1683.86 |υ C=O  |-CONH2  |Carbon      |
|        |       |        |oxygen      |
|        |       |        |double bond |
|        |       |        |stretching  |
|        |       |        |vibration   |
|--------|-------|--------|------------|
|1660.71 |υ C=C  |-C=C    |Carbon      |
|        |       |        |carbon      |
|        |       |        |double bond |
|        |       |        |stretching  |
|        |       |        |vibration   |
|--------|-------|--------|------------|
|1606.70,|υ υ C=N|Pyridine|Expansion   |
|1573.91,|C=C ， |ring,   |and         |
|14 89.05|       |benzener|contraction |
|        |       |ing     |vibrations  |
|        |       |        |of carbon   |
|        |       |        |nitro gen   |
|        |       |        |and carbon  |
|        |       |        |carbon do   |
|        |       |        |uble bonds  |
|--------|-------|--------|------------|
|1463.97 |δCH    |-CH2    |In-plane    |
|        |       |        |shear       |
|        |       |        |vibration   |
|--------|-------|--------|------------|
|1454.33,|δCH    |-CH3    |Out of plane|
|1431.18,|       |        |deformationv|
|14      |       |        |ibration    |
|08.04,  |       |        |            |
|1381.03 |       |        |            |
|--------|-------|--------|------------|
|1267.23,|υ      |-OCH3   |Asymmetric  |
|1257.59 |=C-O-C |-OCH2   |stretching  |
|        |       |        |vibr ation  |
|        |       |        |of ether    |
|        |       |        |bond        |
|--------|-------|--------|------------|
|1138.00,|υ      |-OCH3   |Symmetric   |
|1031.92 |=C-O-C |-OCH2   |stretching  |
|        |       |        |vibrat ion  |
|        |       |        |of ether    |
|        |       |        |bond        |
----------------------------------------"""
    ),
    # teste 9
    (
        "table_page18",
        18,
        0,
        """-----------------------------------------------------
|Peak   |Chemical   |Multiplicity|Number |Attributed|
|No.    |shift (ppm)|            |of     |to        |
|       |           |            |protons|          |
|-------|-----------|------------|-------|----------|
|1      |1.029-1.064|t           |3H     |27-CH 3   |
|-------|-----------|------------|-------|----------|
|2      |2.118      |s           |3H     |28-CH 3   |
|-------|-----------|------------|-------|----------|
|3      |2.190      |s           |3H     |22-CH 3   |
|-------|-----------|------------|-------|----------|
|4      |3.824      |s           |3H     |24-CH 3   |
|-------|-----------|------------|-------|----------|
|5      |3.986-4.030|m           |2H     |26-CH 2   |
|-------|-----------|------------|-------|----------|
|6      |5.384      |s           |1H     |9-CH      |
|-------|-----------|------------|-------|----------|
|7      |6.714-6.804|bs          |2H     |20-NH 2   |
|-------|-----------|------------|-------|----------|
|8      |7.144-7.163|d           |1H     |4-CH      |
|-------|-----------|------------|-------|----------|
|9      |7.274-7.293|d           |1H     |5-CH      |
|-------|-----------|------------|-------|----------|
|10     |7.376      |s           |1H     |1-CH      |
|-------|-----------|------------|-------|----------|
|11     |7.551      |s           |1H     |16-CH     |
|-------|-----------|------------|-------|----------|
|12     |7.711      |s           |1H     |12-NH     |
-----------------------------------------------------"""
    ),
    # teste 10
    (
        "table_page20",
        20,
        0,
        """-------------------------------------
|Peak   |Chemical|Number |Attributed|
|No.    |shift   |ofC    |to        |
|       |(ppm)   |atoms  |          |
|-------|--------|-------|----------|
|1      |13.846  |1      |28-C      |
|-------|--------|-------|----------|
|2      |14.353  |1      |27-C      |
|-------|--------|-------|----------|
|3      |18.146  |1      |22-C      |
|-------|--------|-------|----------|
|4      |32.442  |1      |9-C       |
|-------|--------|-------|----------|
|5      |56.098  |1      |24-C      |
|-------|--------|-------|----------|
|6      |60.624  |1      |26-C      |
|-------|--------|-------|----------|
|7      |103.245 |1      |14-C      |
|-------|--------|-------|----------|
|8      |105.399 |1      |10-C      |
|-------|--------|-------|----------|
|9      |109.596 |1      |2-C       |
|-------|--------|-------|----------|
|10     |111.538 |1      |15-C      |
|-------|--------|-------|----------|
|11     |114.182 |1      |1-C       |
|-------|--------|-------|----------|
|12     |119.057 |1      |6-C       |
|-------|--------|-------|----------|
|13     |124.852 |1      |5-C       |
|-------|--------|-------|----------|
|14     |130.989 |1      |4-C       |
|-------|--------|-------|----------|
|15     |138.229 |1      |11-C      |
|-------|--------|-------|----------|
|16     |141.761 |1      |7-C       |
|-------|--------|-------|----------|
|17     |144.271 |1      |13-C      |
|-------|--------|-------|----------|
|18     |144.271 |1      |16-C      |
|-------|--------|-------|----------|
|19     |155.720 |1      |3-C       |
|-------|--------|-------|----------|
|20     |159.464 |1      |18-C      |
|-------|--------|-------|----------|
|21     |169.843 |1      |19-C      |
-------------------------------------"""
    ),
    # teste 11
    (
        "table_page23",
        23,
        0,
        """-----------------------------
|Lot No.|Endothermic process|
|       |-------------------|
|       |Temperature|Mass   |
|       |           |loss   |
|-------|-----------|-------|
|231101 |~ 35.0 ℃  |0.60%  |
|       |184.2 ℃   |       |
-----------------------------"""
    ),
    # teste 12
    (
        "table_page24",
        24,
        0,
        """---------------------------------------------
|	|242.7 ℃ ⁓ 438.8 ℃ | 97.30%	    |
---------------------------------------------"""
    ),
    # teste 13
    (
        "table_page24",
        24,
        1,
        """-----------------------------------------
|Batch  |endothermic process            |
|No     |                               |
|       |-------------------------------|
|       |The initial     |Peak          |
|       |temperature of  |temperature/℃|
|       |extrapolation/℃|              |
-----------------------------------------"""
    ),
    # teste 14
    (
        "table_page25",
        25,
        0,
        """---------------------------
|231101  |254.38 | 257.04 |
---------------------------"""
    ),
    # teste 15
    (
        "table_page26",
        26,
        0,
        """----------------------------------------------
|Elements |C      |H      |N      |O         |
|---------|-------|-------|-------|----------|
|Test     |66.57% |5.95%  |14.96% |12.52%    |
|results  |       |       |       |(This     |
|         |       |       |       |value is  |
|         |       |       |       |calculated|
|         |       |       |       |from the  |
|         |       |       |       |first     |
|         |       |       |       |three test|
|         |       |       |       |values)   |
|---------|-------|-------|-------|----------|
|Theoretic|66.65% |5.86%  |14.81% |12.68%    |
|al result|       |       |       |          |
----------------------------------------------"""
    ),
    # teste 16
    (
        "table_page29",
        29,
        0,
        """----------------------------
|Batch No. |Appearance     |
|----------|---------------|
|EMA       |White to yellow|
|Assessment|crystalline    |
|report    |non-hygroscopic|
|          |powder         |
|----------|---------------|
|231201    |White powder   |
|----------|---------------|
|240101    |White powder   |
|----------|---------------|
|240102    |White powder   |
----------------------------"""
    ),
    # teste 17
    (
        "table_page29",
        29,
        1,
        """-----------------------
|Batch  |Solubility   |
|No.    |             |
|-------|-------------|
|231201 |This product |
|       |is dissolved |
|       |in methanol, |
|       |slightly     |
|       |soluble in   |
|       |ethanol,     |
|       |acetonitrile |
|       |and acetone, |
|       |slightly     |
|       |soluble in   |
|       |isopropyl    |
|       |alcohol, and |
|       |almost       |
|       |insoluble in |
|       |water        |
|-------|-------------|
|240101 |This product |
|       |is dissolved |
|       |in methanol, |
|       |slightly     |
|       |soluble in   |
|       |ethanol,     |
|       |acetonitrile,|
|       |and acetone, |
|       |slightly     |
|       |soluble in   |
|       |isopropyl    |
|       |alcohol, and |
|       |almost       |
|       |insoluble in |
|       |water        |
|-------|-------------|
|240102 |This product |
|       |is dissolved |
|       |in methanol, |
|       |slightly     |
|       |soluble in   |
|       |ethanol,     |
|       |acetonitrile,|
|       |and acetone, |
|       |slightly     |
|       |soluble in   |
|       |isopropyl    |
|       |alcohol, and |
|       |almost       |
|       |insoluble in |
|       |water        |
-----------------------"""
    ),
    # teste 18
    (
        "table_page30",
        30,
        0,
        """---------------------------------------------------------------
|Batch  |Weight  |Sample |Weight of|Mass      |Results        |
|No.    |of blank|(g)    |blank    |increasing|               |
|       |vessel  |       |vessel   |percentage|               |
|       |(g)     |       |and      |(%)       |               |
|       |        |       |sample(g)|          |               |
|       |        |       |after    |          |               |
|       |        |       |standing |          |               |
|       |        |       |for 24h  |          |               |
|-------|--------|-------|---------|----------|---------------|
|231201 |20.63545|1.01424|21.64979 |0.01%     |non-hygroscopic|
|-------|--------|-------|---------|----------|---------------|
|240101 |22.77341|1.02158|23.79492 |-0.01%    |non-hygroscopic|
|-------|--------|-------|---------|----------|---------------|
|240102 |20.08668|1.19142|21.27804 |-0.01%    |non-hygroscopic|
|-------|--------|-------|---------|----------|---------------|
|Original formulation IF |non-hygroscopic                     |
|file                    |                                    |
---------------------------------------------------------------"""
    ),
    # teste 19
    (
        "table_page31",
        31,
        0,
        """----------------------------------------
|Batch  |Endothermic   		       |
|No.    |process      		       |
|       |---------------|--------------|
|	|Extrapolation 	|Peak 	       |
	|starting 	|temperature/℃|
|	|temperature/℃ |              |
|       |---------------|--------------|
|       |254.38 	|257.04        |
|-------|---------------|--------------|
|231201 |253.88 	|256.33        |
|-------|---------------|--------------|
|240101 |254.02 	|256.74        |
|-------|---------------|--------------|
|240102 |255.16 	|256.93        |
----------------------------------------"""
    ),
    # teste 20
    (
        "table_page32",
        32,
        0,
        """-------------------------------
|Batch  |Endothermic process  |
|No.    |                     |
|       |---------------------|
|       |Temperature/C|Mass   |
|       |             |loss % |
|-------|-------------|-------|
|231101 |184 ℃        |0.60% |
|-------|-------------|-------|
|231201 |175 ℃        |0.16% |
|-------|-------------|-------|
|240101 |175 ℃        |0.24% |
|-------|-------------|-------|
|240102 |175 ℃        |0.16% |
-------------------------------"""
    ),
    # teste 21
    (
        "table_page36",
        36,
        0,
        """-------------------------------------------
|Product   |CAS No.     |Formula|Molecular|
|name      |            |       |weight   |
|----------|------------|-------|---------|
|Finerenone|1050477-31-0|C21 H22|378.43   |
|          |            |N4 O3  |g/mol    | 
-------------------------------------------"""
    ),
    #table image
    # teste 22
    (
        "table_page87",
        87,
        0,
        """-----------------------------------------------------------------------
|Names of  |Chemical name   |Chemical |Impurity  |Control|Whether to  |
|impurities|                |structure|source    |limits |set quality |
|          |                |formula  |          |       |standards   |
|----------|----------------|---------|----------|-------|------------|
|YA2304-23 |4-bromo-2-hydrox|         |Impurity  |N/A    |no          |
|          |ybenzaldehyde   |         |introduced|       |(Including  |
|          |                |         |by        |       |warning     |
|          |                |         |starting  |       |structure,  |
|          |                |         |material  |       |see         |
|          |                |         |YA2304-1  |       |"3.2.S.3.2.5|
|          |                |         |          |       |Genotoxic   |
|          |                |         |          |       |Impurities" |
|          |                |         |          |       |section for |
|          |                |         |          |       |relevant    |
|          |                |         |          |       |control     |
|          |                |         |          |       |strategies) |
|----------|----------------|---------|----------|-------|------------|
|YA2304-24 |4-bromo-2-metho |         |Impurity  |N/A    |no          |
|          |xybenzaldehyde  |         |introduced|       |(Including  |
|          |                |         |by        |       |warning     |
|          |                |         |starting  |       |structure,  |
|          |                |         |material  |       |see         |
|          |                |         |YA2304-1  |       |"3.2.S.3.2.5|
|          |                |         |          |       |Genotoxic   |
|          |                |         |          |       |Impurities" |
|          |                |         |          |       |section for |
|          |                |         |          |       |relevant    |
|          |                |         |          |       |control     |
|          |                |         |          |       |strategies) |
|----------|----------------|---------|----------|-------|------------|
|YA2304-33 |4-cyano-2-hydrox|         |Impurity  |N/A    |no          |
|          |ybenzaldehyde   |         |introduced|       |(Including  |
|          |                |         |by        |       |warning     |
|          |                |         |starting  |       |structure,  |
|          |                |         |material  |       |see         |
|          |                |         |YA2304-1  |       |"3.2.S.3.2.5|
|          |                |         |          |       |Genotoxic   |
|          |                |         |          |       |Impurities" |
|          |                |         |          |       |section for |
|          |                |         |          |       |relevant    |
|          |                |         |          |       |control     |
|          |                |         |          |       |strategies) |
|----------|----------------|---------|----------|-------|------------|
|YA2304-43 |4-aminoformyl-2-|         |Impurity  |N/A    |no          |
|          |methoxybenzalde |         |introduced|       |(Including  |
|          |hyde            |         |by        |       |warning     |
|          |                |         |starting  |       |structure,  |
|          |                |         |material  |       |the relevant|
|          |                |         |YA2304-1  |       |control     |
|          |                |         |          |       |strategies  |
|          |                |         |          |       |are detailed|
|          |                |         |          |       |in section  |
-----------------------------------------------------------------------"""
    ),
    # teste 23
    (
        "table_page88",
        88,
        0,
        """---------------------------------------------------------------------
|    	  |                 |       |          |       |3.2.S.3.2.5 |
|    	  |                 |       |          |       |Genotoxic   |
|    	  |                 |       |          |       |Impurities) |                       
|-------------------------------------------------------------------|
|YA2304-38|4-methylene-2-ox |       |Impurity  |N/A    |no          |
|         |acyclobutanone   |       |introduced|       |(Including  |
|         |                 |       |by        |       |warning     |
|         |                 |       |starting  |       |structure,  |
|         |                 |       |material  |       |see         |
|         |                 |       |YA2304-2  |       |"3.2.S.3.2.5|
|         |                 |       |          |       |Genotoxic   |
|         |                 |       |          |       |Impurities" |
|         |                 |       |          |       |section for |
|         |                 |       |          |       |relevant    |
|         |                 |       |          |       |control     |
|         |                 |       |          |       |strategies) |
|---------|-----------------|-------|----------|-------|------------|
|YA2304-39|3-hydroxypropioni|       |Impurity  |N/A    |no (The     |
|         |trile            |       |introduced|       |impurity has|
|         |                 |       |by        |       |been        |
|         |                 |       |starting  |       |controlled  |
|         |                 |       |material  |       |by no more  |
|         |                 |       |YA2304-2  |       |than 5.0% in|
|         |                 |       |          |       |the starting|
|         |                 |       |          |       |material    |
|         |                 |       |          |       |YA2304-2 and|
|         |                 |       |          |       |is no longer|
|         |                 |       |          |       |controlled  |
|         |                 |       |          |       |in the      |
|         |                 |       |          |       |finished    |
|         |                 |       |          |       |product)    |
|---------|-----------------|-------|----------|-------|------------|
|YA2304-57|2-cyanoethyl     |       |Impurity  |N/A    |no (The     |
|         |acetate          |       |introduced|       |impurity has|
|         |                 |       |by        |       |been        |
|         |                 |       |starting  |       |controlled  |
|         |                 |       |material  |       |by no more  |
|         |                 |       |YA2304-2  |       |than 2.0% in|
|         |                 |       |          |       |the starting|
|         |                 |       |          |       |material    |
|         |                 |       |          |       |YA2304-2 and|
|         |                 |       |          |       |is no longer|
|         |                 |       |          |       |controlled  |
|         |                 |       |          |       |in the      |
|         |                 |       |          |       |finished    |
|         |                 |       |          |       |product)    |
---------------------------------------------------------------------"""
    ),
    # teste 24
    (
        "table_page89",
        89,
        0,
        """------------------------------------------------------------------------
|YA2304-40|2-chloro-5-methyl |       |Impurities  |NA     |no          |
|         |-4-nitropyridine-n|       |introduced  |       |(Including  |
|         |-oxide            |       |by starting |       |warning     |
|         |                  |       |material    |       |structure,  |
|         |                  |       |YA2304-4    |       |see         |
|         |                  |       |            |       |"3.2.S.3.2.5|
|         |                  |       |            |       |Genotoxic   |
|         |                  |       |            |       |Impurities" |
|         |                  |       |            |       |section for |
|         |                  |       |            |       |relevant    |
|         |                  |       |            |       |control     |
|         |                  |       |            |       |strategies) |
|---------|------------------|-------|------------|-------|------------|
|YA2304-41|2-chloro-4-amino- |       |Impurity    |N/A    |no          |
|         |5-methylpyridine  |       |introduced  |       |(Including  |
|         |                  |       |by starting |       |warning     |
|         |                  |       |material    |       |structure,  |
|         |                  |       |YA2304-4    |       |see         |
|         |                  |       |            |       |"3.2.S.3.2.5|
|         |                  |       |            |       |Genotoxic   |
|         |                  |       |            |       |Impurities" |
|         |                  |       |            |       |section for |
|         |                  |       |            |       |relevant    |
|         |                  |       |            |       |control     |
|         |                  |       |            |       |strategies) |
|---------|------------------|-------|------------|-------|------------|
|YA2304-1 |4-cyano-2-methox  |       |Incompletely|N/A    |no          |
|         |ybenzaldehyde     |       |reacted star|       |(Including  |
|         |                  |       |ting        |       |warning     |
|         |                  |       |material in |       |structure,  |
|         |                  |       |the prep    |       |see         |
|         |                  |       |aration of  |       |"3.2.S.3.2.5|
|         |                  |       |in          |       |Genotoxic   |
|         |                  |       |termediate Y|       |Impurities" |
|         |                  |       |A2304-5     |       |section for |
|         |                  |       |            |       |relevant    |
|         |                  |       |            |       |control     |
|         |                  |       |            |       |strategies) |
|---------|------------------|-------|------------|-------|------------|
|YA2304-2 |2-cyanoacetoaceta |       |Incompletely|N/A    |no (The     |
|         |te ethyl ester    |       |reacted star|       |impurity has|
|         |                  |       |ting        |       |been        |
|         |                  |       |material in |       |controlled  |
|         |                  |       |the prep    |       |not more    |
|         |                  |       |aration of  |       |than 0.5% in|
|         |                  |       |in          |       |the         |
|         |                  |       |termediate Y|       |intermediate|
|         |                  |       |A2304-5     |       |YA2304-5 and|
|         |                  |       |            |       |is no longer|
|         |                  |       |            |       |controlled  |
|         |                  |       |            |       |in the      |
|         |                  |       |            |       |finished    |
|         |                  |       |            |       |product)    |
------------------------------------------------------------------------"""
    ),
    # teste 25
    (
        "table_page90",
        90,
        0,
        """------------------------------------------------------------------------
|YA2304-3 |2-cyanoethyl      |       |An incomple |N/A    |no (The     |
|         |2-(4-cyano-2-met  |       |te intermedi|       |impurity has|
|         |hoxybenzyl)       |       |ate producti|       |been        |
|         |-3-oxy-butyrate   |       |nthe prepar |       |controlled  |
|         |                  |       |ation of    |       |less than   |
|         |                  |       |inte        |       |1.0% in the |
|         |                  |       |rmediate YA |       |intermediate|
|         |                  |       |2304-5      |       |YA2304-5 and|
|         |                  |       |            |       |is no longer|
|         |                  |       |            |       |controlled  |
|         |                  |       |            |       |in the      |
|         |                  |       |            |       |finished    |
|         |                  |       |            |       |product)    |
|---------|------------------|-------|------------|-------|------------|
|YA2304-4 |4-amino-5-methyl  |       |Incompletes |N/A    |no          |
|         |-2-hydroxypyridine|       |tarting mate|       |(Including  |
|         |                  |       |rial for    |       |warning     |
|         |                  |       |prep aration|       |structure,  |
|         |                  |       |of in       |       |see         |
|         |                  |       |termediate Y|       |"3.2.S.3.2.5|
|         |                  |       |A2304-5     |       |Genotoxic   |
|         |                  |       |            |       |Impurities" |
|         |                  |       |            |       |section for |
|         |                  |       |            |       |relevant    |
|         |                  |       |            |       |control     |
|         |                  |       |            |       |strategies) |
|---------|------------------|-------|------------|-------|------------|
|YA2304-47|4-amino-5-methyl  |       |Derived imp |N/A    |no          |
|         |-2-methoxypyridi  |       |urity of    |       |(Including  |
|         |ne                |       |start ing   |       |warning     |
|         |                  |       |material    |       |structure,  |
|         |                  |       |YA2304-4    |       |see         |
|         |                  |       |            |       |"3.2.S.3.2.5|
|         |                  |       |            |       |Genotoxic   |
|         |                  |       |            |       |Impurities" |
|         |                  |       |            |       |section for |
|         |                  |       |            |       |relevant    |
|         |                  |       |            |       |control     |
|         |                  |       |            |       |strategies) |
|---------|------------------|-------|------------|-------|------------|
|YA2304-5 |4 - (4 - cyano - 2|       |During thep |N/A    |no (The     |
|         |- methoxy phenyl) |       |reparation  |       |impurity has|
|         |- 2, 8 - dimethyl |       |of          |       |been        |
|         |- 5 -,4,5,6 oxygen|       |intermediate|       |controlled  |
|         |generation - 1-4 h|       |YA2304-8,   |       |not more    |
|         |- 1, 6 - nalidixic|       |the interme |       |than 0.3% in|
|         |- 3-2 - ethyl     |       |diate was   |       |the         |
|         |cyano formate     |       |not         |       |intermediate|
|         |                  |       |completely  |       |YA2304-8,   |
|         |                  |       |reacted     |       |and is no   |
|         |                  |       |            |       |longer      |
|         |                  |       |            |       |controlled  |
|         |                  |       |            |       |in the      |
|         |                  |       |            |       |finished    |
|         |                  |       |            |       |product)    |
------------------------------------------------------------------------"""
    ),
    # teste 26
    (
        "table_page91",
        91,
        0,
        """------------------------------------------------------------------------------
|YA2304-6     |4 - (4 - cyano - 2|       |Incompletei |N/A    |no (The       |
|             |- methoxy phenyl) |       |ntermediate |       |impurity has  |
|             |- 5 - ethoxy - 2, |       |product int |       |been          |
|             |8 - dimethyl - 1, |       |he preparati|       |controlled not|
|             |4-2 h - 1, 6 -    |       |on of interm|       |more than 0.3%|
|             |nalidixic - 3-2 - |       |ediate YA230|       |in the        |
|             |ethyl cyano       |       |4-8         |       |intermediate  |
|             |formate           |       |            |       |YA2304-8, and |
|             |                  |       |            |       |is no longer  |
|             |                  |       |            |       |controlled in |
|             |                  |       |            |       |the finished  |
|             |                  |       |            |       |product)      |
|-------------|------------------|-------|------------|-------|--------------|
|YA2304-7     |4-(4-cyano-2-met  |       |Incompletei |N/A    |no (The       |
|             |hoxyphenyl)       |       |ntermediate |       |impurity has  |
|             |-5-ethoxy-2,      |       |product int |       |been          |
|             |8-dimethyl-1,     |       |he preparati|       |controlled not|
|             |4-dihydro-1,      |       |on of interm|       |more than     |
|             |6-nalididine-3-car|       |ediate YA230|       |0.15% in the  |
|             |boxylic acid      |       |4-8         |       |intermediate  |
|             |                  |       |            |       |YA2304-8, and |
|             |                  |       |            |       |is no longer  |
|             |                  |       |            |       |controlled in |
|             |                  |       |            |       |the finished  |
|             |                  |       |            |       |product)      |
|-------------|------------------|-------|------------|-------|--------------|
|YA2304-55    |4 - (4 - cyano - 2|       |Incompletei |N/A    |no (The       |
|             |- methoxy phenyl) |       |ntermediate |       |impurity has  |
|             |- 2, 8 - dimethyl |       |product int |       |been          |
|             |- 5 -,4,5,6 oxygen|       |he preparati|       |controlled not|
|             |generation - 1-4 h|       |on of interm|       |more than     |
|             |- 1, 6 - nalidixic|       |ediate YA230|       |0.20% in the  |
|             |- 3 - carbonyl    |       |4-8         |       |intermediate  |
|             |imidazole         |       |            |       |YA2304-8, and |
|             |                  |       |            |       |is no longer  |
|             |                  |       |            |       |controlled in |
|             |                  |       |            |       |the finished  |
|             |                  |       |            |       |product)      |
|-------------|------------------|-------|------------|-------|--------------|
|4-dimethylami|4-dimethylaminop  |       |Reaction rea|N/A    |no (Including |
|nopyridine   |yridine (DMAP)    |       |gent in the |       |warning       |
|(DMAP)       |                  |       |synthesis of|       |structu re,   |
|             |                  |       |intermediate|       |see           |
|             |                  |       |YA2304-8    |       |"3.2.S.3.4.2.5|
|             |                  |       |            |       |Geno toxic    |
|             |                  |       |            |       |Impurities"   |
|             |                  |       |            |       |section for   |
|             |                  |       |            |       |relevant      |
|             |                  |       |            |       |control       |
|             |                  |       |            |       |strategie     |
------------------------------------------------------------------------------"""
    ),
    # teste 27
    (
        "table_page92",
        92,
        0,
        """------------------------------------------------------------------------
|         |                  |       |            |       |s)          |
|----------------------------------------------------------------------|
|YA2304-10|D- (+)            |       |Resolution  |≤0.10% |is (Finished|
|         |-dibenzoyltartaric|       |reagent in  |       |products are|
|         |acid              |       |the         |       |controlled  |
|         |                  |       |synthesis of|       |according to|
|         |                  |       |intermediate|       |specific    |
|         |                  |       |YA2304-CP   |       |impurities) |
|---------|------------------|-------|------------|-------|------------|
|YA2304-12|4 - (4 - amino    |       |Side        |≤0.10% |is          |
|         |formyl - 2 -      |       |reaction    |       |(Controlled |
|         |methoxy phenyl) - |       |product in  |       |by          |
|         |5 - ethoxy - 2, 8 |       |the         |       |non-specific|
|         |- dimethyl - 1,   |       |synthesis of|       |impurities  |
|         |4-2 h - 1, 6 -    |       |the         |       |in finished |
|         |nalidixic - 3 -   |       |intermediate|       |product)    |
|         |formamide         |       |YA2304-CP   |       |            |
|---------|------------------|-------|------------|-------|------------|
|YA2304-14|4 - (4 - cyano - 2|       |Side        |≤0.15% |is (Finished|
|         |- methoxy phenyl) |       |reaction    |       |products are|
|         |- 5 - methoxy - 2,|       |products    |       |controlled  |
|         |8 - dimethyl - 1, |       |            |       |by specific |
|         |4-2 h - 1, 6 -    |       |            |       |impurities) |
|         |nalidixic - 3 -   |       |            |       |            |
|         |formamide         |       |            |       |            |
|---------|------------------|-------|------------|-------|------------|
|YA2304-15|4 - (4 - cyano - 2|       |Side        |≤0.15% |is (Finished|
|         |- methoxy phenyl) |       |reaction    |       |products are|
|         |- 5-2 - isopropyl |       |products    |       |controlled  |
|         |oxygen radicals, 8|       |            |       |by specific |
|         |- dimethyl - 1,   |       |            |       |impurities) |
|         |4-2               |       |            |       |            |
------------------------------------------------------------------------"""
    ),
    # teste 28
    (
        "table_page93",
        93,
        0,
        """------------------------------------------------------------------------
|         |h - 1, 6 -        |       |            |       |            |
|         |nalidixic - 3 -   |       |            |       |            |
|         |formamide         |       |            |       |            |
|----------------------------------------------------------------------|
|YA2304-16|4-(4-cyano-2-met  |       |Side        |≤0.10% |is          |
|         |hoxyphenyl)       |       |reaction    |       |(Controlled |
|         |-5-methoxy-2,     |       |products    |       |by          |
|         |8-dimethyl-1,     |       |            |       |non-specific|
|         |4-dihydro-1,      |       |            |       |impurities  |
|         |6-nalididine-3-car|       |            |       |in finished |
|         |boxylic acid      |       |            |       |product)    |
|---------|------------------|-------|------------|-------|------------|
|YA2304-17|Benzoic acid      |       |Aby-product |N/A    |no (The     |
|         |                  |       |in the      |       |impurity is |
|         |                  |       |synthesis of|       |controlled  |
|         |                  |       |the         |       |asa specific|
|         |                  |       |intermediate|       |impurity in |
|         |                  |       |YA2304-CP   |       |the crude   |
|         |                  |       |            |       |product, the|
|         |                  |       |            |       |limit is not|
|         |                  |       |            |       |more than   |
|         |                  |       |            |       |0.15%, and  |
|         |                  |       |            |       |the finished|
|         |                  |       |            |       |product is  |
|         |                  |       |            |       |no longer   |
|         |                  |       |            |       |controlled) |
|---------|------------------|-------|------------|-------|------------|
|YA2304-52|D-(+)- monobenzoyl|       |Aby-product |N/A    |no (Has been|
|         |tartaric acid     |       |in the      |       |in the      |
|         |                  |       |synthesis of|       |finished    |
|         |                  |       |the         |       |Finerenone  |
|         |                  |       |intermediate|       |through     |
|         |                  |       |YA2304-CP   |       |multiple    |
|         |                  |       |            |       |batches of  |
|         |                  |       |            |       |statistical |
|         |                  |       |            |       |proof can be|
|         |                  |       |            |       |completely  |
|         |                  |       |            |       |removed, the|
|         |                  |       |            |       |finished    |
|         |                  |       |            |       |product is  |
|         |                  |       |            |       |no longer   |
|         |                  |       |            |       |controlled) |
------------------------------------------------------------------------"""
    ),
    # teste 29
    (
        "table_page94",
        94,
        0,
        """------------------------------------------------------------------------
|YA2304-18|4 - (4 - cyano - 2|       |Side        |≤0.10% |is          |
|         |- methoxy phenyl) |       |reaction    |       |(Controlled |
|         |- 5-2 - isopropyl |       |products    |       |by          |
|         |oxygen radicals, 8|       |            |       |non-specific|
|         |- dimethyl - 1,   |       |            |       |impurities  |
|         |4-2 h - 1, 6 -    |       |            |       |in finished |
|         |nalidixic - 3 -   |       |            |       |product)    |
|         |formic acid       |       |            |       |            |
|---------|------------------|-------|------------|-------|------------|
|YA2304-19|4-(4-cyano-2-met  |       |Side        |≤0.15% |is (Finished|
|         |hoxyphenyl)       |       |reaction    |       |products are|
|         |-5-ethoxy-2,      |       |products    |       |controlled  |
|         |8-dimethyl-1,     |       |            |       |by specific |
|         |6-nalididine-3-for|       |            |       |impurities) |
|         |mamide            |       |            |       |            |
|---------|------------------|-------|------------|-------|------------|
|YA2304-20|4 - (4 - (4 r) -  |       |enantiomers |≤0.15% |is (Finished|
|         |cyano - 2 -       |       |            |       |products are|
|         |methoxy phenyl) - |       |            |       |controlled  |
|         |5 - ethoxy - 2, 8 |       |            |       |by specific |
|         |- dimethyl - 1,   |       |            |       |impurities) |
|         |4-2 h - 1, 6 -    |       |            |       |            |
|         |nalidixic - 3 -   |       |            |       |            |
|         |formamide         |       |            |       |            |
|---------|------------------|-------|------------|-------|------------|
|imidazole|Imidazole         |       |Reaction    |N/A    |no (has been|
|         |                  |       |byproducts  |       |completely  |
|         |                  |       |and         |       |removed in  |
|         |                  |       |hydrolysates|       |the finished|
|         |                  |       |of CDI      |       |fineridone  |
|         |                  |       |            |       |through     |
|         |                  |       |            |       |multiple    |
|         |                  |       |            |       |batches of  |
|         |                  |       |            |       |statistical |
|         |                  |       |            |       |proof, no   |
|         |                  |       |            |       |longer      |
|         |                  |       |            |       |controlled  |
|         |                  |       |            |       |in the      |
|         |                  |       |            |       |finished    |
|         |                  |       |            |       |product)    |
------------------------------------------------------------------------"""
    ),
    # teste 30
    (
        "table_page95",
        95,
        0,
        """------------------------------------------------------------------------
|Dimethyl    |Dimethyl sulfate |       |Side    |≤75ppm |is (Including |
|sulfate     |                 |       |reaction|       |the warning   |
|            |                 |       |products|       |structure, the|
|            |                 |       |        |       |relevant      |
|            |                 |       |        |       |control       |
|            |                 |       |        |       |strategy is   |
|            |                 |       |        |       |detailed in   |
|            |                 |       |        |       |the section   |
|            |                 |       |        |       |"3.2.S.3.4.2.5|
|            |                 |       |        |       |Genotoxic     |
|            |                 |       |        |       |Impurities")  |
|------------|-----------------|-------|--------|-------|--------------|
|Diethyl     |Diethyl sulfate  |       |Side    |≤75ppm |is (Including |
|sulfate     |                 |       |reaction|       |the warning   |
|            |                 |       |products|       |structure, the|
|            |                 |       |        |       |relevant      |
|            |                 |       |        |       |control       |
|            |                 |       |        |       |strategy is   |
|            |                 |       |        |       |detailed in   |
|            |                 |       |        |       |the section   |
|            |                 |       |        |       |"3.2.S.3.4.2.5|
|            |                 |       |        |       |Genotoxic     |
|            |                 |       |        |       |Impurities")  |
|------------|-----------------|-------|--------|-------|--------------|
|Diisopropyl |Diisopropyl      |       |Side    |≤75ppm |is (Including |
|sulfate     |sulfate          |       |reaction|       |the warning   |
|            |                 |       |products|       |structure, the|
|            |                 |       |        |       |relevant      |
|            |                 |       |        |       |control       |
|            |                 |       |        |       |strategy is   |
|            |                 |       |        |       |detailed in   |
|            |                 |       |        |       |the section   |
|            |                 |       |        |       |"3.2.S.3.4.2.5|
|            |                 |       |        |       |Genotoxic     |
|            |                 |       |        |       |Impurities")  |
|------------|-----------------|-------|--------|-------|--------------|
|Di-sec-butyl|Di-sec-butyl     |       |Side    |≤75ppm |is (Including |
|sulfate     |sulfate          |       |reaction|       |the warning   |
|            |                 |       |products|       |structure, the|
|            |                 |       |        |       |relevant      |
|            |                 |       |        |       |control       |
|            |                 |       |        |       |strategy is   |
|            |                 |       |        |       |detailed in   |
|            |                 |       |        |       |the section   |
|            |                 |       |        |       |"3.2.S.3.4.2.5|
|            |                 |       |        |       |Genotoxic     |
|            |                 |       |        |       |Impurities")  |
|------------|-----------------|-------|--------|-------|--------------|
|NDMA        |N-nitrosodimethyl|N N O  |Side    |N/A    |no (No longer |
|            |amine            |       |reaction|       |controlled in |
|            |                 |       |product |       |finished      |
|            |                 |       |        |       |fineridone    |
|            |                 |       |        |       |after         |
------------------------------------------------------------------------"""
    ),
    # teste 31
    (
        "table_page96",
        96,
        0,
        """--------------------------------------------------------------------
|       |                  |       |        |       |multiple batch| 
|       |                  |       |        |       |statistics    |
|       |                  |       |        |       |prove         |
|       |                  |       |        |       |non-existence)|    
|------------------------------------------------------------------|
|NDEA   |N-nitrosodiethyla |N N O  |Side    |N/A    |no (No longer |
|       |mine              |       |reaction|       |controlled in |
|       |                  |       |products|       |finished      |
|       |                  |       |        |       |fineridone    |
|       |                  |       |        |       |after multiple|
|       |                  |       |        |       |batch         |
|       |                  |       |        |       |statistics    |
|       |                  |       |        |       |prove         |
|       |                  |       |        |       |non-existence)|
|-------|------------------|-------|--------|-------|--------------|
|NDIPA  |N-nitrosodiisopro |N N O  |Side    |N/A    |no (No longer |
|       |pylamine          |       |reaction|       |controlled in |
|       |                  |       |products|       |finished      |
|       |                  |       |        |       |fineridone    |
|       |                  |       |        |       |after multiple|
|       |                  |       |        |       |batch         |
|       |                  |       |        |       |statistics    |
|       |                  |       |        |       |prove         |
|       |                  |       |        |       |non-existence)|
|-------|------------------|-------|--------|-------|--------------|
|NDBA   |N-nitrosodibutyla |N N O  |Side    |N/A    |no (No longer |
|       |mine              |       |reaction|       |controlled in |
|       |                  |       |products|       |finished      |
|       |                  |       |        |       |fineridone    |
|       |                  |       |        |       |after multiple|
|       |                  |       |        |       |batch         |
|       |                  |       |        |       |statistics    |
|       |                  |       |        |       |prove         |
|       |                  |       |        |       |non-existence)|
|-------|------------------|-------|--------|-------|--------------|
|NIEPA  |N-nitrosoethylisop|N N O  |Side    |N/A    |no (No longer |
|       |ropylamine        |       |reaction|       |controlled in |
|       |                  |       |products|       |finished      |
|       |                  |       |        |       |fineridone    |
|       |                  |       |        |       |after multiple|
|       |                  |       |        |       |batch         |
|       |                  |       |        |       |statistics    |
|       |                  |       |        |       |prove         |
|       |                  |       |        |       |non-existence)|
|-------|------------------|-------|--------|-------|--------------|
|NMBA   |N-nitroso-methyl- |OOH N N|Side    |N/A    |no (No longer |
|       |4-aminobutyric    |O      |reaction|       |controlled in |
|       |acid              |       |products|       |finished      |
|       |                  |       |        |       |fineridone    |
|       |                  |       |        |       |after multiple|
|       |                  |       |        |       |batch         |
|       |                  |       |        |       |statistics    |
|       |                  |       |        |       |prove         |
|       |                  |       |        |       |non-existence)|
--------------------------------------------------------------------"""
    ),
    # teste 32
    (
        "table_page97",
        97,
        0,
        """--------------------------------------------------------------------
|NMPA   |N-nitrosotoluidine|N N O  |Side    |N/A    |no (No longer |
|       |                  |       |reaction|       |controlled in |
|       |                  |       |products|       |finished      |
|       |                  |       |        |       |fineridone    |
|       |                  |       |        |       |after multiple|
|       |                  |       |        |       |batch         |
|       |                  |       |        |       |statistics    |
|       |                  |       |        |       |prove         |
|       |                  |       |        |       |non-existence)|
|-------|------------------|-------|--------|-------|--------------|
|NDPA   |N-nitrosodipropyl |O H C 3|Side    |N/A    |no (No longer |
|       |amine             |N H C N|reaction|       |controlled in |
|       |                  |3      |products|       |finished      |
|       |                  |       |        |       |fineridone    |
|       |                  |       |        |       |after multiple|
|       |                  |       |        |       |batch         |
|       |                  |       |        |       |statistics    |
|       |                  |       |        |       |prove         |
|       |                  |       |        |       |non-existence)|
--------------------------------------------------------------------"""
    ),
    # teste 33
    (
        "table_page112",
        112,
        0,
        """-------------------------------------------------
|Impurity |Batch No. |231201  |240101  |240102  |
|name	  |          |        |        |        |
|--------------------|--------|--------|--------|
|Related   |YA2304-10|Not     |Not     |Not     |
|subs tance|         |detected|detected|detected|
|	   |---------|--------|--------|--------|
|          |YA2304-12|Not     |Not     |Not     |
|          |         |detected|detected|detected|
|	   |---------|--------|--------|--------|
|          |YA2304-14|Not     |Not     |Not     |
|          |         |detected|detected|detected|
|	   |---------|--------|--------|--------|
|          |YA2304-15|Not     |Not     |< 0.05% |
|          |         |detected|detected|        |
|	   |---------|--------|--------|--------|
|          |YA2304-16|Not     |Not     |Not     |
|          |         |detected|detected|detected|
|	   |---------|--------|--------|--------|
|          |YA2304-17|Not     |Not     |Not     |
|          |         |detected|detected|detected|
|	   |---------|--------|--------|--------|
|          |YA2304-18|Not     |Not     |Not     |
|          |         |detected|detected|detected|
|	   |---------|--------|--------|--------|
|          |YA2304-19|< 0.05% |< 0.05% |< 0.05% |
|	   |---------|--------|--------|--------|
|          |Other    |< 0.05% |< 0.05% |< 0.05% |
|          |singlei  |        |        |        |
|          |mpurities|        |        |        |
|	   |---------|--------|--------|--------|
|          |Total    |< 0.05% |< 0.05% |< 0.05% |
|          |Miscella |        |        |        |
|          |neous    |        |        |        |
|----------|---------|--------|--------|--------|
|Related   |YA2304-52|Not     |Not     |Not     |
|Sub stance|         |detected|detected|detected|
|II        |         |        |        |        |
|	   |---------|--------|--------|--------|
|          |Imidazole|Not     |Not     |Not     |
|          |         |detected|detected|detected|
|----------|---------|--------|--------|--------|
|Enantiomer|YA2304-20|0.03%   |0.03%   |0.02%   |
-------------------------------------------------"""
    ),
    # teste 34
    (
        "table_page113",
        113,
        0,
        """--------------------------------------------------
|Names of  |Impurity source|Control     |Is the  |
|Impurities|and removal    |Limits      |negation|
|          |analysis       |            |of entry|
|          |               |            |criteria|
|----------|---------------|------------|--------|
|Sodium    |Sodium acetate |Incandescent|is      |
|acetate   |is the reagent |residue     |        |
|          |used in step 2.|≤0.1%       |        |
|          |This impurity  |            |        |
|          |is soluble in  |            |        |
|          |water and can  |            |        |
|          |be removed by  |            |        |
|          |subsequent     |            |        |
|          |washing and    |            |        |
|          |crystallization|            |        |
|          |processes. The |            |        |
|          |applicant      |            |        |
|          |intends to     |            |        |
|          |control sodium |            |        |
|          |acetate by     |            |        |
|          |burning        |            |        |
|          |residue.       |            |        |
|----------|---------------|------------|--------|
|Sulfuric  |Sulfuric acid  |Incandescent|is      |
|acid      |is the reagent |residue     |        |
|          |used in step 2 |≤0.1%       |        |
|          |and is         |            |        |
|          |converted to   |            |        |
|          |sodium sulfate.|            |        |
|          |Sodium sulfate |            |        |
|          |dissolves      |            |        |
|          |easily in water|            |        |
|          |and can be     |            |        |
|          |removed by     |            |        |
|          |subsequent     |            |        |
|          |washing and    |            |        |
|          |crystallization|            |        |
|          |processes. The |            |        |
|          |applicant      |            |        |
|          |intends to     |            |        |
|          |control sodium |            |        |
|          |sulfate by     |            |        |
|          |burning        |            |        |
|          |residue.       |            |        |
|----------|---------------|------------|--------|
|Sodium    |Sodium         |Incandescent|is      |
|hydroxide |hydroxide is   |residue     |        |
|          |the reagent    |≤0.1%       |        |
|          |used in step 2,|            |        |
|          |which can be   |            |        |
|          |converted into |            |        |
|          |sodium         |            |        |
|          |chloride.      |            |        |
|          |Sodium chloride|            |        |
|          |is soluble in  |            |        |
|          |water and can  |            |        |
|          |be removed by  |            |        |
|          |subsequent     |            |        |
|          |washing and    |            |        |
|          |crystallization|            |        |
|          |processes. The |            |        |
|          |applicant      |            |        |
|          |intends to     |            |        |
|          |control sodium |            |        |
|          |chloride by    |            |        |
|          |burning        |            |        |
|          |residue.       |            |        |
|----------|---------------|------------|--------|
|Hydrochlo |Hydrochloric   |Incandescent|is      |
|ric acid  |acid is the    |residue     |        |
|          |reagent used in|≤0.1%       |        |
|          |step 2 and is  |            |        |
|          |converted to   |            |        |
|          |sodium         |            |        |
|          |chloride.      |            |        |
|          |Sodium chloride|            |        |
|          |is soluble in  |            |        |
|          |water and can  |            |        |
|          |be removed by  |            |        |
|          |subsequent     |            |        |
|          |washing and    |            |        |
|          |crystallization|            |        |
|          |processes. The |            |        |
|          |applicant      |            |        |
|          |intends to     |            |        |
|          |control sodium |            |        |
|          |chloride by    |            |        |
|          |burning        |            |        |
|          |residue.       |            |        |
|----------|---------------|------------|--------|
|Sodium    |Sodium         |Burning     |is      |
|phosphate |phosphate is   |residue     |        |
|          |the reagent    |≤0.1%       |        |
|          |used in Step 3 |            |        |
|          |that is        |            |        |
|          |partially      |            |        |
|          |converted to   |            |        |
|          |convert        |            |        |
|          |disodium       |            |        |
|          |hydrogen       |            |        |
|          |phosphate after|            |        |
|          |the base is    |            |        |
|          |modulated in   |            |        |
|          |step 3. The    |            |        |
|          |impurities of  |            |        |
--------------------------------------------------"""
    ),
    # teste 35
    (
        "table_page114",
        114,
        0,
        """-----------------------------------------------------------------
|	|disodium hydrogen phosphate	 |		|	|
|	|and sodium phosphate are 	 |		|	|
|	|soluble in water and can be	 |		|	|
|	|removed by subsequent washing 	 |		|	|
|	|and crystallization processes.	 |		|	|
|	|The applicant intends to control|		|	|
|	|disodium hydrogen phosphate and |		|	|
|	|sodium phosphate through 	 |		|	|
|	|incandescent residue.		 | 		|	|
-----------------------------------------------------------------"""
    ),
    # teste 36
    (
        "table_page114",
        114,
        1,
        """-------------------------------------------
|	   |Batch No.|231201|240101|240102|
|----------|-------------------------------
|Test items|				  |
-------------------------------------------"""
    ),
    # teste 37
    (
        "table_page115",
        115,
        0,
        """-----------------------------------------
|Incandescent|conforms|conforms|conforms|
|residue     |        |        |        |
|≤0.1%       |        |        |        |
|------------|--------|--------|--------|
|Chloride    |conforms|conforms|conforms|
|≤0.02%      |        |        |        |
|------------|--------|--------|--------|
|Sulfate     |conforms|conforms|conforms|
|≤0.1%       |        |        |        |
-----------------------------------------"""
    ),
    # teste 38
    (
        "table_page116",
        116,
        0,
        """---------------------------------------------------
|Name of solvent    |Steps to use |Solvent|Limits |
|                   |             |classif|       |
|                   |             |ication|       |
|-------------------|-------------|-------|-------|
|Ethyl acetate      |Starting     |Class 3|≤0.5%  |
|                   |material     |       |       |
|                   |YA2304-2     |       |       |
|                   |isintroduced,|       |       |
|                   |reagent      |       |       |
|                   |triethyl     |       |       |
|                   |orthoacetate |       |       |
|                   |reaction andh|       |       |
|                   |ydrolysis    |       |       |
|-------------------|-------------|-------|-------|
|Isopropyl alcohol  |Steps 1      |Class 3|≤0.5%  |
|-------------------|-------------|-------|-------|
|sec-butanol        |Step 1       |Class 3|≤0.5%  |
|-------------------|-------------|-------|-------|
|N-methylpyrrolidone|Step 2       |Class 2|≤0.053%|
|-------------------|-------------|-------|-------|
|Toluene            |Step 2       |Class 2|≤0.089%|
|-------------------|-------------|-------|-------|
|tetrahydrofuran    |Step 2       |Class 2|≤0.072%|
|-------------------|-------------|-------|-------|
|Ethanol            |Steps 1,3,4  |Class 3|≤0.5%  |
|-------------------|-------------|-------|-------|
|Piperidine         |Steps 1      |Class 4|≤0.1%  |
---------------------------------------------------"""
    ),
    # teste 39
    (
        "table_page117",
        117,
        0,
        """---------------------------------------------
|benzene|Toluene and     |Class 1 |≤0.0002% |
|	|ethanol introdu |        |         |
|	|ced		 |	  |         |
---------------------------------------------"""
    ),
    # teste 40
    (
        "table_page118",
        118,
        0,
        """------------------------------------------------------------------------
|Batch   |Limits |10% of  |231201  |240101  |240102  |LOD     |LOQ     |
|        |       |limits  |        |        |        |        |        |
|--------|-------|--------|--------|--------|--------|--------|--------|
|Ethanol |≤0. 5% |≤0.05%  |0.1%    |0.2%    |0.1%    |0.003%  |0.01%   |
|--------|-------|--------|--------|--------|--------|--------|--------|
|Isoprop |≤0. 5% |≤0.05%  |Not     |Not     |Not     |0.003%  |0.01%   |
|yl alco |       |        |detected|detected|detected|        |        |
|hol     |       |        |        |        |        |        |        |
|--------|-------|--------|--------|--------|--------|--------|--------|
|Ethyla  |≤0. 5% |≤0.05%  |Not     |Not     |Not     |0.003%  |0.01%   |
|cetate  |       |        |detected|detected|detected|        |        |
|--------|-------|--------|--------|--------|--------|--------|--------|
|sec-but |≤0. 5% |≤0.05%  |Not     |Not     |Not     |0.002%  |0.008%  |
|anol    |       |        |detected|detected|detected|        |        |
|--------|-------|--------|--------|--------|--------|--------|--------|
|tetrahy |≤0.07  |≤0.0072%|Not     |Not     |Not     |0.002%  |0.007%  |
|drofuran|2%     |        |detected|detected|detected|        |        |
|--------|-------|--------|--------|--------|--------|--------|--------|
|Toluene |≤0.08  |≤0.0089%|Not     |Not     |Not     |0.001%  |0.004%  |
|        |9%     |        |detected|detected|detected|        |        |
|--------|-------|--------|--------|--------|--------|--------|--------|
|N-meth  |≤0.05  |≤0.0053%|Not     |Not     |Not     |0.002%  |0.006%  |
|ylpyrrol|3%     |        |detected|detected|detected|        |        |
|idone   |       |        |        |        |        |        |        |
|--------|-------|--------|--------|--------|--------|--------|--------|
|piperidi|≤0. 1% |≤0.01%  |Not     |Not     |Not     |0.003%  |0.01%   |
|ne      |       |        |detected|detected|detected|        |        |
|--------|-------|--------|--------|--------|--------|--------|--------|
|benzene |≤0.00  |≤0.00002|Not     |Not     |Not     |0.000016|0.000053|
|        |02%    |%       |detected|detected|detected|%       |%       |
------------------------------------------------------------------------"""
    ),
    # teste 41
    (
        "table_page120",
        120,
        0,
        """-----------------------------------------------
|Batch No. |231201 |240101 |240102 |10% of the|
|Elemental |       |       |       |acceptable|
|impurities|       |       |       |limit     |
|----------|-------|-------|-------|----------|
|Palladium |0.053  |0.084  |0.009  |1ppm      |
|content   |ppm    |ppm    |ppm    |          |
-----------------------------------------------"""
    ),
    # teste 42
    (
        "table_page121",
        121,
        0,
        """-----------------------------------------------
|Batch No. |231201 |240101 |240102 |30% of the|
|Elemental |       |       |       |acceptable|
|impurities|       |       |       |limit     |
|----------|-------|-------|-------|----------|
|Lead      |0.380  |0.386  |0.093  |1.5 ppm   |
|content   |ppm    |ppm    |ppm    |          |
-----------------------------------------------"""
    ),
    # teste 43
    (
        "table_page122",
        122,
        0,
        """-------------------------------------------------------------
|Batch No. |231201    |240101  |240102    |LODs   |30% ofa  |
|Elemental |          |        |          |       |cceptable|
|impurities|          |        |          |       |limits   |
|----------|----------|--------|----------|-------|---------|
|Arsenic   |Undetected|Not     |Not       |0.004  |0.45 ppm |
|levels    |          |detected|detected  |ppm    |         |
|----------|----------|--------|----------|-------|---------|
|Cadmium   |Undetected|Not     |Not       |0.001  |0.15 ppm |
|content   |          |detected|detected  |ppm    |         |
|----------|----------|--------|----------|-------|---------|
|Mercury   |Undetected|Not     |Not       |0.020  |0.9 ppm  |
|content   |          |detected|detected  |ppm    |         |
|----------|----------|--------|----------|-------|---------|
|Cobalt    |0.005 ppm |0.028   |Undetected|0.001  |1.5 ppm  |
|content   |          |ppm     |          |ppm    |         |
|----------|----------|--------|----------|-------|---------|
|Nickel    |0.174 ppm |0.351   |0.088 ppm |0.011  |6ppm     |
|content   |          |ppm     |          |ppm    |         |
|----------|----------|--------|----------|-------|---------|
|Vanadium  |Undetected|0.004   |Undetected|0.002  |3ppm     |
|content   |          |ppm     |          |ppm    |         |
-------------------------------------------------------------"""
    ),
    # teste 44
    (
        "table_page123",
        123,
        0,
        """------------------------------------
|Batch  |231201  |240101  |240102  |
|No.    |        |        |        |
|-------|--------|--------|--------|
|≤10ppm |conforms|conforms|conforms|
------------------------------------"""
    ),
    # teste 45
    (
        "table_page129",
        129,
        0,
        """------------------------------------------------
|Batch No.|231201    |240101  |240102  |LOD    |
|---------|----------|--------|--------|-------|
|YA2304-38|Undetected|Not     |Not     |0.027% |
|         |          |detected|detected|       |
------------------------------------------------"""
    ),
    # teste 46
    (
        "table_page137",
        137,
        0,
        """-"""
    ),
    # teste 47
    (
        "table_page137",
        137,
        1,
        """-"""
    ),
    # teste 48
    (
        "table_page139",
        139,
        0,
        """-"""
    ),
    # teste 49
    (
        "table_page141",
        141,
        0,
        """-"""
    ),
    # teste 50
    (
        "table_page141",
        141,
        1,
        """-"""
    ),
    # teste 51
    (
        "table_page142",
        142,
        0,
        """-"""
    ),
    # teste 52
    (
        "table_page143",
        143,
        0,
        """-"""
    ),
    # teste 53
    (
        "table_page144",
        144,
        0,
        """-"""
    ),
    # teste 54
    (
        "table_page147",
        147,
        0,
        """-"""
    ),
    # teste 55
    (
        "table_page148",
        148,
        0,
        """-"""
    ),
    # teste 56
    (
        "table_page150",
        150,
        0,
        """-"""
    ),
    # teste 57
    (
        "table_page155",
        155,
        0,
        """-"""
    ),
    # teste 58
    (
        "table_page158",
        158,
        0,
        """-"""
    ),
    # teste 59
    (
        "table_page159",
        159,
        0,
        """-"""
    ),
    # teste 60
    (
        "table_page163",
        163,
        0,
        """-"""
    ),
    # teste 61
    (
        "table_page164",
        164,
        0,
        """-"""
    ),
    # teste 62
    (
        "table_page165",
        165,
        0,
        """-"""
    ),
    # teste 63
    (
        "table_page166",
        166,
        0,
        """-"""
    ),
    # teste 64
    (
        "table_page171",
        171,
        0,
        """-"""
    ),
    # teste 65
    (
        "table_page172",
        172,
        0,
        """-"""
    ),
    # teste 66
    (
        "table_page173",
        173,
        0,
        """-"""
    ),
    # teste 67
    (
        "table_page174",
        174,
        0,
        """-"""
    ),
    # teste 68
    (
        "table_page175",
        175,
        0,
        """-"""
    ),
    # teste 69
    (
        "table_page178",
        178,
        0,
        """-"""
    ),
    # teste 70
    (
        "table_page179",
        179,
        0,
        """-"""
    ),
    # teste 71
    (
        "table_page180",
        180,
        0,
        """-"""
    ),
    # teste 72
    (
        "table_page181",
        181,
        0,
        """-"""
    ),
    # teste 73
    (
        "table_page182",
        182,
        0,
        """-"""
    ),
    # teste 74
    (
        "table_page183",
        183,
        0,
        """-"""
    ),
    # teste 75
    (
        "table_page184",
        184,
        0,
        """-"""
    ),
    # teste 76
    (
        "table_page185",
        185,
        0,
        """-"""
    ),
    # teste 77
    (
        "table_page185",
        185,
        1,
        """-"""
    ),
    # teste 78
    (
        "table_page186",
        186,
        0,
        """-"""
    ),
    # teste 79
    (
        "table_page187",
        187,
        0,
        """-"""
    ),
    # teste 80
    (
        "table_page190",
        190,
        0,
        """-"""
    ),
    # teste 81
    (
        "table_page191",
        191,
        0,
        """-"""
    ),
    # teste 82
    (
        "table_page192",
        192,
        0,
        """-"""
    ),
    # teste 83
    (
        "table_page193",
        193,
        0,
        """-"""
    ),
    # teste 84
    (
        "table_page194",
        194,
        0,
        """-"""
    ),
    # teste 85
    (
        "table_page195",
        195,
        0,
        """-"""
    ),
    # teste 86
    (
        "table_page196",
        196,
        0,
        """-"""
    ),
    # teste 87
    (
        "table_page197",
        197,
        0,
        """-"""
    ),
    # teste 88
    (
        "table_page198",
        198,
        0,
        """-"""
    ),
    # teste 89
    (
        "table_page199",
        199,
        0,
        """-"""
    ),
    # teste 90
    (
        "table_page199",
        199,
        1,
        """-"""
    ),
    # teste 91
    (
        "table_page200",
        200,
        0,
        """-"""
    ),
    # teste 92
    (
        "table_page204",
        204,
        0,
        """-"""
    ),
    # teste 93
    (
        "table_page205",
        205,
        0,
        """-"""
    ),
    # teste 94
    (
        "table_page206",
        206,
        0,
        """-"""
    ),
    # teste 95
    (
        "table_page207",
        207,
        0,
        """-"""
    ),
    # teste 96
    (
        "table_page208",
        208,
        0,
        """-"""
    ),
    # teste 97
    (
        "table_page209",
        209,
        0,
        """-"""
    ),
    # teste 98
    (
        "table_page210",
        210,
        0,
        """-"""
    ),
    # teste 99
    (
        "table_page211",
        211,
        0,
        """-"""
    ),
    # teste 100
    (
        "table_page214",
        214,
        0,
        """-"""
    ),
    # teste 101
    (
        "table_page215",
        215,
        0,
        """-"""
    ),
    # teste 102
    (
        "table_page219",
        219,
        0,
        """-"""
    ),
    # teste 103
    (
        "table_page220",
        220,
        0,
        """-"""
    ),
    # teste 104
    (
        "table_page221",
        221,
        0,
        """-"""
    ),
    # teste 105
    (
        "table_page222",
        222,
        0,
        """-"""
    ),
    # teste 106
    (
        "table_page225",
        225,
        0,
        """-"""
    ),
    # teste 107
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
     # teste 108
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
     # teste 109
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
    # teste 110
    (
        "table_page227",
        227,
        0,
        """-"""
    ),
    # teste 111
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
    # teste 112
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
    # teste 113
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
    
# teste 114
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
######Parei aqui
    # teste 115
    (
        "table_page231",
        231,
        0,
        """-"""
    ),
    # teste 116
    (
        "table_page232",
        232,
        0,
        """-"""
    ),
    # teste 117
    (
        "table_page235",
        235,
        0,
        """-"""
    ),
    # teste 118
    (
        "table_page236",
        236,
        0,
        """-"""
    ),
    # teste 119
    (
        "table_page237",
        237,
        0,
        """-"""
    ),
    # teste 120
    (
        "table_page238",
        238,
        0,
        """-"""
    ),
    # teste 121
    (
        "table_page238",
        238,
        1,
        """-"""
    ),
    # teste 122
    (
        "table_page239",
        239,
        0,
        """-"""
    ),
    # teste 123
    (
        "table_page240",
        240,
        0,
        """-"""
    ),
    # teste 124
    (
        "table_page241",
        241,
        0,
        """-"""
    ),
    # teste 125
    (
        "table_page242",
        242,
        0,
        """-"""
    ),
    # teste 126
    (
        "table_page243",
        243,
        0,
        """-"""
    ),
    # teste 127
    (
        "table_page244",
        244,
        0,
        """-"""
    ),
    # teste 128
    (
        "table_page245",
        245,
        0,
        """-"""
    ),
    # teste 129
    (
        "table_page246",
        246,
        0,
        """-"""
    ),
    # teste 130
    (
        "table_page247",
        247,
        0,
        """-"""
    ),
    # teste 131
    (
        "table_page248",
        248,
        0,
        """-"""
    ),
    # teste 132
    (
        "table_page249",
        249,
        0,
        """-"""
    ),
    # teste 133
    (
        "table_page251",
        251,
        0,
        """-"""
    ),
    # teste 134
    (
        "table_page252",
        252,
        0,
        """-"""
    ),
    # teste 135
    (
        "table_page253",
        253,
        0,
        """-"""
    ),
    # teste 136
    (
        "table_page258",
        258,
        0,
        """-"""
    ),
    # teste 137
    (
        "table_page259",
        259,
        0,
        """-"""
    ),
    # teste 138
    (
        "table_page261",
        261,
        0,
        """-"""
    ),
    # teste 139
    (
        "table_page262",
        262,
        0,
        """-"""
    ),
    # teste 140
    (
        "table_page263",
        263,
        0,
        """-"""
    ),
    # teste 141
    (
        "table_page263",
        263,
        1,
        """-"""
    ),
    # teste 142
    (
        "table_page264",
        264,
        0,
        """-"""
    ),
    # teste 143
    (
        "table_page266",
        266,
        0,
        """-"""
    ),
    # teste 144
    (
        "table_page267",
        267,
        0,
        """-"""
    ),
    # teste 145
    (
        "table_page268",
        268,
        0,
        """-"""
    ),
    # teste 146
    (
        "table_page269",
        269,
        0,
        """-"""
    ),
    # teste 147
    (
        "table_page271",
        271,
        0,
        """-"""
    ),
    # teste 148
    (
        "table_page272",
        272,
        0,
        """-"""
    ),
    # teste 149
    (
        "table_page273",
        273,
        0,
        """-"""
    ),
    # teste 150
    (
        "table_page275",
        275,
        0,
        """-"""
    ),
    # teste 151
    (
        "table_page277",
        277,
        0,
        """-"""
    ),
    # teste 152
    (
        "table_page278",
        278,
        0,
        """-"""
    ),
    # teste 153
    (
        "table_page278",
        278,
        1,
        """-"""
    ),
    # teste 154
    (
        "table_page279",
        279,
        0,
        """-"""
    ),
    # teste 155
    (
        "table_page280",
        280,
        0,
        """-"""
    ),
    # teste 156
    (
        "table_page280",
        280,
        1,
        """-"""
    ),
    # teste 157
    (
        "table_page281",
        281,
        0,
        """-"""
    ),
    # teste 158
    (
        "table_page282",
        282,
        0,
        """-"""
    ),
    # teste 159
    (
        "table_page283",
        283,
        0,
        """-"""
    ),
    # teste 160
    (
        "table_page284",
        284,
        0,
        """-"""
    ),
    # teste 161
    (
        "table_page285",
        285,
        0,
        """-"""
    ),
    # teste 162
    (
        "table_page286",
        286,
        0,
        """-"""
    ),
    # teste 163
    (
        "table_page287",
        287,
        0,
        """-"""
    ),
    # teste 164
    (
        "table_page288",
        288,
        0,
        """-"""
    ),
    # teste 165
    (
        "table_page289",
        289,
        0,
        """-"""
    ),
    # teste 166
    (
        "table_page293",
        293,
        0,
        """-"""
    ),
    # teste 167
    (
        "table_page294",
        294,
        0,
        """-"""
    ),
    # teste 168
    (
        "table_page297",
        297,
        0,
        """-"""
    ),
    # teste 169
    (
        "table_page299",
        299,
        0,
        """-"""
    ),
    # teste 170
    (
        "table_page300",
        300,
        0,
        """-"""
    ),
    # teste 171
    (
        "table_page302",
        302,
        0,
        """-"""
    ),
    # teste 172
    (
        "table_page303",
        303,
        0,
        """-"""
    ),
    # teste 173
    (
        "table_page304",
        304,
        0,
        """-"""
    ),
    # teste 174
    (
        "table_page305",
        305,
        0,
        """-"""
    ),
    # teste 175
    (
        "table_page306",
        306,
        0,
        """-"""
    ),
    # teste 176
    (
        "table_page307",
        307,
        0,
        """-"""
    ),
    # teste 177
    (
        "table_page308",
        308,
        0,
        """-"""
    ),
    # teste 178
    (
        "table_page309",
        309,
        0,
        """-"""
    ),
    # teste 179
    (
        "table_page309",
        309,
        1,
        """-"""
    ),
    # teste 180
    (
        "table_page310",
        310,
        0,
        """-"""
    ),
    # teste 181
    (
        "table_page311",
        311,
        0,
        """-"""
    ),
    # teste 182
    (
        "table_page312",
        312,
        0,
        """-"""
    ),
    # teste 183
    (
        "table_page313",
        313,
        0,
        """-"""
    ),
    # teste 184
    (
        "table_page313",
        313,
        1,
        """-"""
    ),
    # teste 185
    (
        "table_page314",
        314,
        0,
        """-"""
    ),
    # teste 186
    (
        "table_page314",
        314,
        1,
        """-"""
    ),
    # teste 187
    (
        "table_page315",
        315,
        0,
        """-"""
    ),
    # teste 188
    (
        "table_page316",
        316,
        0,
        """-"""
    ),
    # teste 189
    (
        "table_page317",
        317,
        0,
        """-"""
    ),
    # teste 190
    (
        "table_page318",
        318,
        0,
        """-"""
    ),
    # teste 191
    (
        "table_page319",
        319,
        0,
        """-"""
    ),
    # teste 192
    (
        "table_page320",
        320,
        0,
        """-"""
    ),
    # teste 193
    (
        "table_page323",
        323,
        0,
        """-"""
    ),
    # teste 194
    (
        "table_page324",
        324,
        0,
        """-"""
    ),
    # teste 195
    (
        "table_page328",
        328,
        0,
        """-"""
    ),
    # teste 196
    (
        "table_page329",
        329,
        0,
        """-"""
    ),
    # teste 197
    (
        "table_page331",
        331,
        0,
        """-"""
    ),
    # teste 198
    (
        "table_page332",
        332,
        0,
        """-"""
    ),
    # teste 199
    (
        "table_page333",
        333,
        0,
        """-"""
    ),
    # teste 200
    (
        "table_page334",
        334,
        0,
        """-"""
    ),
    # teste 201
    (
        "table_page335",
        335,
        0,
        """-"""
    ),
    # teste 202
    (
        "table_page335",
        335,
        1,
        """-"""
    ),
    # teste 203
    (
        "table_page336",
        336,
        0,
        """-"""
    ),
    # teste 204
    (
        "table_page337",
        337,
        0,
        """-"""
    ),
    # teste 205
    (
        "table_page339",
        339,
        0,
        """-"""
    ),
    # teste 206
    (
        "table_page340",
        340,
        0,
        """-"""
    ),
    # teste 207
    (
        "table_page341",
        341,
        0,
        """-"""
    ),
    # teste 208
    (
        "table_page342",
        342,
        0,
        """-"""
    ),
    # teste 209
    (
        "table_page343",
        343,
        0,
        """-"""
    ),
    # teste 210
    (
        "table_page344",
        344,
        0,
        """-"""
    ),
    # teste 211
    (
        "table_page345",
        345,
        0,
        """-"""
    ),
    # teste 212
    (
        "table_page346",
        346,
        0,
        """-"""
    ),
    # teste 213
    (
        "table_page348",
        348,
        0,
        """-"""
    ),
    # teste 214
    (
        "table_page349",
        349,
        0,
        """-"""
    ),
    # teste 215
    (
        "table_page350",
        350,
        0,
        """-"""
    ),
    # teste 216
    (
        "table_page351",
        351,
        0,
        """-"""
    ),
    # teste 217
    (
        "table_page352",
        352,
        0,
        """-"""
    ),
    # teste 218
    (
        "table_page354",
        354,
        0,
        """-"""
    ),
    # teste 219
    (
        "table_page354",
        354,
        1,
        """-"""
    ),
    # teste 220
    (
        "table_page355",
        355,
        0,
        """-"""
    ),
    # teste 221
    (
        "table_page357",
        357,
        0,
        """-"""
    ),
    # teste 222
    (
        "table_page358",
        358,
        0,
        """-"""
    ),
    # teste 223
    (
        "table_page359",
        359,
        0,
        """-"""
    ),
    # teste 224
    (
        "table_page359",
        359,
        1,
        """-"""
    ),
    # teste 225
    (
        "table_page360",
        360,
        0,
        """-"""
    ),
    # teste 226
    (
        "table_page360",
        360,
        1,
        """-"""
    ),
    # teste 227
    (
        "table_page361",
        361,
        0,
        """-"""
    ),
    # teste 228
    (
        "table_page362",
        362,
        0,
        """-"""
    ),
    # teste 229
    (
        "table_page363",
        363,
        0,
        """-"""
    ),
    # teste 230
    (
        "table_page363",
        363,
        1,
        """-"""
    ),
    # teste 231
    (
        "table_page364",
        364,
        0,
        """-"""
    ),
    # teste 232
    (
        "table_page365",
        365,
        0,
        """-"""
    ),
    # teste 233
    (
        "table_page366",
        366,
        0,
        """-"""
    ),
    # teste 234
    (
        "table_page367",
        367,
        0,
        """-"""
    ),
    # teste 235
    (
        "table_page368",
        368,
        0,
        """-"""
    ),
    # teste 236
    (
        "table_page369",
        369,
        0,
        """-"""
    ),
    # teste 237
    (
        "table_page370",
        370,
        0,
        """-"""
    ),
    # teste 238
    (
        "table_page373",
        373,
        0,
        """-"""
    ),
    # teste 239
    (
        "table_page374",
        374,
        0,
        """-"""
    ),
    # teste 240
    (
        "table_page375",
        375,
        0,
        """-"""
    ),
    # teste 241
    (
        "table_page376",
        376,
        0,
        """-"""
    ),
    # teste 242
    (
        "table_page377",
        377,
        0,
        """-"""
    ),
    # teste 243
    (
        "table_page378",
        378,
        0,
        """-"""
    ),
    # teste 244
    (
        "table_page379",
        379,
        0,
        """-"""
    ),
    # teste 245
    (
        "table_page380",
        380,
        0,
        """-"""
    ),
    # teste 246
    (
        "table_page381",
        381,
        0,
        """-"""
    ),
    # teste 247
    (
        "table_page381",
        381,
        1,
        """-"""
    ),
    # teste 248
    (
        "table_page382",
        382,
        0,
        """-"""
    ),
    # teste 249
    (
        "table_page383",
        383,
        0,
        """-"""
    ),
    # teste 250
    (
        "table_page384",
        384,
        0,
        """-"""
    ),
    # teste 251
    (
        "table_page385",
        385,
        0,
        """-"""
    ),
    # teste 252
    (
        "table_page386",
        386,
        0,
        """-"""
    ),
    # teste 253
    (
        "table_page387",
        387,
        0,
        """-"""
    ),
    # teste 254
    (
        "table_page389",
        389,
        0,
        """-"""
    ),
    # teste 255
    (
        "table_page390",
        390,
        0,
        """-"""
    ),
    # teste 256
    (
        "table_page391",
        391,
        0,
        """-"""
    ),
    # teste 257
    (
        "table_page392",
        392,
        0,
        """-"""
    ),
    # teste 258
    (
        "table_page393",
        393,
        0,
        """-"""
    ),
    # teste 259
    (
        "table_page394",
        394,
        0,
        """-"""
    ),
    # teste 260
    (
        "table_page394",
        394,
        1,
        """-"""
    ),
    # teste 261
    (
        "table_page395",
        395,
        0,
        """-"""
    ),
    # teste 262
    (
        "table_page396",
        396,
        0,
        """-"""
    ),
    # teste 263
    (
        "table_page396",
        396,
        1,
        """-"""
    ),
    # teste 264
    (
        "table_page397",
        397,
        0,
        """-"""
    ),
    # teste 265
    (
        "table_page398",
        398,
        0,
        """-"""
    ),
    # teste 266
    (
        "table_page399",
        399,
        0,
        """-"""
    ),
    # teste 267
    (
        "table_page399",
        399,
        1,
        """-"""
    ),
    # teste 268
    (
        "table_page399",
        399,
        2,
        """-"""
    ),
    # teste 269
    (
        "table_page400",
        400,
        0,
        """-"""
    ),
    # teste 270
    (
        "table_page401",
        401,
        0,
        """-"""
    ),
    # teste 271
    (
        "table_page402",
        402,
        0,
        """-"""
    ),
    # teste 272
    (
        "table_page403",
        403,
        0,
        """-"""
    ),
    # teste 273
    (
        "table_page405",
        405,
        0,
        """-"""
    ),
    # teste 274
    (
        "table_page406",
        406,
        0,
        """-"""
    ),
    # teste 275
    (
        "table_page408",
        408,
        0,
        """-"""
    ),
    # teste 276
    (
        "table_page409",
        409,
        0,
        """-"""
    ),
    # teste 277
    (
        "table_page410",
        410,
        0,
        """-"""
    ),
    # teste 278
    (
        "table_page411",
        411,
        0,
        """-"""
    ),
    # teste 279
    (
        "table_page412",
        412,
        0,
        """-"""
    ),
    # teste 280
    (
        "table_page413",
        413,
        0,
        """-"""
    ),
    # teste 281
    (
        "table_page413",
        413,
        1,
        """-"""
    ),
    # teste 282
    (
        "table_page414",
        414,
        0,
        """-"""
    ),
    # teste 283
    (
        "table_page415",
        415,
        0,
        """-"""
    ),
    # teste 284
    (
        "table_page415",
        415,
        1,
        """-"""
    ),
    # teste 285
    (
        "table_page416",
        416,
        0,
        """-"""
    ),
    # teste 286
    (
        "table_page417",
        417,
        0,
        """-"""
    ),
    # teste 287
    (
        "table_page417",
        417,
        1,
        """-"""
    ),
    # teste 288
    (
        "table_page418",
        418,
        0,
        """-"""
    ),
    # teste 289
    (
        "table_page418",
        418,
        1,
        """-"""
    ),
    # teste 290
    (
        "table_page419",
        419,
        0,
        """-"""
    ),
    # teste 291
    (
        "table_page420",
        420,
        0,
        """-"""
    ),
    # teste 292
    (
        "table_page421",
        421,
        0,
        """-"""
    ),
    # teste 293
    (
        "table_page422",
        422,
        0,
        """-"""
    ),
    # teste 294
    (
        "table_page423",
        423,
        0,
        """-"""
    ),
    # teste 295
    (
        "table_page425",
        425,
        0,
        """-"""
    ),
    # teste 296
    (
        "table_page426",
        426,
        0,
        """-"""
    ),
    # teste 297
    (
        "table_page427",
        427,
        0,
        """-"""
    ),
    # teste 298
    (
        "table_page433",
        433,
        0,
        """-"""
    ),
    # teste 299
    (
        "table_page434",
        434,
        0,
        """-"""
    ),
    # teste 300
    (
        "table_page436",
        436,
        0,
        """-"""
    ),
    # teste 301
    (
        "table_page436",
        436,
        1,
        """-"""
    ),
    # teste 302
    (
        "table_page437",
        437,
        0,
        """-"""
    ),
    # teste 303
    (
        "table_page437",
        437,
        1,
        """-"""
    ),
    # teste 304
    (
        "table_page438",
        438,
        0,
        """-"""
    ),
    # teste 305
    (
        "table_page440",
        440,
        0,
        """-"""
    ),
    # teste 306
    (
        "table_page440",
        440,
        1,
        """-"""
    ),
    # teste 307
    (
        "table_page441",
        441,
        0,
        """-"""
    ),
    # teste 308
    (
        "table_page442",
        442,
        0,
        """-"""
    ),
    # teste 309
    (
        "table_page445",
        445,
        0,
        """-"""
    ),
    # teste 310
    (
        "table_page446",
        446,
        0,
        """-"""
    ),
    # teste 311
    (
        "table_page447",
        447,
        0,
        """-"""
    ),
    # teste 312
    (
        "table_page450",
        450,
        0,
        """-"""
    ),
    # teste 313
    (
        "table_page451",
        451,
        0,
        """-"""
    ),
    # teste 314
    (
        "table_page451",
        451,
        1,
        """-"""
    ),
    # teste 315
    (
        "table_page452",
        452,
        0,
        """-"""
    ),
    # teste 316
    (
        "table_page452",
        452,
        1,
        """-"""
    ),
    # teste 317
    (
        "table_page456",
        456,
        0,
        """-"""
    ),
    # teste 318
    (
        "table_page456",
        456,
        1,
        """-"""
    ),
    # teste 319
    (
        "table_page457",
        457,
        0,
        """-"""
    ),
    # teste 320
    (
        "table_page458",
        458,
        0,
        """-"""
    ),
    # teste 321
    (
        "table_page460",
        460,
        0,
        """-"""
    ),
    # teste 322
    (
        "table_page461",
        461,
        0,
        """-"""
    ),
    # teste 323
    (
        "table_page461",
        461,
        1,
        """-"""
    ),
    # teste 324
    (
        "table_page464",
        464,
        0,
        """-"""
    ),
    # teste 325
    (
        "table_page464",
        464,
        1,
        """-"""
    ),
    # teste 326
    (
        "table_page465",
        465,
        0,
        """-"""
    ),
    # teste 327
    (
        "table_page466",
        466,
        0,
        """-"""
    ),
    # teste 328
    (
        "table_page469",
        469,
        0,
        """-"""
    ),
    # teste 329
    (
        "table_page469",
        469,
        1,
        """-"""
    ),
    # teste 330
    (
        "table_page470",
        470,
        0,
        """-"""
    ),
    # teste 331
    (
        "table_page471",
        471,
        0,
        """-"""
    ),
    # teste 332
    (
        "table_page474",
        474,
        0,
        """-"""
    ),
    # teste 333
    (
        "table_page474",
        474,
        1,
        """-"""
    ),
    # teste 334
    (
        "table_page475",
        475,
        0,
        """-"""
    ),
    # teste 335
    (
        "table_page475",
        475,
        1,
        """-"""
    ),
    # teste 336
    (
        "table_page476",
        476,
        0,
        """-"""
    ),
    # teste 337
    (
        "table_page480",
        480,
        0,
        """-"""
    ),
    # teste 338
    (
        "table_page480",
        480,
        1,
        """-"""
    ),
    # teste 339
    (
        "table_page481",
        481,
        0,
        """-"""
    ),
    # teste 340
    (
        "table_page481",
        481,
        1,
        """-"""
    ),
    # teste 341
    (
        "table_page482",
        482,
        0,
        """-"""
    ),
    # teste 342
    (
        "table_page484",
        484,
        0,
        """-"""
    ),
    # teste 343
    (
        "table_page484",
        484,
        1,
        """-"""
    ),
    # teste 344
    (
        "table_page486",
        486,
        0,
        """-"""
    ),
    # teste 345
    (
        "table_page487",
        487,
        0,
        """-"""
    ),
    # teste 346
    (
        "table_page487",
        487,
        1,
        """-"""
    ),
    # teste 347
    (
        "table_page489",
        489,
        0,
        """-"""
    ),
    # teste 348
    (
        "table_page489",
        489,
        1,
        """-"""
    ),
    # teste 349
    (
        "table_page490",
        490,
        0,
        """-"""
    ),
    # teste 350
    (
        "table_page492",
        492,
        0,
        """-"""
    ),
    # teste 351
    (
        "table_page492",
        492,
        1,
        """-"""
    ),
    # teste 352
    (
        "table_page493",
        493,
        0,
        """-"""
    ),
    # teste 353
    (
        "table_page493",
        493,
        1,
        """-"""
    ),
    # teste 354
    (
        "table_page496",
        496,
        0,
        """-"""
    ),
    # teste 355
    (
        "table_page497",
        497,
        0,
        """-"""
    ),
    # teste 356
    (
        "table_page497",
        497,
        1,
        """-"""
    ),
    # teste 357
    (
        "table_page498",
        498,
        0,
        """-"""
    ),
    # teste 358
    (
        "table_page498",
        498,
        1,
        """-"""
    ),
    # teste 359
    (
        "table_page500",
        500,
        0,
        """-"""
    ),
    # teste 360
    (
        "table_page501",
        501,
        0,
        """-"""
    ),
    # teste 361
    (
        "table_page502",
        502,
        0,
        """-"""
    ),
    # teste 362
    (
        "table_page507",
        507,
        0,
        """-"""
    ),
    # teste 363
    (
        "table_page507",
        507,
        1,
        """-"""
    ),
    # teste 364
    (
        "table_page512",
        512,
        0,
        """-"""
    ),
    # teste 365
    (
        "table_page513",
        513,
        0,
        """-"""
    ),
    # teste 366
    (
        "table_page516",
        516,
        0,
        """-"""
    ),
    # teste 367
    (
        "table_page517",
        517,
        0,
        """-"""
    ),
    # teste 368
    (
        "table_page519",
        519,
        0,
        """-"""
    ),
    # teste 369
    (
        "table_page520",
        520,
        0,
        """-"""
    ),
    # teste 370
    (
        "table_page521",
        521,
        0,
        """-"""
    ),
    # teste 371
    (
        "table_page522",
        522,
        0,
        """-"""
    ),
    # teste 372
    (
        "table_page523",
        523,
        0,
        """-"""
    ),
    # teste 373
    (
        "table_page524",
        524,
        0,
        """-"""
    ),
    # teste 374
    (
        "table_page525",
        525,
        0,
        """-"""
    ),
    # teste 375
    (
        "table_page526",
        526,
        0,
        """-"""
    ),
    # teste 376
    (
        "table_page527",
        527,
        0,
        """-"""
    ),
    # teste 377
    (
        "table_page528",
        528,
        0,
        """-"""
    ),
    # teste 378
    (
        "table_page529",
        529,
        0,
        """-"""
    ),
    # teste 379
    (
        "table_page530",
        530,
        0,
        """-"""
    ),
    # teste 380
    (
        "table_page531",
        531,
        0,
        """-"""
    ),
    # teste 381
    (
        "table_page532",
        532,
        0,
        """-"""
    ),
    # teste 382
    (
        "table_page533",
        533,
        0,
        """-"""
    ),
    # teste 383
    (
        "table_page534",
        534,
        0,
        """-"""
    ),
    # teste 384
    (
        "table_page535",
        535,
        0,
        """-"""
    ),
    # teste 385
    (
        "table_page536",
        536,
        0,
        """-"""
    ),

# teste 386
(
        "table_page537",
        537,
        0,
        """------------------------------------------------------------------------------
|Batch No.         |240101         |Batch size             |10 kg            |
|------------------|---------------|-----------------------|-----------------|
|Manufacturing date|Jan. 06, 2024  |Stability starting date|Feb. 26, 2024    |
|------------------|---------------|-----------------------|-----------------|
|Degradation       |High temperature 60℃: place appropriate amount of        |
|condition         |substance without package intoa container evenly, the    |
|                  |thickness of substance is 3 ~ 5 mm. Store the culture    |
|                  |dish into constant temperature oven at 60℃.              |
|------------------|---------------------------------------------------------|
|Items     |Specifications |Time points                                      |
|          |               |-------------------------------------------------|
|          |               |0d             |5d     |15d            |30d      |
|----------|---------------|---------------|-------|---------------|---------|
|Appearance|White to yellow|White powder   |Off    |Yellowish      |Yellowish|
|          |powder.        |               |White  |powder         |powder   |
|          |               |               |powder |               |         |
|----------|---------------|---------------|-------|---------------|---------|
|Related   |YA2304-10 NMT  |ND             |ND     |ND             |ND       |
|substances|0.10%；        |               |       |               |         |
|          |---------------|---------------|-------|---------------|---------|
|          |YA2304-12 NMT  |ND             |＜0.05%|＜0.05%        |＜0.05%  |
|          |0.15%；        |               |       |               |         |
|          |---------------|---------------|-------|---------------|---------|
|          |YA2304-14 NMT  |ND             |＜0.05%|＜0.05%        |＜0.05%  |
|          |0.15%；        |               |       |               |         |
|          |---------------|---------------|-------|---------------|---------|
|          |YA2304-15 NMT  |ND             |ND     |ND             |ND       |
|          |0.15%；        |               |       |               |         |
|          |---------------|---------------|-------|---------------|---------|
|          |YA2304-16 NMT  |ND             |ND     |ND             |0.05%    |
|          |0.15%；        |               |       |               |         |
|          |---------------|---------------|-------|---------------|---------|
|          |YA2304-17 NMT  |ND             |ND     |ND             |ND       |
|          |0.15%；        |               |       |               |         |
|          |---------------|---------------|-------|---------------|---------|
|          |YA2304-18 NMT  |ND             |ND     |ND             |＜0.05%  |
|          |0.15%；        |               |       |               |         |
|          |---------------|---------------|-------|---------------|---------|
|          |YA2304-19 NMT  |＜0.05%        |＜0.05%|0.07%          |0.11%    |
|          |0.15%；        |               |       |               |         |
|          |---------------|---------------|-------|---------------|---------|
|          |Other single   |＜0.05%        |＜0.05%|＜0.05%        |0.06%    |
|          |impurities: NMT|               |       |               |         |
|          |0.10%          |               |       |               |         |
------------------------------------------------------------------------------"""),
# teste 387
(
        "table_page538",
        538,
        0,
        """-----------------------------------------------------------------------------
|Batch No.          |240101         |Batch size             |10 kg          |
|-------------------|---------------|-----------------------|---------------|
|Manufacturing date |Jan. 06, 2024  |Stability starting date|Feb. 26, 2024  |
|-------------------|---------------|-----------------------|---------------|
|Degradation        |High temperature 60℃: place appropriate amount of      |
|condition          |substance without package intoa container evenly, the  |
|                   |thickness of substance is 3 ~ 5 mm. Store the culture  |
|                   |dish into constant temperature oven at 60℃.            |
|-------------------|-------------------------------------------------------|
|Items      |Specifications |Time points                                    |
|           |               |-----------------------------------------------|
|           |               |0d             |5d     |15d            |30d    |
|-----------|---------------|---------------|-------|---------------|-------|
|           |Total impurity:|＜0.05%        |＜0.05%|0.11%          |0.22%  |
|           |NMT 1.0%       |               |       |               |       |
|-----------|---------------|---------------|-------|---------------|-------|
|Enantiomers|YA2304-20: NMT |0.03%          |ND     |ND             |ND     |
|           |0.15%          |               |       |               |       |
|-----------|---------------|---------------|-------|---------------|-------|
|Water      |NMT 0.5%       |0.05%          |0.03%  |0.05%          |0.06%  |
|-----------|---------------|---------------|-------|---------------|-------|
|Assay      |98.0% to 102.0%|99.8%          |100.2% |99.9%          |99.9%  |
|           |(anhydrous     |               |       |               |       |
|           |substance).    |               |       |               |       |
-----------------------------------------------------------------------------"""),
# teste 388
(
        "table_page539",
        539,
        0,
        """----------------------------------------------------------------------------
|Batch No.         |231201         |Batch size             |10 kg          |
|------------------|---------------|-----------------------|---------------|
|Manufacturing date|Jan. 06, 2024  |Stability starting date|Feb. 24, 2024  |
|------------------|---------------|-----------------------|---------------|
|Degradation       |High humidity 92.5%RH: place appropriate amount of     |
|condition         |substance without package intoa container evenly, the  |
|                  |thickness of substance is 3 ~ 5 mm. Store the culture  |
|                  |dish at condition 92.5%RH, 25℃.                        |
|------------------|-------------------------------------------------------|
|Items     |Specifications |Time points                                    |
|          |               |-----------------------------------------------|
|          |               |0d             |5d     |15d            |30d    |
|----------|---------------|---------------|-------|---------------|-------|
|Appearance|White to yellow|White powder   |White  |White powder   |White  |
|          |powder.        |               |powder |               |powder |
|----------|---------------|---------------|-------|---------------|-------|
|Related   |YA2304-10 NMT  |ND             |ND     |ND             |ND     |
|substances|0.10%；        |               |       |               |       |
|          |---------------|---------------|-------|---------------|-------|
|          |YA2304-12 NMT  |ND             |＜0.05%|＜0.05%        |＜0.05%|
|          |0.15%；        |               |       |               |       |
|          |---------------|---------------|-------|---------------|-------|
|          |YA2304-14 NMT  |ND             |ND     |＜0.05%        |ND     |
|          |0.15%；        |               |       |               |       |
|          |---------------|---------------|-------|---------------|-------|
|          |YA2304-15 NMT  |ND             |＜0.05%|＜0.05%        |＜0.05%|
|          |0.15%；        |               |       |               |       |
|          |---------------|---------------|-------|---------------|-------|
|          |YA2304-16 NMT  |ND             |ND     |ND             |ND     |
|          |0.15%；        |               |       |               |       |
|          |---------------|---------------|-------|---------------|-------|
|          |YA2304-17 NMT  |ND             |ND     |ND             |ND     |
|          |0.15%；        |               |       |               |       |
|          |---------------|---------------|-------|---------------|-------|
|          |YA2304-18 NMT  |ND             |ND     |ND             |ND     |
|          |0.15%；        |               |       |               |       |
|          |---------------|---------------|-------|---------------|-------|
|          |YA2304-19 NMT  |＜0.05%        |＜0.05%|＜0.05%        |＜0.05%|
|          |0.15%；        |               |       |               |       |
|          |---------------|---------------|-------|---------------|-------|
|          |Other single   |＜0.05%        |＜0.05%|＜0.05%        |＜0.05%|
|          |impurities: NMT|               |       |               |       |
|          |0.10%          |               |       |               |       |
|          |---------------|---------------|-------|---------------|-------|
|          |Total impurity:|＜0.05%        |＜0.05%|＜0.05%        |＜0.05%|
|          |NMT 1.0%       |               |       |               |       |
----------------------------------------------------------------------------"""),
# teste 389
(
        "table_page540",
        540,
        0,
        """-----------------------------------------------------------------------------
|Batch No.          |231201         |Batch size             |10 kg          |
|-------------------|---------------|-----------------------|---------------|
|Manufacturing date |Jan. 06, 2024  |Stability starting date|Feb. 24, 2024  |
|-------------------|---------------|-----------------------|---------------|
|Degradation        |High humidity 92.5%RH: place appropriate amount of     |
|condition          |substance without package intoa container evenly, the  |
|                   |thickness of substance is 3 ~ 5 mm. Store the culture  |
|                   |dish at condition 92.5%RH, 25℃.                        |
|-------------------|-------------------------------------------------------|
|Items      |Specifications |Time points                                    |
|           |               |-----------------------------------------------|
|           |               |0d             |5d     |15d            |30d    |
|-----------|---------------|---------------|-------|---------------|-------|
|Enantiomers|YA2304-20: NMT |0.03%          |0.02%  |0.02%          |0.02%  |
|           |0.15%          |               |       |               |       |
|-----------|---------------|---------------|-------|---------------|-------|
|Water      |NMT 0.5%       |0.05%          |0.03%  |0.03%          |0.03%  |
|-----------|---------------|---------------|-------|---------------|-------|
|Assay      |98.0% to 102.0%|99.8%          |100.0% |100.3%         |100.0% |
|           |(anhydrous     |               |       |               |       |
|           |substance).    |               |       |               |       |
|-----------|---------------|---------------|-------|---------------|-------|
|Hygroscopic|NMT 5%         |/              |0.04%  |0.02%          |0.05%  |
|gain (%)   |               |               |       |               |       |
-----------------------------------------------------------------------------"""),
# teste 390
(
        "table_page541",
        541,
        0,
        """-"""
    ),
    # teste 391
    (
        "table_page542",
        542,
        0,
        """-"""
    ),
]


@pytest.mark.parametrize(
    "test_id,page,table_index,expected_ascii_matrix",
    TEST_CONFIGURATIONS,
    ids=[config[0] for config in TEST_CONFIGURATIONS]
)
def test_ascii_matrix_comparison(test_id, page, table_index, expected_ascii_matrix):
    run_table_test(PDF_ENV_VAR, test_id, page, table_index, expected_ascii_matrix)

