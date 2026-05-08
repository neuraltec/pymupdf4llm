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
        "table_page10",
        10,
        0,
        """-------------------------------------------------------------------------
|Sl no. |Name of the impurity             |Structure|Origin  |Control   |
|-------|---------------------------------|---------|--------|----------|
|1      |7-HDQ                            |         |Key     |Existing  |
|       |[7-Hydroxy-3,4-dihydroquinolin-  |         |starting|HPLC      |
|       |2(1H)-one]                       |         |material|method,   |
|       |                                 |         |        |employed  |
|       |                                 |         |        |to detect |
|       |                                 |         |        |related   |
|       |                                 |         |        |substances|
|       |                                 |         |        |in the    |
|       |                                 |         |        |drug      |
|       |                                 |         |        |substance |
|       |                                 |         |        |is capable|
|       |                                 |         |        |of        |
|       |                                 |         |        |resolving |
|       |                                 |         |        |7-HDQ, if |
|       |                                 |         |        |present.  |
|-------|---------------------------------|---------|--------|----------|
|2      |DCPP                             |         |Key     |Existing  |
|       |[1-(2,3-Dichlorophenyl)piperazine|         |starting|HPLC      |
|       |hydrochloride]                   |         |material|method,   |
|       |                                 |         |        |employed  |
|       |                                 |         |        |to detect |
|       |                                 |         |        |related   |
|       |                                 |         |        |substances|
|       |                                 |         |        |in the    |
|       |                                 |         |        |drug      |
|       |                                 |         |        |substance |
|       |                                 |         |        |is capable|
|       |                                 |         |        |of        |
|       |                                 |         |        |resolving |
|       |                                 |         |        |DCPP, if  |
|       |                                 |         |        |present.  |
-------------------------------------------------------------------------"""
    ),
    (
        "table_page11",
        11,
        0,
        """--------------------------------------------------------------------------------
|3      |2-Chloro APR                           |       |From KSM|Existing HPLC|
|       |[7-[4-[4-(2-chlorophenyl)piperazin-1-  |       |DCPP    |method,      |
|       |yl]butoxy]-3,4-dihydroquinolin-2(1H)-  |       |        |employed to  |
|       |one]                                   |       |        |detect       |
|       |                                       |       |        |related      |
|       |                                       |       |        |substances in|
|       |                                       |       |        |the drug     |
|       |                                       |       |        |substance is |
|       |                                       |       |        |capable of   |
|       |                                       |       |        |resolving    |
|       |                                       |       |        |2-Chloro APR,|
|       |                                       |       |        |if present.  |
|-------|---------------------------------------|-------|--------|-------------|
|4      |3-Chloro APR                           |       |From KSM|Existing HPLC|
|       |[7-[4-[4-(3-chlorophenyl)piperazin-1-  |       |DCPP    |method,      |
|       |yl]butoxy]-3,4-dihydroquinolin-2(1H)-  |       |        |employed to  |
|       |one]                                   |       |        |detect       |
|       |                                       |       |        |related      |
|       |                                       |       |        |substances in|
|       |                                       |       |        |the drug     |
|       |                                       |       |        |substance is |
|       |                                       |       |        |capable of   |
|       |                                       |       |        |resolving    |
|       |                                       |       |        |3-Chloro APR,|
|       |                                       |       |        |if present.  |
|-------|---------------------------------------|-------|--------|-------------|
|5      |Aripiprazole Related compound- G       |       |Process |Controlled in|
|       |[Dehydro Aripiprazole]                 |       |related |drug         |
|       |[7-[4-[4-(2,3-Dichlorophenyl)piperazin-|       |impurity|substance    |
|       |1-yl]butoxy)quinolin-2 (1H)-one]       |       |        |specification|
|       |                                       |       |        |with the     |
|       |                                       |       |        |limit of ‘Not|
|       |                                       |       |        |more than    |
|       |                                       |       |        |0.10%’       |
--------------------------------------------------------------------------------"""
    ),
    (
        "table_page12",
        12,
        0,
        """---------------------------------------------------------------------------------
|6      |Aripiprazole Related compound- F       |       |Degradant|Controlled in|
|       |[N-oxide]                              |       |         |drug         |
|       |4-(2,3-Dichlorophenyl)-1-[4-(2-oxo-    |       |         |substance    |
|       |1,2,3,4-tetrahydroquinolin-7-ylox-     |       |         |specification|
|       |y)butyl]piperazine 1-oxide             |       |         |with the     |
|       |                                       |       |         |limit of ‘Not|
|       |                                       |       |         |more than    |
|       |                                       |       |         |0.10%’       |
|-------|---------------------------------------|-------|---------|-------------|
|7      |Aripiprazole 4,4-Dimer                 |       |Process  |Controlled in|
|       |1,1’-(Ethane-1,1-diyl)bis(2,3-dichloro-|       |related  |drug         |
|       |4-(4-[3,4-dihydroquinolin-2(1H)-one-7- |       |impurity |substance    |
|       |yloxybutyl]piperazin-1-yl}benzene)     |       |         |specification|
|       |                                       |       |         |with the     |
|       |                                       |       |         |limit of ‘Not|
|       |                                       |       |         |more than    |
|       |                                       |       |         |0.10%’       |
|-------|---------------------------------------|-------|---------|-------------|
|8      |Dimer 7-(4-Hydroxy-butoxy)-bis-3,4-    |       |Process  |Existing HPLC|
|       |dihydro-1H-quinolin-2-one              |       |related  |method,      |
|       |                                       |       |impurity |employed to  |
|       |                                       |       |         |detect       |
|       |                                       |       |         |related      |
|       |                                       |       |         |substances in|
|       |                                       |       |         |the drug     |
|       |                                       |       |         |substance is |
|       |                                       |       |         |capable of   |
|       |                                       |       |         |resolving    |
|       |                                       |       |         |Dimer, if    |
|       |                                       |       |         |present.     |
---------------------------------------------------------------------------------"""
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
        "table_page16",
        16,
        0,
        """-------------------------------------------------------------------------------------------------------
|Sl. no |Structure and         |Source |Code#  |Classification|QSAR      |Mutagenic/   |Control /     |
|       |Chemical/IUPAC name   |       |       |as per ICHM7  |Evaluation|Non-Mutagenic|Carryover     |
|       |                      |       |       |              |          |             |studies       |
|-------|----------------------|-------|-------|--------------|----------|-------------|--------------|
|1.     |1,4-Dibromobutane     |APR-1  |1,4-DBB|3             |CHARACTE  |RISATION     |Controlled in |
|       |                      |stage  |       |              |Derek:    |Mutagenic    |Final         |
|       |                      |       |       |              |Plausible |             |specification |
|       |                      |       |       |              |Sarah:    |             |as part of CSO|
|       |                      |       |       |              |Positive  |             |with TTC limit|
|       |                      |       |       |              |          |             |[NMT 50 PPM]  |
|       |                      |       |       |              |          |             |[i.e Customer |
|       |                      |       |       |              |          |             |specific      |
|       |                      |       |       |              |          |             |order] CSO#   |
|       |                      |       |       |              |          |             |1566 Refer    |
|       |                      |       |       |              |          |             |section       |
|       |                      |       |       |              |          |             |3.2.S.4.1 for |
|       |                      |       |       |              |          |             |CSO           |
|-------|----------------------|-------|-------|--------------|----------|-------------|--------------|
|2.     |7-(4-Bromobutoxy)-3,4-|APR    |APR-1  |              |Derek:    |Mutagenic    |Controlled in |
|       |dihydroquinolin-2(1H)-|       |       |              |Plausible |             |Final         |
|       |one                   |       |       |              |Sarah:    |             |specification |
|       |                      |       |       |              |Positive  |             |as part of CSO|
|       |                      |       |       |              |          |             |with TTC limit|
|       |                      |       |       |              |          |             |[NMT 50 PPM]  |
|       |                      |       |       |              |          |             |[i.e Customer |
|       |                      |       |       |              |          |             |specific      |
|       |                      |       |       |              |          |             |order] CSO#   |
|       |                      |       |       |              |          |             |1566 Refer    |
|       |                      |       |       |              |          |             |section       |
|       |                      |       |       |              |          |             |3.2.S.4.1 for |
|       |                      |       |       |              |          |             |CSO           |
|-------|----------------------|-------|-------|--------------|----------|-------------|--------------|
|3.     |3-Hydroxyaniline      |7-HDQ  |HAN    |3             |Derek:    |Mutagenic    |Based on the  |
|       |                      |       |       |              |Plausible |             |Justification |
|       |                      |       |       |              |Sarah:    |             |for fate and  |
|       |                      |       |       |              |Positive  |             |purge factor  |
|       |                      |       |       |              |          |             |and batch     |
|       |                      |       |       |              |          |             |analysis      |
|       |                      |       |       |              |          |             |provided in   |
|       |                      |       |       |              |          |             |subsequent    |
|       |                      |       |       |              |          |             |pages of this |
|       |                      |       |       |              |          |             |section, it   |
|       |                      |       |       |              |          |             |can be        |
|       |                      |       |       |              |          |             |concluded that|
|       |                      |       |       |              |          |             |the impurity  |
|       |                      |       |       |              |          |             |HAN is found  |
|       |                      |       |       |              |          |             |less than 30% |
|       |                      |       |       |              |          |             |of TTC based  |
|       |                      |       |       |              |          |             |limit.        |
|       |                      |       |       |              |          |             |Therefore, no |
|       |                      |       |       |              |          |             |test is       |
|       |                      |       |       |              |          |             |proposed for  |
|       |                      |       |       |              |          |             |this impurity |
|       |                      |       |       |              |          |             |in the drug   |
|       |                      |       |       |              |          |             |substance     |
|       |                      |       |       |              |          |             |specification.|
|-------|----------------------|-------|-------|--------------|----------|-------------|--------------|
|4.     |3-Methoxyaniline      |7-HDQ  |3-MA   |3             |Derek:    |Mutagenic    |Based on the  |
|       |                      |       |       |              |Plausible |             |Justification |
|       |                      |       |       |              |Sarah:    |             |for fate and  |
|       |                      |       |       |              |Positive  |             |purge factor  |
|       |                      |       |       |              |          |             |and batch     |
|       |                      |       |       |              |          |             |analysis      |
|       |                      |       |       |              |          |             |provided in   |
|       |                      |       |       |              |          |             |subsequent    |
|       |                      |       |       |              |          |             |pages of this |
|       |                      |       |       |              |          |             |section, it   |
|       |                      |       |       |              |          |             |can be        |
|       |                      |       |       |              |          |             |concluded that|
|       |                      |       |       |              |          |             |the impurity  |
|       |                      |       |       |              |          |             |3-MA is found |
|       |                      |       |       |              |          |             |less than 30% |
|       |                      |       |       |              |          |             |of TTC based  |
|       |                      |       |       |              |          |             |limit.        |
|       |                      |       |       |              |          |             |Therefore, no |
|       |                      |       |       |              |          |             |test is       |
|       |                      |       |       |              |          |             |proposed for  |
|       |                      |       |       |              |          |             |this impurity |
|       |                      |       |       |              |          |             |in the drug   |
|       |                      |       |       |              |          |             |substance     |
|       |                      |       |       |              |          |             |specification.|
|-------|----------------------|-------|-------|--------------|----------|-------------|--------------|
|5.     |3-Chloro-N-(3-        |7-HDQ  |3-MCA  |3             |Derek:    |Mutagenic    |Based on the  |
|       |methoxyphenyl)propion |       |       |              |Plausible |             |Justification |
|       |amide                 |       |       |              |Sarah:    |             |for fate and  |
|       |                      |       |       |              |Positive  |             |purge factor  |
|       |                      |       |       |              |          |             |and batch     |
|       |                      |       |       |              |          |             |analysis      |
|       |                      |       |       |              |          |             |provided in   |
|       |                      |       |       |              |          |             |subsequent    |
|       |                      |       |       |              |          |             |pages of this |
|       |                      |       |       |              |          |             |section, it   |
|       |                      |       |       |              |          |             |can be        |
|       |                      |       |       |              |          |             |concluded that|
|       |                      |       |       |              |          |             |the impurity  |
|       |                      |       |       |              |          |             |3-MCA is found|
|       |                      |       |       |              |          |             |less than 30% |
|       |                      |       |       |              |          |             |of TTC based  |
|       |                      |       |       |              |          |             |limit.        |
|       |                      |       |       |              |          |             |Therefore, no |
|       |                      |       |       |              |          |             |test is       |
|       |                      |       |       |              |          |             |proposed for  |
|       |                      |       |       |              |          |             |this impurity |
|       |                      |       |       |              |          |             |in the drug   |
|       |                      |       |       |              |          |             |substance     |
|       |                      |       |       |              |          |             |specification.|
-------------------------------------------------------------------------------------------------------"""
    ),
    (
        "table_page17",
        17,
        0,
        """-------------------------------------------------------------------------------------------
|6.     |3-Chloro-N-(3-        |7-HDQ  |3-HCA  |3      |Derek:   |Mutagenic|Based on the  |
|       |hydroxyphenyl)propiona|       |       |       |Plausible|         |Justification |
|       |mide                  |       |       |       |Sarah:   |         |for fate and  |
|       |                      |       |       |       |Positive |         |purge factor  |
|       |                      |       |       |       |         |         |and batch     |
|       |                      |       |       |       |         |         |analysis      |
|       |                      |       |       |       |         |         |provided in   |
|       |                      |       |       |       |         |         |subsequent    |
|       |                      |       |       |       |         |         |pages of this |
|       |                      |       |       |       |         |         |section, it   |
|       |                      |       |       |       |         |         |can be        |
|       |                      |       |       |       |         |         |concluded that|
|       |                      |       |       |       |         |         |the impurity  |
|       |                      |       |       |       |         |         |3-HCA is found|
|       |                      |       |       |       |         |         |less than 30% |
|       |                      |       |       |       |         |         |of TTC based  |
|       |                      |       |       |       |         |         |limit.        |
|       |                      |       |       |       |         |         |Therefore, no |
|       |                      |       |       |       |         |         |test is       |
|       |                      |       |       |       |         |         |proposed for  |
|       |                      |       |       |       |         |         |this impurity |
|       |                      |       |       |       |         |         |in the drug   |
|       |                      |       |       |       |         |         |substance     |
|       |                      |       |       |       |         |         |specification.|
|-------|----------------------|-------|-------|-------|---------|---------|--------------|
|7.     |3-Chloropropionyl     |7-HDQ  |3-CPC  |3      |Derek:   |Mutagenic|Based on the  |
|       |chloride              |       |       |       |Plausible|         |Justification |
|       |                      |       |       |       |Sarah:   |         |for fate and  |
|       |                      |       |       |       |Positive |         |purge factor  |
|       |                      |       |       |       |         |         |and batch     |
|       |                      |       |       |       |         |         |analysis      |
|       |                      |       |       |       |         |         |provided in   |
|       |                      |       |       |       |         |         |subsequent    |
|       |                      |       |       |       |         |         |pages of this |
|       |                      |       |       |       |         |         |section, it   |
|       |                      |       |       |       |         |         |can be        |
|       |                      |       |       |       |         |         |concluded that|
|       |                      |       |       |       |         |         |the impurity  |
|       |                      |       |       |       |         |         |3-CPC is found|
|       |                      |       |       |       |         |         |less than 30% |
|       |                      |       |       |       |         |         |of TTC based  |
|       |                      |       |       |       |         |         |limit.        |
|       |                      |       |       |       |         |         |Therefore, no |
|       |                      |       |       |       |         |         |test is       |
|       |                      |       |       |       |         |         |proposed for  |
|       |                      |       |       |       |         |         |this impurity |
|       |                      |       |       |       |         |         |in the drug   |
|       |                      |       |       |       |         |         |substance     |
|       |                      |       |       |       |         |         |specification.|
|-------|----------------------|-------|-------|-------|---------|---------|--------------|
|8.     |2-Chloropropane       |DCPP   |2-CRP  |2      |Derek:   |Mutagenic|Based on the  |
|       |                      |       |       |       |Plausible|         |Justification |
|       |                      |       |       |       |Sarah:   |         |for fate and  |
|       |                      |       |       |       |Positive |         |purge factor  |
|       |                      |       |       |       |         |         |and batch     |
|       |                      |       |       |       |         |         |analysis      |
|       |                      |       |       |       |         |         |provided in   |
|       |                      |       |       |       |         |         |subsequent    |
|       |                      |       |       |       |         |         |pages of this |
|       |                      |       |       |       |         |         |section, it   |
|       |                      |       |       |       |         |         |can be        |
|       |                      |       |       |       |         |         |concluded that|
|       |                      |       |       |       |         |         |the impurity  |
|       |                      |       |       |       |         |         |2-CRP is found|
|       |                      |       |       |       |         |         |less than 30% |
|       |                      |       |       |       |         |         |of TTC based  |
|       |                      |       |       |       |         |         |limit.        |
|       |                      |       |       |       |         |         |Therefore, no |
|       |                      |       |       |       |         |         |test is       |
|       |                      |       |       |       |         |         |proposed for  |
|       |                      |       |       |       |         |         |this impurity |
|       |                      |       |       |       |         |         |in the drug   |
|       |                      |       |       |       |         |         |substance     |
|       |                      |       |       |       |         |         |specification.|
-------------------------------------------------------------------------------------------"""
    ),
    (
        "table_page18",
        18,
        0,
        """| 9.   | 2,3-Dichloroaniline                        | DCPP  | DCN   | --     | Derek: Inactive                 | Non-Mutagenic  | Based on the QSAR evaluation, DCN is found to be          |
|      |                                            |       |       |        | Sarah: Negative                 |                | non-Mutagenic. Drug substance batches are tested for      |
|      |                                            |       |       |        |                                 |                | DCN content with validated GCMS method and results        |
|      |                                            |       |       |        |                                 |                | are found "Below detection limit".                        |
|      |                                            |       |       |        |                                 |                |                                                           |
|      |                                            |       |       |        |                                 |                | Batch data is presented hereunder                         |
|      |                                            |       |       |        |                                 |                | Batch no       | DCN content by GCMS                      |
|      |                                            |       |       |        |                                 |                | 3APR3/12001    | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | 3APR3/12002    | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | 3APR3/12003    | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | Detection limit| 2.49 ppm                                 |
|      |                                            |       |       |        |                                 |                | Quantitation   | 7.48 ppm                                 |
|      |                                            |       |       |        |                                 |                |                                                           |
|      |                                            |       |       |        |                                 |                | Based on batch, there is no carryover of DCN to           |
|      |                                            |       |       |        |                                 |                | final drug substance. Hence, control of this impurity     |
|      |                                            |       |       |        |                                 |                | in drug substance specification is not proposed.          |
| 10.  | N,N-Bis(2-chloroethyl)amine hydrochloride | DCPP  | CNS-1 | 3      | Derek: Plausible                | Mutagenic      | Based on the Justification for fate and purge factor      |
|      |                                            |       |       |        | Sarah: Positive                 |                | and batch analysis provided in subsequent pages of        |
|      |                                            |       |       |        |                                 |                | this section, it can be concluded that the impurity       |
|      |                                            |       |       |        |                                 |                | CNS-1 is found less than 30% of TTC based limit.          |
|      |                                            |       |       |        |                                 |                | Therefore, no test is proposed for this impurity in the   |
|      |                                            |       |       |        |                                 |                | drug substance specification."""
    ),
    (
        "table_page19",
        19,
        0,
        """| 11.  | 3,4-Dichloroaniline                        | DCPP  |3,4-DCN| --     | Derek: Inactive                 | Non-Mutagenic  | Based on the QSAR evaluation, 3,4-DCN is found to be      |
|      |                                            |       |       |        | Sarah: Negative                 |                | non-Mutagenic. Drug substance batches are tested for      |
|      |                                            |       |       |        |                                 |                | 3,4-DCN content with validated GCMS method and results     |
|      |                                            |       |       |        |                                 |                | are found 'Below detection limit'.                        |
|      |                                            |       |       |        |                                 |                |                                                           |
|      |                                            |       |       |        |                                 |                | Batch data is presented hereunder                         |
|      |                                            |       |       |        |                                 |                | Batch no       | 3,4-DCN content by GCMS                  |
|      |                                            |       |       |        |                                 |                | APR/C0944/16   | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | APR/C0944/17   | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | APR/C0944/18   | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | Detection limit| 0.51 ppm                                 |
|      |                                            |       |       |        |                                 |                | Quantitation   | 1.5 ppm                                  |
|      |                                            |       |       |        |                                 |                |                                                           |
|      |                                            |       |       |        |                                 |                | Based on batch, there is no carryover of DCN to           |
|      |                                            |       |       |        |                                 |                | final drug substance. Hence, control of this impurity     |
|      |                                            |       |       |        |                                 |                | in drug substance specification is not proposed.          |
| 12.  | 3-Chloroaniline                            | DCPP  | 3-CLA | 3      | Derek: Inactive                 | Non-Mutagenic  | Based on the QSAR evaluation, 3-CLA is found to be       |
|      |                                            |       |       |        | Sarah: Positive (7%)            |                | non-Mutagenic. Drug substance batches are tested for      |
|      |                                            |       |       |        |                                 |                | 3-CLA content with validated LCMS method and results are   |
|      |                                            |       |       |        |                                 |                | found 'Below detection limit'.                            |
|      |                                            |       |       |        |                                 |                |                                                           |
|      |                                            |       |       |        |                                 |                | Batch data is presented hereunder                         |
|      |                                            |       |       |        |                                 |                | Batch no       | 3-CLA content by LCMS                    |
|      |                                            |       |       |        |                                 |                | APR/C0944/16   | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | APR/C0944/17   | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | APR/C0944/18   | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | Detection limit| 0.51 ppm                                 |
|      |                                            |       |       |        |                                 |                | Quantitation   | 1.5 ppm                                  |
|      |                                            |       |       |        |                                 |                |                                                           |
|      |                                            |       |       |        |                                 |                | Based on batch, there is no carryover of 3-CLA to         |
|      |                                            |       |       |        |                                 |                | final drug substance. Hence, control of this impurity     |
|      |                                            |       |       |        |                                 |                | in drug substance specification is not proposed.          |"""
    ),
    (
        "table_page20",
        20,
        0,
        """| 13.  | 4-Chloroaniline                            | DCPP  | 4-CLA | 3      | Derek: Inactive                 | Mutagenic      | Based on the Justification for fate and purge factor      |
|      |                                            |       |       |        | Sarah: Positive                 |                | and batch analysis provided in subsequent pages of        |
|      |                                            |       |       |        |                                 |                | this section, it can be concluded that the impurity       |
|      |                                            |       |       |        |                                 |                | 4-CLA is found less than 30% of TTC based limit.          |
|      |                                            |       |       |        |                                 |                | Therefore, no test is proposed for this impurity in the   |
|      |                                            |       |       |        |                                 |                | drug substance specification.                             |
| 14.  | 2-Chloroaniline                            | DCPP  | 2-CLA | --     | Derek: Inactive                 | Non-Mutagenic  | Based on the QSAR evaluation, 2-CLA is found to be       |
|      |                                            |       |       |        | Sarah: Negative                 |                | non-Mutagenic. Drug substance batches are tested for      |
|      |                                            |       |       |        |                                 |                | 2-CLA content with validated HPLC method and results are   |
|      |                                            |       |       |        |                                 |                | found 'Below detection limit'.                            |
|      |                                            |       |       |        |                                 |                |                                                           |
|      |                                            |       |       |        |                                 |                | Batch data is presented hereunder                         |
|      |                                            |       |       |        |                                 |                | Batch no       | 2-CLA content by HPLC                    |
|      |                                            |       |       |        |                                 |                | APR/C0944/16   | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | APR/C0944/17   | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | APR/C0944/18   | Below detection limit                    |
|      |                                            |       |       |        |                                 |                | Detection limit| 0.51 ppm                                 |
|      |                                            |       |       |        |                                 |                | Quantitation   | 1.5 ppm                                  |
|      |                                            |       |       |        |                                 |                |                                                           |
|      |                                            |       |       |        |                                 |                | Based on batch, there is no carryover of 2-CLA to         |
|      |                                            |       |       |        |                                 |                | final drug substance. Hence, control of this impurity     |
|      |                                            |       |       |        |                                 |                | in drug substance specification is not proposed.          |
| 15.  | 1,2-Dichloro-3-nitrobenzene                | DCPP  | DNB   | 3      | Derek: Plausible                | Mutagenic      | Based on the Justification for fate and purge factor      |
|      |                                            |       |       |        | Sarah: Positive                 |                | and batch analysis provided in subsequent pages of        |
|      |                                            |       |       |        |                                 |                | this section, it can be concluded that the impurity DNB   |
|      |                                            |       |       |        |                                 |                | is found less than 30% of TTC based limit.                |
|      |                                            |       |       |        |                                 |                | Therefore, no test is proposed for this impurity in the   |
|      |                                            |       |       |        |                                 |                | drug substance specification.                             |"""
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
|             |            |There is a   |
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
        "table_page22",
        22,
        0,
        """----------------------------------
|TABLE-1                         |
|--------------------------------|
|Batch No.   |Results [ppm]      |
|            |-------------------|
|            |NDMA     |NDEA     |
|------------|---------|---------|
|3ARP320001  |Below    |Below    |
|            |detection|detection|
|            |limit    |limit    |
|------------|---------|---------|
|3ARP320002  |Below    |Below    |
|            |detection|detection|
|            |limit    |limit    |
|------------|---------|---------|
|3ARP320003  |Below    |Below    |
|            |detection|detection|
|            |limit    |limit    |
|------------|---------|---------|
|Detection   |0.01     |0.01     |
|Limit       |         |         |
|------------|---------|---------|
|Quantitation|0.03     |0.03     |
|Limit       |         |         |
----------------------------------"""
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

