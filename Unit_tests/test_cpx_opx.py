import unittest
import pandas as pd
import Thermobar as pt
import numpy as np

CpxT=pd.DataFrame(data={"Sample_ID_Cpx": 'test1',
                        "SiO2_Cpx": 49,
                            "TiO2_Cpx": 0.3,
                            "Al2O3_Cpx": 4.89,
                            "FeOt_Cpx": 5.71,
                            "MnO_Cpx": 0.165,
                            "MgO_Cpx": 16.89,
                            "CaO_Cpx": 20.319,
                            "Na2O_Cpx": 0.319,
                            "K2O_Cpx": 0.1,
                            "Cr2O3_Cpx": 0.11}, index=[0])

OpxT=pd.DataFrame(data={"Sample_ID_Opx": 'test2',
                            "SiO2_Opx": 55,
                            "TiO2_Opx": 0.34,
                            "Al2O3_Opx": 1.5,
                            "FeOt_Opx": 11.3,
                            "MnO_Opx": 0.24,
                            "MgO_Opx": 30.7,
                            "CaO_Opx": 0.9,
                            "Na2O_Opx": 0.01,
                            "K2O_Opx": 0,
                            "Cr2O3_Opx": 0.19}, index=[0])
decimalPlace=4




class test_cpx_opx_press(unittest.TestCase):
    def test_press_38(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press(opx_comps=OpxT,
        cpx_comps=CpxT, equationP="P_Put2008_eq38")[0], 2.9945,
        decimalPlace, "P from equation 38 not equal to test value")

    def test_press_39(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press(opx_comps=OpxT,
        cpx_comps=CpxT, equationP="P_Put2008_eq39", T=1400)[0], 4.695694,
        decimalPlace, "P from equation 38 not equal to test value")

class test_cpx_opx_temp(unittest.TestCase):
    def test_temp_36(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_temp(opx_comps=OpxT,
        cpx_comps=CpxT, equationT="T_Put2008_eq36", P=3)[0], 1305.519026,
        decimalPlace, "T from equation 38 not equal to test value")


    def test_temp_37(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_temp(opx_comps=OpxT,
        cpx_comps=CpxT, equationT="T_Put2008_eq37", P=3)[0], 1317.286386,
        decimalPlace, "T from equation 38 not equal to test value")

class test_cpx_opx_press_temp(unittest.TestCase):
    def test_36_38_press(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp(opx_comps=OpxT,
        cpx_comps=CpxT, equationT="T_Put2008_eq36", equationP="P_Put2008_eq38")
        .P_kbar_calc[0], 2.9945,
        decimalPlace, "T from eq36-eq38 iter not equal to test value")

    def test_36_38_temp(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp(opx_comps=OpxT,
        cpx_comps=CpxT, equationT="T_Put2008_eq36", equationP="P_Put2008_eq38")
        .T_K_calc[0], 1305.489718,
        decimalPlace, "P from eq36-eq38 iter not equal to test value")

    def test_36_38_temp_Kd(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp(opx_comps=OpxT,
        cpx_comps=CpxT, equationT="T_Put2008_eq36", equationP="P_Put2008_eq38",
        eq_tests=True).Kd_Fe_Mg_Cpx_Opx[0], 0.918473,
        decimalPlace, "Kd Cpx-Opx not equal to test value")

OpxT1=OpxT.copy()
num_cols1 = OpxT1.select_dtypes(include=[np.number]).columns
OpxT1[num_cols1] = OpxT1[num_cols1].apply(lambda x: x + 0.2)

OpxT2=OpxT.copy()
num_cols2 = OpxT2.select_dtypes(include=[np.number]).columns
OpxT2[num_cols2] = OpxT2[num_cols2].apply(lambda x: x -0.05)

CpxT1=CpxT.copy()
num_cols1 = CpxT1.select_dtypes(include=[np.number]).columns
CpxT1[num_cols1] = CpxT1[num_cols1].apply(lambda x: x +0.1)

CpxT3=CpxT.copy()
num_cols1 = CpxT3.select_dtypes(include=[np.number]).columns
CpxT3[num_cols1] = CpxT3[num_cols1].apply(lambda x: x -0.1)

CpxT2=CpxT.copy()
num_cols2 = CpxT2.select_dtypes(include=[np.number]).columns
CpxT2[num_cols2] = CpxT2[num_cols2].apply(lambda x: x -0.02)



Cpx_Several=pd.concat([CpxT, CpxT1, CpxT3, CpxT2]).reset_index(drop=True)
Opx_Several=pd.concat([OpxT, OpxT1, OpxT2]).reset_index(drop=True)

class test_cpx_opx_press_temp_matching(unittest.TestCase):

    def test_36_38a_match_KdFiltHT_Temp(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp_matching(
        cpx_comps=Cpx_Several, opx_comps=Opx_Several, equationT="T_Put2008_eq36",
        equationP="P_Put2008_eq38", Kd_Match="Subsolidus").get("Av_PTs_perCPX")
        .Mean_T_K_calc[0], 1325.070489,
        decimalPlace, "Calc T not equal to test value")

    def test_36_38a_match_press(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp_matching(
        cpx_comps=Cpx_Several, opx_comps=Opx_Several, equationT="T_Put2008_eq36",
        equationP="P_Put2008_eq38",  return_all_pairs=True).get("All_PTs").P_kbar_calc[0], 2.9944999,
        decimalPlace, "Calc P not equal to test value")

    def test_36_38a_match_press(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp_matching(
        cpx_comps=Cpx_Several, opx_comps=Opx_Several, equationT="T_Put2008_eq36",
        equationP="P_Put2008_eq38",  return_all_pairs=True).get("Av_PTs_perCPX").Mean_T_K_calc[0], 1311.457528,
        decimalPlace, "Calc T not equal to test value")

    def test_36_38a_match_KdFilt_Temp(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp_matching(
        cpx_comps=Cpx_Several, opx_comps=Opx_Several, equationT="T_Put2008_eq36",
        equationP="P_Put2008_eq38", Kd_Match=1, Kd_Err=0.1).get("Av_PTs_perCPX")
        .Mean_T_K_calc[0], 1311.6940596719433,
        decimalPlace, "Calc T not equal to test value")



if __name__ == '__main__':
    unittest.main()