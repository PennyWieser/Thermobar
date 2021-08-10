import unittest
import pandas as pd
import sys
sys.path.append("..")
import Thermobar as pt

CpxT=pd.DataFrame(data={"SiO2_Cpx": 49,
                            "TiO2_Cpx": 0.3,
                            "Al2O3_Cpx": 4.89,
                            "FeOt_Cpx": 5.71,
                            "MnO_Cpx": 0.165,
                            "MgO_Cpx": 16.89,
                            "CaO_Cpx": 20.319,
                            "Na2O_Cpx": 0.319,
                            "K2O_Cpx": 0.1,
                            "Cr2O3_Cpx": 0.11}, index=[0])

OpxT=pd.DataFrame(data={"SiO2_Opx": 55,
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


Cpx_Several=pd.concat([CpxT, CpxT+0.1, CpxT-0.1, CpxT-0.02])
Opx_Several=pd.concat([OpxT, OpxT+0.2, OpxT-0.05])

class test_cpx_opx_press_temp_matching(unittest.TestCase):
    def test_36_38a_match_press(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp_matching(
        cpx_comps=Cpx_Several, opx_comps=Opx_Several, equationT="T_Put2008_eq36",
        equationP="P_Put2008_eq38").get("All_PTs").P_kbar_calc[0], 2.9944999,
        decimalPlace, "Kd Cpx-Opx not equal to test value")

    def test_36_38a_match_press(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp_matching(
        cpx_comps=Cpx_Several, opx_comps=Opx_Several, equationT="T_Put2008_eq36",
        equationP="P_Put2008_eq38").get("Av_PTs_perCPX").Mean_T_K_calc[0], 1311.457528,
        decimalPlace, "Kd Cpx-Opx not equal to test value")

    def test_36_38a_match_KdFilt_Temp(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp_matching(
        cpx_comps=Cpx_Several, opx_comps=Opx_Several, equationT="T_Put2008_eq36",
        equationP="P_Put2008_eq38", KdMatch=1, KdErr=0.1).get("Av_PTs_perCPX")
        .Mean_T_K_calc[0], 1310.2199857766054,
        decimalPlace, "Kd Cpx-Opx not equal to test value")

    def test_36_38a_match_KdFiltHT_Temp(self):
        self.assertAlmostEqual(pt.calculate_cpx_opx_press_temp_matching(
        cpx_comps=Cpx_Several, opx_comps=Opx_Several, equationT="T_Put2008_eq36",
        equationP="P_Put2008_eq38", KdMatch="Subsolidus").get("Av_PTs_perCPX")
        .Mean_T_K_calc[0], 1325.070489,
        decimalPlace, "Kd Cpx-Opx not equal to test value")


if __name__ == '__main__':
    unittest.main()