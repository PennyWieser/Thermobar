import unittest
import pandas as pd
import Thermobar as pt
LiqT=pd.DataFrame(data={"SiO2_Liq": 51,
                            "TiO2_Liq": 0.48,
                            "Al2O3_Liq": 19,
                            "FeOt_Liq": 5.3,
                            "MnO_Liq": 0.1,
                            "MgO_Liq": 4.5,
                            "CaO_Liq": 9,
                            "Na2O_Liq": 4.2,
                            "K2O_Liq": 0.1,
                            "Cr2O3_Liq": 0.11,
                            "P2O5_Liq": 0.11,
                            "H2O_Liq": 5,
 "Fe3Fet_Liq":0.1,
}, index=[0])

CpxT=pd.DataFrame(data={"Sample_ID_Cpx": 'test',
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
decimalPlace=4


class test_cpx_only_press(unittest.TestCase):
    def test_press_32a(self):
       self.assertAlmostEqual(pt.calculate_cpx_only_press(cpx_comps=CpxT, equationP="P_Put2008_eq32a",
         T=1300)[0], 3.668079, decimalPlace, "equation 32a not equal to test value")

    def test_press_32b_noH(self):
       self.assertAlmostEqual(pt.calculate_cpx_only_press(cpx_comps=CpxT, equationP="P_Put2008_eq32b",
         T=1300)[0], -1.575924, decimalPlace, "equation 32b not equal to test value")

    def test_press_32b_6H(self):
       self.assertAlmostEqual(pt.calculate_cpx_only_press(cpx_comps=CpxT, equationP="P_Put2008_eq32b",
         T=1300, H2O_Liq=6)[0], 1.142076, decimalPlace, "equation 32b (changed H2O) not equal to typed value")


    def test_press_Petrelli(self):
       self.assertAlmostEqual(pt.calculate_cpx_only_press(cpx_comps=CpxT, equationP="P_Petrelli2021_Cpx_only").P_kbar_calc[0],
       5.435313, decimalPlace, "Petrelli 2021 not equal to typed value")

class test_cpx_only_temp(unittest.TestCase):
    def test_temp_32d(self):
       self.assertAlmostEqual(pt.calculate_cpx_only_temp(cpx_comps=CpxT, equationT="T_Put2008_eq32d",
         P=5)[0], 1479.055921, decimalPlace, "equation 32d not equal to test value")

    def test_temp_32d_subsol(self):
       self.assertAlmostEqual(pt.calculate_cpx_only_temp(cpx_comps=CpxT, equationT="T_Put2008_eq32d_subsol",
         P=5)[0], 1252.508521, decimalPlace, "equation 32d not equal to test value")

    def test_temp_Petrelli(self):
       self.assertAlmostEqual(pt.calculate_cpx_only_temp(cpx_comps=CpxT, equationT="T_Petrelli2021_Cpx_only").T_K_calc[0],
       1377.2655555555673, decimalPlace, "Petrelli 2021 not equal to typed value")


class test_cpx_only_press_temp(unittest.TestCase):
    def test_32a_32d_press(self):
       self.assertAlmostEqual(pt.calculate_cpx_only_press_temp(cpx_comps=CpxT, equationT="T_Put2008_eq32d",
       equationP="P_Put2008_eq32a").P_kbar_calc[0], 5.821624, decimalPlace,
       "Calc P from iterating eq32d-32a not equal to test value")

    def test_32a_32d_temp(self):
       self.assertAlmostEqual(pt.calculate_cpx_only_press_temp(cpx_comps=CpxT, equationT="T_Put2008_eq32d",
       equationP="P_Put2008_eq32a").T_K_calc[0], 1485.955147, decimalPlace,
       "Calc T from iterating eq32d-32a not equal to test value")

class test_cpx_liq_temp(unittest.TestCase):
    def test_put33(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_temp(cpx_comps=CpxT,
       liq_comps=LiqT, equationT="T_Put2008_eq33", P=5)[0], 1352.020448,
       decimalPlace,
       "Calc T from  T_Put2008_eq33 not equal to test value")


class test_cpx_liq_press(unittest.TestCase):
    def test_maseq1(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Mas2013_eqPalk1", T=800)[0], -0.77056,
       decimalPlace,
       "Calc P from  P_Mas2013_eqPalk1 not equal to test value")

    def test_Put32c(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Put2008_eq32c", T=1300)[0], 3.472161,
       decimalPlace,
       "Calc P from  P_Put2008_eq32c not equal to test value")

    def test_Put32_DeltaKd(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Put2008_eq32c", T=1300,
       eq_tests=True).Delta_Kd_Put2008[0], 0.079459,
       decimalPlace,
       "Calc Delta Kd from  P_Put2008_eq32c not equal to test value")

    def test_Put32_deltaKd_30fe(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Put2008_eq32c", T=1300,
       eq_tests=True, Fe3Fet_Liq=0.3).Delta_Kd_Put2008[0], 0.170583,
       decimalPlace,
       "Calc Delta Kd from  P_Put2008_eq32c not equal to test value")


    def test_Put32_DeltaEnFs(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Put2008_eq32c", T=1300,
       eq_tests=True).Delta_EnFs[0], 0.073576,
       decimalPlace,
       "Calc Delta EnFs from  P_Put2008_eq32c not equal to test value")


    def test_Put32_DeltaCaTs(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Put2008_eq32c", T=1300,
       eq_tests=True).Delta_CaTs[0], 0.017332,
       decimalPlace,
       "Calc Delta CaTs from  P_Put2008_eq32c not equal to test value")

    def test_Put32_DeltaDiHd(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Put2008_eq32c", T=1300,
       eq_tests=True).Delta_DiHd[0], 0.006175,
       decimalPlace,
       "Calc Delta DiHd from  P_Put2008_eq32c not equal to test value")


class test_cpx_liq_press_temp(unittest.TestCase):
    def test_put30_Put2003_press(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press_temp(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Put2008_eq30", equationT="T_Put2003").P_kbar_calc[0],
        7.499172, decimalPlace,
       "Calc P from iterating eq30-Put2003 not equal to test value")

    def test_put30_Put2003_temp(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press_temp(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Put2008_eq30", equationT="T_Put2003").T_K_calc[0],
        1416.389119, decimalPlace,
       "Calc T from iterating eq30-Put2003 not equal to test value")

    def test_put30_Put2003_press_6H(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press_temp(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Put2008_eq30", equationT="T_Put2003", H2O_Liq=6).P_kbar_calc[0],
        8.291964, decimalPlace,
       "Calc P from iterating eq30-Put2003 not equal to test value")

    def test_put30_Put2003_temp_6H(self):
       self.assertAlmostEqual(pt.calculate_cpx_liq_press_temp(cpx_comps=CpxT,
       liq_comps=LiqT, equationP="P_Put2008_eq30", equationT="T_Put2003", H2O_Liq=6).T_K_calc[0],
        1417.395008, decimalPlace,
       "Calc T from iterating eq30-Put2003 not equal to test value")

Liq2=pd.DataFrame(data={"SiO2_Liq": 47.2916,
                            "TiO2_Liq": 1.7307,
                            "Al2O3_Liq": 15.525,
                            "FeOt_Liq": 9.3999,
                            "MnO_Liq": 0.1588,
                            "MgO_Liq": 6.322718447,
                            "CaO_Liq": 12.3696,
                            "Na2O_Liq": 3.9281,
                            "K2O_Liq": 1.2285,
                            "Cr2O3_Liq": 0,
                            "P2O5_Liq": 0.2406,
                            "H2O_Liq": 0}, index=[0])

Cpx2=pd.DataFrame(data={"SiO2_Cpx": 49.7147,
                            "TiO2_Cpx": 0.7249,
                            "Al2O3_Cpx": 6.2489,
                            "FeOt_Cpx": 3.988,
                            "MnO_Cpx": 0.1128,
                            "MgO_Cpx": 15.0415,
                            "CaO_Cpx": 21.6397,
                            "Na2O_Cpx": 0.4081,
                            "K2O_Cpx": 0,
                            "Cr2O3_Cpx": 1.355}, index=[0])


class test_cpx_liq_melt_matching(unittest.TestCase):
   def test_eq33_P2017_press(self):
      self.assertAlmostEqual(pt.calculate_cpx_liq_press_temp_matching(cpx_comps=Cpx2, liq_comps=Liq2,
      equationT="T_Put2008_eq33", equationP="P_Neave2017", KdErr=0.11, eq_crit="All",
Fe3Fet_Liq=0.0, sigma=4).get("Av_PTs").Mean_P_kbar_calc[0], 5.993997, decimalPlace,
"Calc P from melt matching eq33-N2017 not equal to test value")
#
   def test_eq33_P2017_temp(self):
      self.assertAlmostEqual(pt.calculate_cpx_liq_press_temp_matching(cpx_comps=Cpx2, liq_comps=Liq2,
equationT="T_Put2008_eq33", equationP="P_Neave2017", KdErr=0.11, eq_crit="All",
Fe3Fet_Liq=0.0, sigma=4).get("Av_PTs").Mean_T_K_calc[0], 1480.039007, decimalPlace,
       "Calc T from melt matching eq33-N2017 not equal to test value")


class test_cpx_equilibirum(unittest.TestCase):
   def test_rhodes_lines_cpx_28(self):
      self.assertAlmostEqual(pt.calculate_cpx_rhodes_diagram_lines(Min_Mgno=0.4,
      Max_Mgno=0.7, T=1300, KdMin=0.2, KdMax=0.3).get("Eq_Cpx_Mg# (Kd=0.28)")[0],
      0.7042253, decimalPlace,
       "Calc Mg# from Kd=0.28 not equal to test value")

   def test_rhodes_lines_cpx_eq35(self):
      self.assertAlmostEqual(pt.calculate_cpx_rhodes_diagram_lines(Min_Mgno=0.4,
      Max_Mgno=0.7, T=1300, KdMin=0.2, KdMax=0.3).get("Eq_Cpx_Mg# (Eq 35 P2008)-0.08")[0],
      0.8069644, decimalPlace,
       "Calc Mg# from eq35 not equal to test value")




# if __name__ == '__main__':
#     unittest.main()

if __name__ == '__main__':
    unittest.main()