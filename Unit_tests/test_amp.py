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

AmpT=pd.DataFrame(data={"SiO2_Amp": 40.57,
                     "TiO2_Amp": 2.45,
                     "Al2O3_Amp": 12.82,
                     "FeOt_Amp": 13.110,
                     "MnO_Amp": 0.26,
                     "MgO_Amp": 13.02,
                     "CaO_Amp": 11.63,
                     "Na2O_Amp": 2.20,
                     "K2O_Amp": 0.92,
                     "Cr2O3_Amp": 0.01,
                       'F_Amp': 0,
                       'Cl_Amp': 0}, index=[0])
decimalPlace=4


class test_amp_only_press(unittest.TestCase):
    def test_press_ridolfi21(self):
        self.assertAlmostEqual(pt.calculate_amp_only_press(amp_comps=AmpT,
     equationP="P_Ridolfi2021").P_kbar_calc[0], 4.589114, decimalPlace, "P Rildofi2021 not equal to test value")

    def test_press_Mutch(self):
        self.assertAlmostEqual(pt.calculate_amp_only_press(amp_comps=AmpT,
     equationP="P_Mutch2016").P_kbar_calc[0], 6.251692054556109, decimalPlace,
      "P Mutch2016 not equal to test value")


    def test_press_Anderson1995(self):
        self.assertAlmostEqual(pt.calculate_amp_only_press(amp_comps=AmpT,
     equationP="P_Anderson1995", T=1100)[0], 4.199270205779337,
     decimalPlace, "P Anderson 1995 not equal to test value")

class test_amp_only_temp(unittest.TestCase):
    def test_press_eq5(self):
        self.assertAlmostEqual(pt.calculate_amp_only_temp(amp_comps=AmpT,
     equationT="T_Put2016_eq5")[0], 1229.1534012, decimalPlace,
     "T eq5 not equal to test value")

    def test_press_eq8(self):
        self.assertAlmostEqual(pt.calculate_amp_only_temp(amp_comps=AmpT,
     equationT="T_Put2016_eq8", P=6)[0], 1227.6548856, decimalPlace,
     "T eq5 not equal to test value")

class test_amp_press_temp(unittest.TestCase):
    def test_eq5_anderson_press(self):
        self.assertAlmostEqual(pt.calculate_amp_only_press_temp(amp_comps=AmpT,
     equationP="P_Anderson1995", equationT="T_Put2016_eq8").P_kbar_calc[0], 0.10279462682872165,
     decimalPlace, "Iterating eq 8 and P Anderson 1995 P not equal to test value")

    def test_eq5_anderson_temp(self):
        self.assertAlmostEqual(pt.calculate_amp_only_press_temp(amp_comps=AmpT,
     equationP="P_Anderson1995", equationT="T_Put2016_eq8").T_K_calc[0], 1203.682745837,
     decimalPlace, "Iterating eq 8 and P Anderson 1995 P not equal to test value")

    def test_eq5_Ridolfi_press(self):
        self.assertAlmostEqual(pt.calculate_amp_only_press_temp(amp_comps=AmpT,
     equationP="P_Ridolfi2021", equationT="T_Put2016_eq8").P_kbar_calc[0], 4.589113613,
     decimalPlace, "Iterating eq 8 and P Anderson 1995 P not equal to test value")

    def test_eq5_Ridolfi_temp(self):
        self.assertAlmostEqual(pt.calculate_amp_only_press_temp(amp_comps=AmpT,
     equationP="P_Ridolfi2021", equationT="T_Put2016_eq8").T_K_calc[0], 1221.919632516,
     decimalPlace, "Iterating eq 8 and P Anderson 1995 P not equal to test value")

class test_amp_liq_temp(unittest.TestCase):
    def test_eq4a_amp_sat(self):
        self.assertAlmostEqual(pt.calculate_amp_liq_temp(liq_comps=LiqT, amp_comps=AmpT,
     equationT="T_Put2016_eq4a_amp_sat")[0], 1247.384143,
     decimalPlace, "T from eq4a amp sat is not equal to test value")

    def test_eq4b(self):
        self.assertAlmostEqual(pt.calculate_amp_liq_temp(liq_comps=LiqT, amp_comps=AmpT,
     equationT="T_Put2016_eq4b")[0], 1234.702307,
     decimalPlace, "T from eq4b is not equal to test value")

    def test_eq4b_noH(self):
        self.assertAlmostEqual(pt.calculate_amp_liq_temp(liq_comps=LiqT, amp_comps=AmpT,
    equationT="T_Put2016_eq4b", H2O_Liq=0)[0], 1220.480674,
    decimalPlace, "T from eq4b no H is not equal to test value")

class test_amp_liq_press(unittest.TestCase):
    def test_eq7a_noH(self):
        self.assertAlmostEqual(pt.calculate_amp_liq_press(liq_comps=LiqT,
    amp_comps=AmpT, equationP="P_Put2016_eq7a", H2O_Liq=0)[0], 2.701862,
    decimalPlace, "P from equation 7a not equal to test value")

    def test_eq7b_noH(self):
        self.assertAlmostEqual(pt.calculate_amp_liq_press(liq_comps=LiqT,
     amp_comps=AmpT, equationP="P_Put2016_eq7b", H2O_Liq=0)[0], 0.495501,
     decimalPlace, "P from equation 7a not equal to test value")

    def test_eq7b(self):
        self.assertAlmostEqual(pt.calculate_amp_liq_press(liq_comps=LiqT,
     amp_comps=AmpT, equationP="P_Put2016_eq7b")[0], 4.359844,
     decimalPlace, "P from equation 7a not equal to test value")


    def test_eq7b_Kd(self):
        self.assertAlmostEqual(pt.calculate_amp_liq_press(liq_comps=LiqT,
     amp_comps=AmpT, equationP="P_Put2016_eq7b", eq_tests=True).get("Kd-Fe-Mg")[0], 0.854897,
     decimalPlace, "Kd Amp-liq not equal to test value")

class test_amp_liq_press_temp(unittest.TestCase):
    def test_eq7a_4b_press(self):
        self.assertAlmostEqual(pt.calculate_amp_liq_press_temp(liq_comps=LiqT,
     amp_comps=AmpT, equationP="P_Put2016_eq7a", equationT="T_Put2016_eq4b").P_kbar_calc[0],
      5.264159, decimalPlace, "P from iterating 7a and 4b not equal to test value")

    def test_eq7a_4b_temp(self):
        self.assertAlmostEqual(pt.calculate_amp_liq_press_temp(liq_comps=LiqT,
     amp_comps=AmpT, equationP="P_Put2016_eq7a", equationT="T_Put2016_eq4b").T_K_calc[0],
      1234.702307, decimalPlace, "P from iterating 7a and 4b not equal to test value")


if __name__ == '__main__':
     unittest.main()