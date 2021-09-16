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

LiqT2=pd.DataFrame(data={"SiO2_Liq": 61,
                            "TiO2_Liq": 0.45,
                            "Al2O3_Liq": 18.56,
                            "FeOt_Liq": 3.17,
                            "MnO_Liq": 0.27,
                            "MgO_Liq": 0.23,
                            "CaO_Liq": 1.64,
                            "Na2O_Liq": 6.11,
                            "K2O_Liq": 7.09,
                            "Cr2O3_Liq": 0,
                            "P2O5_Liq": 0.02,
                            "H2O_Liq": 2,
                            "Fe3Fet_Liq":0,
}, index=[0])

PlagT=pd.DataFrame(data={"SiO2_Plag": 57.3,
                            "TiO2_Plag": 0.09,
                            "Al2O3_Plag": 26.6,
                            "FeOt_Plag": 0.43,
                            "MnO_Plag": 0,
                            "MgO_Plag": 0.03,
                            "CaO_Plag": 8.83,
                            "Na2O_Plag": 6.11,
                            "K2O_Plag": 0.49,
                            "Cr2O3_Plag": 0.0}, index=[0])

KsparT=pd.DataFrame(data={"SiO2_Kspar": 65.5,
                            "TiO2_Kspar": 0.0,
                            "Al2O3_Kspar": 19.6,
                            "FeOt_Kspar": 0.07,
                            "MnO_Kspar": 0,
                            "MgO_Kspar": 0.0,
                            "CaO_Kspar": 0.75,
                            "Na2O_Kspar": 4.81,
                            "K2O_Kspar": 9.36,
                            "Cr2O3_Kspar": 0.0}, index=[0])
decimalPlace=4


class test_fspar_liq_temp(unittest.TestCase):
    def test_eq23(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_temp(plag_comps=PlagT, liq_comps=LiqT,
        equationT="T_Put2008_eq23", P=5)[0], 1295.359719, decimalPlace,
          "P eq23 not equal to test value")


    def test_eq24a(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_temp(plag_comps=PlagT, liq_comps=LiqT,
        equationT="T_Put2008_eq24a", P=5)[0], 1308.200806, decimalPlace,
          "P eq23 not equal to test value")

    def test_eq24a_noH(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_temp(plag_comps=PlagT, liq_comps=LiqT,
        equationT="T_Put2008_eq24a", P=5, H2O_Liq=0)[0], 1483.297388, decimalPlace,
          "P eq23 not equal to test value")

    def test_eq24b(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_temp(kspar_comps=KsparT, liq_comps=LiqT2,
        equationT="T_Put2008_eq24b", P=5)[0], 1207.97364620, decimalPlace,
          "P eq23 not equal to test value")


class test_fspar_liq_press(unittest.TestCase):
    def test_eq25(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_press(plag_comps=PlagT, liq_comps=LiqT,
        equationP="P_Put2008_eq25", T=1000)[0], -7.199104, decimalPlace,
          "P eq23 not equal to test value")


class test_fspar_liq_press_temp(unittest.TestCase):
    def test_eq25_24a_press(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_press_temp(plag_comps=PlagT, liq_comps=LiqT,
        equationP="P_Put2008_eq25", equationT="T_Put2008_eq24a").P_kbar_calc[0], 7.005588, decimalPlace,
          "P from iter 25-24a  not equal to test value")

    def test_eq25_24a_temp(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_press_temp(plag_comps=PlagT, liq_comps=LiqT,
        equationP="P_Put2008_eq25", equationT="T_Put2008_eq24a").T_K_calc[0], 1319.769669, decimalPlace,
          "T from iter 25-24a  not equal to test value")

class test_fspar_liq_hygro(unittest.TestCase):
    def test_Waters2015_H2O(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_hygr(liq_comps=LiqT,
        plag_comps=PlagT, equationH="H_Waters2015", T=1000+273.15, P=1).H2O_calc[0], 5.148449, decimalPlace,
        "H2O from Waters and Lange 2015 not equal to test value")

    def test_Waters2015_H2O_An_Ab(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_hygr(liq_comps=LiqT, XAn=0.5,
        XAb=0.4, equationH="H_Waters2015", T=1000+273.15, P=1).H2O_calc[0], 5.289477, decimalPlace,
        "H2O from Waters and Lange 2015 with Xan Xab not equal to test value")


    def test_Put2005_eqH_H2O(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_hygr(liq_comps=LiqT,
        plag_comps=PlagT, equationH="H_Put2005_eqH", T=1000+273.15, P=1).H2O_calc[0], 9.646943, decimalPlace,
        "T from iter 25-24a  not equal to test value")


    def test_Put2005_eqH_Obs_Kd_Ab_An(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_hygr(liq_comps=LiqT,
        plag_comps=PlagT, equationH="H_Put2005_eqH", T=1000+273.15, P=1).
        Obs_Kd_Ab_An[0], 0.65105, decimalPlace,
        "T from iter 25-24a  not equal to test value")


class test_fspar_liq_temp_hygr(unittest.TestCase):
    def test_Waters2015_eq23_H2O(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_temp_hygr(plag_comps=PlagT, liq_comps=LiqT, equationT="T_Put2008_eq23",
                                       equationH="H_Waters2015", iterations=10, P=10).get("T_H_calc").H2O_calc[0], 3.820911, decimalPlace,
        "H2O from iter 23-Waters2015  not equal to test value")

    def test_Waters2015_eq23_T(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_temp_hygr(plag_comps=PlagT, liq_comps=LiqT, equationT="T_Put2008_eq23",
                                       equationH="H_Waters2015", iterations=10, P=10).get("T_H_calc").T_K_calc[0], 1362.263707, decimalPlace,
        "T from iter 23-Waters2015  not equal to test value")


    def test_Waters2015_eq24a_H2O(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_temp_hygr(plag_comps=PlagT, liq_comps=LiqT, equationT="T_Put2008_eq24a",
                                       equationH="H_Waters2015", iterations=10, P=10).get("T_H_calc").H2O_calc[0], 3.380639, decimalPlace,
        "H2O from iter 24a-Waters2015  not equal to test value")

    def test_Put2005_eq23_H2O(self):
        self.assertAlmostEqual(pt.calculate_fspar_liq_temp_hygr(plag_comps=PlagT, liq_comps=LiqT, equationT="T_Put2008_eq24a",
                                       equationH="H_Put2005_eqH", iterations=10, P=10).get("T_H_calc").H2O_calc[0], 9.09812, decimalPlace,
        "H2O from iter 23-Putirka 2005  not equal to test value")


if __name__ == '__main__':
     unittest.main()