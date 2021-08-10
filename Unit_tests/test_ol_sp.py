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


OlT=pd.DataFrame(data={"SiO2_Ol": 40.5,
                            "TiO2_Ol": 0.02,
                            "Al2O3_Ol": 0.08,
                            "FeOt_Ol": 12.40,
                            "MnO_Ol": 0.17,
                            "MgO_Ol": 47.4,
                            "CaO_Ol": 0.30,
                            "Na2O_Ol": 0,
                            "K2O_Ol": 0,
                            "Cr2O3_Ol": 0.03},  index=[0])


SpT=pd.DataFrame(data={"SiO2_Sp": 0,
                            "TiO2_Sp": 0.248333,
                            "Al2O3_Sp":40.09866,
                            "FeOt_Sp": 11.805333,
                            "MnO_Sp": 0.16866,
                            "MgO_Sp": 19.131667,
                            "CaO_Sp": 0.006667,
                            "Na2O_Sp": 0.000667,
                            "K2O_Sp": 0,
                            "Cr2O3_Sp": 27.107333,
                            "NiO_Sp": 0.239000}, index=[0])

decimalPlace=4


class test_liq_only_temp(unittest.TestCase):
    def test_HT87(self):

        self.assertAlmostEqual(pt.calculate_liq_only_temp(liq_comps=LiqT,
        equationT="T_Helz1987_MgO")[0], 1377.6,
        decimalPlace, "T from Helz and Thornber not equal to test value")

    def test_eq15(self):

        self.assertAlmostEqual(pt.calculate_liq_only_temp(liq_comps=LiqT,
        equationT="T_Put2008_eq15", P=5)[0], 1347.12506667889,
        decimalPlace, "T from eq15 not equal to test value")

    def test_eq15_noH(self):

        self.assertAlmostEqual(pt.calculate_liq_only_temp(liq_comps=LiqT,
        equationT="T_Put2008_eq15", P=5, H2O_Liq=0)[0], 1411.2750666788997,
        decimalPlace, "T from eq15 not equal to test value")

    def test_eq22_Beatt(self):

        self.assertAlmostEqual(pt.calculate_liq_only_temp(liq_comps=LiqT,
        equationT="T_Put2008_eq22_BeattDMg", P=5, H2O_Liq=0)[0], 1401.874871,
        decimalPlace, "T from eq22_BeattDMg not equal to test value")

class test_liq_ol_temp(unittest.TestCase):
    def test_19_T(self):
        self.assertAlmostEqual(pt.calculate_ol_liq_temp(liq_comps=LiqT, ol_comps=OlT,
        equationT="T_Put2008_eq19", P=5, H2O_Liq=0).T_K_calc[0], 1370.742988,
        decimalPlace, "T from eq19 not equal to test value")

    def test_19_Kd(self):
        self.assertAlmostEqual(pt.calculate_ol_liq_temp(liq_comps=LiqT, ol_comps=OlT,
        equationT="T_Put2008_eq19", P=5, H2O_Liq=0).get("Kd (Fe-Mg) Meas")[0], 0.277645,
        decimalPlace, "T from eq19 not equal to test value")

class test_liq_ol_eq(unittest.TestCase):
    def test_Roedder(self):
        self.assertAlmostEqual(pt.calculate_eq_ol_content(liq_comps=LiqT,
        ol_comps=OlT, Kd_model="Roeder1970", Fe3Fet_Liq=0.2).get("Eq Fo (Roeder, Kd=0.3)")[0], 0.8631287764106546,
        decimalPlace, "Eq Ol from Roedder not equal to test value")

    def test_Matzen(self):
        self.assertAlmostEqual(pt.calculate_eq_ol_content(liq_comps=LiqT,
        ol_comps=OlT, Kd_model="Matzen2011", Fe3Fet_Liq=0.2).get("Eq Fo (Matzen, Kd=0.352)")[0], 0.843126,
        decimalPlace, "Eq Ol from Roedder not equal to test value")

    def test_Toplis_Fo(self):
        self.assertAlmostEqual(pt.calculate_eq_ol_content(liq_comps=LiqT,
        ol_comps=OlT, Kd_model="Toplis2005", Fe3Fet_Liq=0.2, P=5, T=1300).get("Eq Fo (Toplis, input Fo)")[0], 0.863482,
        decimalPlace, "Eq Ol from Toplis not equal to test value")

    def test_Toplis_Kd(self):
        self.assertAlmostEqual(pt.calculate_eq_ol_content(liq_comps=LiqT,
        ol_comps=OlT, Kd_model="Toplis2005", Fe3Fet_Liq=0.2, P=5, T=1300).get("Kd (Toplis, input Fo)")[0], 0.299103,
        decimalPlace, "Calculated Kd from Roedder not equal to test value")

    def test_Toplis_Kd_Iter(self):
        self.assertAlmostEqual(pt.calculate_eq_ol_content(liq_comps=LiqT,
        Kd_model="Toplis2005", Fe3Fet_Liq=0.2, P=5, T=1300).get("Kd (Toplis, Iter)")[0], 0.300625,
        decimalPlace, "Calculated Kd from Roedder not equal to test value")

    def test_Toplis_Fo_Iter(self):
        self.assertAlmostEqual(pt.calculate_eq_ol_content(liq_comps=LiqT,
        Kd_model="Toplis2005", Fe3Fet_Liq=0.2, P=5, T=1300).get("Eq Fo (Toplis, Iter)")[0], 0.862883,
        decimalPlace, "Calculated Kd from Toplis Iter not equal to test value")

if __name__ == '__main__':
     unittest.main()