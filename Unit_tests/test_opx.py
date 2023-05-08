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


import warnings as w
with w.catch_warnings():
    w.simplefilter('ignore')

    class test_opx_only_press(unittest.TestCase):
        def test_press_29c(self):
            self.assertAlmostEqual(pt.calculate_opx_only_press(opx_comps=OpxT, equationP="P_Put2008_eq29c",
            T=1300)[0], 0.631893, decimalPlace, "equation 29c not equal to test value")


    class test_opx_liq_temp(unittest.TestCase):
        def test_put29a(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_temp(opx_comps=OpxT,
        liq_comps=LiqT, equationT="T_Put2008_eq28b_opx_sat", P=5)[0], 1344.331255,
        decimalPlace,
        "Calc T from equation 29a not equal to test value")


    class test_opx_liq_press(unittest.TestCase):
        def test_29b(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put2008_eq29b", T=800)[0], -21.129189,
        decimalPlace,
        "Calc P from  eq29c not equal to test value")

        def test_29b_noH(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put2008_eq29b", T=800, H2O_Liq=0)[0], -25.049189,
        decimalPlace,
        "Calc P from  eq32c h2O=0 not equal to test value")

        def test_Global_noH(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put_Global_Opx",)[0], 1.810153,
        decimalPlace,
        "Calc P from  Putirka Global not equal to test value")


        def test_Global_Kdcalc2(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put_Global_Opx", T=1300, eq_tests=True, Fe3Fet_Liq=0.6)
        .eq_tests_Kd_Fe_Mg_Fe2[0], 0.781298,
        decimalPlace,
        "Calc Kd Fe2 not equal to test value")

        def test_Global_Kdcalc2_60Fe3(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put_Global_Opx", T=1300, eq_tests=True)
        .eq_tests_Kd_Fe_Mg_Fe2[0], 0.347244,
        decimalPlace,
        "Calc Kd Fe2 not equal to test value")

        def test_Global_KdcalcT(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put_Global_Opx", T=1300, eq_tests=True)
        .eq_tests_Kd_Fe_Mg_Fet[0], 0.312519,
        decimalPlace,
        "Calc Kd Fet not equal to test value")


        def test_Global_KdcalcT2(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put_Global_Opx", T=1300, eq_tests=True)
        .get("Kd Eq (Put2008+-0.06)")[0], "Y",
        decimalPlace,
        "Calc eq test not equal to test value")




    class test_opx_liq_press_temp(unittest.TestCase):
        def test_global_eq28a_press(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press_temp(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put_Global_Opx", equationT="T_Put2008_eq28a").P_kbar_calc[0],
            1.810153, decimalPlace,
        "Calc P from iterating eq28a-Global Opx not equal to test value")

        def test_global_eq28a_temp(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press_temp(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put_Global_Opx", equationT="T_Put2008_eq28a").T_K_calc[0],
            1305.385773, decimalPlace,
        "Calc T from iterating eq28a-Global Opx not equal to test value")

        def test_29a_eq28a_press(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press_temp(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put2008_eq29a", equationT="T_Put2008_eq28a").P_kbar_calc[0],
            2.568279, decimalPlace,
        "Calc P from iterating eq28a-Global Opx not equal to test value")

        def test_29a_eq28a_temp(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press_temp(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put2008_eq29a", equationT="T_Put2008_eq28a").T_K_calc[0],
            1308.050269, decimalPlace,
        "Calc T from iterating eq28a-Global Opx not equal to test value")

        def test_29a_eq28a_press_noH(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press_temp(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put2008_eq29a", equationT="T_Put2008_eq28a", H2O_Liq=0).P_kbar_calc[0],
            -0.593182, decimalPlace,
        "Calc P from iterating eq28a-29a not equal to test value")

        def test_29a_eq28a_temp_noH(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press_temp(opx_comps=OpxT,
        liq_comps=LiqT, equationP="P_Put2008_eq29a", equationT="T_Put2008_eq28a", H2O_Liq=0).T_K_calc[0],
            1364.045732, decimalPlace,
        "Calc T from iterating eq28a-29a not equal to test value")



    Liq2=pd.DataFrame(data={"Sample_ID_Liq":"test",
                                "SiO2_Liq": 51.1,
                                "TiO2_Liq": 0.93,
                                "Al2O3_Liq": 17.5,
                                "FeOt_Liq": 8.91,
                                "MnO_Liq": 0.18,
                                "MgO_Liq": 6.09,
                                "CaO_Liq": 11.50,
                                "Na2O_Liq": 3.53,
                                "K2O_Liq": 0.17,
                                "Cr2O3_Liq": 0,
                                "P2O5_Liq": 0.15,
                                "H2O_Liq": 3.8,
                                "Fe3Fet_Liq":0}, index=[0])

    Opx2=pd.DataFrame(data={"Sample_ID_Opx":"test",
                            "SiO2_Opx": 55.00,
                                "TiO2_Opx": 0.34,
                                "Al2O3_Opx": 1.50,
                                "FeOt_Opx": 11.30,
                                "MnO_Opx": 0.24,
                                "MgO_Opx": 30.70,
                                "CaO_Opx": 0.90,
                                "Na2O_Opx": 0.01,
                                "K2O_Opx": 0,
                                "Cr2O3_Opx": 0.19}, index=[0])


    class test_opx_liq_melt_matching(unittest.TestCase):
        def test_eq29a_28a_press(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press_temp_matching(liq_comps=Liq2,
        opx_comps=Opx2, equationP="P_Put2008_eq29a",
            equationT="T_Put2008_eq28a").get("Av_PTs").Mean_P_kbar_calc[0], 3.327589, decimalPlace,
    "Calc P from melt matching 29a-28a not equal to test value")
    #
        def test_eq29a_28a_press2(self):
            self.assertAlmostEqual(pt.calculate_opx_liq_press_temp_matching(liq_comps=Liq2,
        opx_comps=Opx2, equationP="P_Put2008_eq29a",
            equationT="T_Put2008_eq28a").get("Av_PTs").Mean_T_K_calc[0], 1384.287957, decimalPlace,
    "Calc T from melt matching 29a-28a not equal to test value")


    class test_opx_equilibirum(unittest.TestCase):
        def test_rhodes_lines_opx_23(self):
            self.assertAlmostEqual(pt.calculate_opx_rhodes_diagram_lines(Min_Mgno=0.4,
        Max_Mgno=0.7, liq_comps=Liq2).get("Eq_Opx_Mg# (Kd=0.23)")[0],
        0.743494, decimalPlace,
        "Calc Mg# from Kd=0.23 not equal to test value")

        def test_rhodes_lines_opx_Kd(self):
            self.assertAlmostEqual(pt.calculate_opx_rhodes_diagram_lines(Min_Mgno=0.4,
        Max_Mgno=0.7, liq_comps=Liq2).get("Kd_XSi_P2008")[0],
        0.304877, decimalPlace,
        "Calc Kd from Putirka XSi not equal to test value")



if __name__ == '__main__':
    unittest.main()