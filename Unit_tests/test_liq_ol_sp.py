import unittest
import pandas as pd
import Thermobar as pt
import numpy as np
from unittest.mock import patch

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
class test_liq_visc(unittest.TestCase):

    def Giordano2008_B(self):
        self.assertAlmostEqual(pt.calculate_viscosity_giordano_2008(liq_comps=LiqT, T=1400).
        get("B")[0], 5808.444491,
        decimalPlace, "Coefficient B not equal to test value")
    def Giordano2008_C(self):
        self.assertAlmostEqual(pt.calculate_viscosity_giordano_2008(liq_comps=LiqT, T=1400).
        get("C")[0], 346.48721,
        decimalPlace, "Coefficient C not equal to test value")
    def Giordano2008_n(self):
        self.assertAlmostEqual(pt.calculate_viscosity_giordano_2008(liq_comps=LiqT, T=1400).
        get("n_melt")[0], 9.19193,
        decimalPlace, "viscosity not equal to test value")

class test_liq_only_temp(unittest.TestCase):


    def test_19_Kd(self):
        self.assertAlmostEqual(pt.calculate_ol_liq_temp(liq_comps=LiqT, ol_comps=OlT,
        equationT="T_Put2008_eq19", P=5, H2O_Liq=0).get("Kd (Fe-Mg) Meas")[0], 0.24679563729002468,
        decimalPlace, "T from eq19 not equal to test value")

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
        equationT="T_Put2008_eq19", P=5, H2O_Liq=0).T_K_calc[0], 1395.6477246471757,
        decimalPlace, "T from eq19 not equal to test value")


    def test_19_Kd_Fe3(self):
        self.assertAlmostEqual(pt.calculate_ol_liq_temp(liq_comps=LiqT, ol_comps=OlT,
        equationT="T_Put2008_eq19", P=5, H2O_Liq=0, Fe3Fet_Liq=0.5).get("Kd (Fe-Mg) Meas")[0], 0.4442321471220445,
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

class test_ol_sp_temp(unittest.TestCase):
    def test_Wan2008(self):
        self.assertAlmostEqual(pt.calculate_ol_sp_temp(ol_comps=OlT, sp_comps=SpT, equationT="T_Wan2008")[0], 1552.345914,
        decimalPlace, "Calculated T from Wan2008 not equal to test value ")

    def test_Coogan2014(self):
        self.assertAlmostEqual(pt.calculate_ol_sp_temp(ol_comps=OlT, sp_comps=SpT, equationT="T_Coogan2014")[0], 1555.821477,
        decimalPlace, "Calculated T from Coogan 2014 not equal to test value ")






MATCHING_RESULT_COLUMNS = {
    False: ['T_K_calc', 'Ol_Fo_Meas', 'Kd (Fe-Mg) Meas'],
    True: ['T_K_calc', 'Ol_Fo_Meas', 'Kd Meas', 'Kd calc (Toplis)',
           'ΔKd, Toplis (M-P)', 'ΔKd, Roeder (M-P)',
           'ΔKd, Matzen (M-P)', 'ΔKd, Shea (M-P)'],
}
MATCHING_INTERMEDIATES = ['DMg_Meas', 'CNML', 'CSiO2L', 'NF', 'Den_Beat93']


def _matching_test_inputs():
    """Asymmetric synthetic inputs retain the existing test compositions."""
    liquids = pd.concat([LiqT.copy(), LiqT.copy()], ignore_index=True)
    liquids['NiO_Liq'] = [0.04, 0.08]
    for column, value in {'SiO2_Liq': 54, 'Al2O3_Liq': 18, 'FeOt_Liq': 6.4,
                          'MgO_Liq': 6.2, 'Na2O_Liq': 3.9, 'K2O_Liq': 1,
                          'H2O_Liq': 2, 'Fe3Fet_Liq': 0.2}.items():
        liquids.loc[1, column] = value
    liquids['Sample_ID_liq'] = ['melt-A', 'melt-B']
    liquids.index = [11, 27]
    olivines = pd.concat([OlT.copy(), OlT.copy(), OlT.copy()], ignore_index=True)
    olivines['NiO_Ol'] = [0.25, 0.3, 0.4]
    for column, values in {'SiO2_Ol': [40.2, 40.7], 'FeOt_Ol': [15.1, 9.0],
                           'MgO_Ol': [44.7, 50.0]}.items():
        olivines.loc[[1, 2], column] = values
    olivines['Sample_ID_Ol'] = ['ol-A', 'ol-B', 'ol-C']
    olivines.index = [101, 203, 307]
    return liquids, olivines


def _matching_pair_oracle(liquids, olivines, equation, eq_tests):
    """Prepare explicit pairs with public helpers, independently of matching output.

    This checks the result/assembly contract, not scientific formula accuracy.
    The lower API's separate-composition path has different Cr/P preparation;
    use its prepared meltmatch path to compare the same inputs.
    """
    expected = []
    for ol_index, liq_index in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]:
        liquid = liquids.iloc[[liq_index]].reset_index(drop=True).copy()
        olivine = olivines.iloc[[ol_index]].reset_index(drop=True).copy()
        if equation in ['T_Pu2017', 'T_Pu2021']:
            liquid_components = pd.concat([
                pt.calculate_anhydrous_mol_fractions_liquid_Ni(liq_comps=liquid),
                liquid], axis=1)
            olivine_components = pd.concat([
                pt.calculate_mol_fractions_olivine_ni(ol_comps=olivine), olivine], axis=1)
        else:
            liquid_components = pt.calculate_anhydrous_cat_fractions_liquid(liq_comps=liquid)
            olivine_components = pd.concat([
                olivine, pt.calculate_cat_fractions_olivine(ol_comps=olivine)], axis=1)
        prepared_pair = pd.concat([liquid_components, olivine_components], axis=1)
        calculated = pt.calculate_ol_liq_temp(
            equationT=equation, meltmatch=prepared_pair, P=1.0, eq_tests=eq_tests)
        row = {column: float(calculated[column].iloc[0])
               for column in MATCHING_RESULT_COLUMNS[eq_tests] if column != 'Ol_Fo_Meas'}
        row['Ol_Fo_Meas'] = float(pt.calculate_ol_fo(ol_comps=olivine).iloc[0])
        for column in MATCHING_INTERMEDIATES:
            if column in prepared_pair:
                row[column] = float(prepared_pair[column].iloc[0])
        expected.append(row)
    return pd.DataFrame(expected)


class TestOlivineLiquidMatchingResults(unittest.TestCase):
    def setUp(self):
        self.liquids, self.olivines = _matching_test_inputs()

    def assert_matching_result(self, equation, eq_tests):
        liquids_before = self.liquids.copy(deep=True)
        olivines_before = self.olivines.copy(deep=True)
        expected = _matching_pair_oracle(self.liquids, self.olivines, equation, eq_tests)
        result = pt.calculate_ol_liq_temp_matching(
            liq_comps=self.liquids, ol_comps=self.olivines,
            equationT=equation, P=1.0, eq_tests=eq_tests)
        pd.testing.assert_frame_equal(self.liquids, liquids_before)
        pd.testing.assert_frame_equal(self.olivines, olivines_before)
        self.assertEqual(len(result), 6)
        self.assertEqual(result.index.tolist(), list(range(6)))
        self.assertTrue(result.columns.is_unique)
        self.assertIn('T_K_calc', result.columns)
        fields = MATCHING_RESULT_COLUMNS[eq_tests]
        self.assertEqual(result.columns[:len(fields)].tolist(), fields)
        for column in expected:
            self.assertIn(column, result.columns)
            np.testing.assert_allclose(result[column].astype(float), expected[column],
                                       rtol=1e-12, atol=1e-9)
        self.assertEqual(result['Sample_ID_Ol'].tolist(),
                         ['ol-A', 'ol-A', 'ol-B', 'ol-B', 'ol-C', 'ol-C'])
        self.assertEqual(result['Sample_ID_liq'].tolist(), ['melt-A', 'melt-B'] * 3)
        np.testing.assert_array_equal(result['ID_Ol'].astype(float),
                                      [101, 101, 203, 203, 307, 307])
        np.testing.assert_array_equal(result['ID_liq'].astype(float), [11, 27] * 3)
        # All source chemistry and labels survive, in the explicit pair order.
        for column in self.liquids:
            self.assertEqual(result[column].tolist(), self.liquids[column].tolist() * 3)
        for column in self.olivines:
            values = [value for value in self.olivines[column] for _ in range(2)]
            self.assertEqual(result[column].tolist(), values)
        return result, expected

    def test_beattie_values_pairs_and_post_calculation_intermediates(self):
        for eq_tests in [False, True]:
            with self.subTest(eq_tests=eq_tests):
                _, expected = self.assert_matching_result('T_Beatt93_ol', eq_tests)
                self.assertEqual(len(np.unique(expected['T_K_calc'])), 6)
                self.assertTrue(set(MATCHING_INTERMEDIATES).issubset(expected.columns))

    def test_hydrous_and_ni_preparation_modes(self):
        for equation in ['T_Put2008_eq22', 'T_Pu2017']:
            for eq_tests in [False, True]:
                with self.subTest(equation=equation, eq_tests=eq_tests):
                    self.assert_matching_result(equation, eq_tests)

    def test_pu2021_return_schema_preserves_existing_lower_api(self):
        # This parity check does not validate or change its pressure correction.
        for eq_tests in [False, True]:
            with self.subTest(eq_tests=eq_tests):
                self.assert_matching_result('T_Pu2021', eq_tests)

    def test_reserved_names_rejected_before_calculation(self):
        target = ('Thermobar.olivine_liquid_olivine_spinel_thermometry.'
                  'calculate_ol_liq_temp')
        for eq_tests in [False, True]:
            for table in ['liquids', 'olivines']:
                for column in MATCHING_RESULT_COLUMNS[eq_tests]:
                    with self.subTest(eq_tests=eq_tests, table=table, column=column):
                        liquids, olivines = _matching_test_inputs()
                        frame = liquids if table == 'liquids' else olivines
                        frame[column] = np.arange(len(frame), dtype=float) + 9998.0
                        liquids_before, olivines_before = liquids.copy(), olivines.copy()
                        with patch(target, wraps=pt.calculate_ol_liq_temp) as calculator:
                            with self.assertRaisesRegex(ValueError, 'calculated output') as error:
                                pt.calculate_ol_liq_temp_matching(
                                    liq_comps=liquids, ol_comps=olivines,
                                    equationT='T_Beatt93_ol', P=1.0, eq_tests=eq_tests)
                            self.assertEqual(calculator.call_count, 0)
                        self.assertIn(column, str(error.exception))
                        self.assertIn('rename', str(error.exception).lower())
                        pd.testing.assert_frame_equal(liquids, liquids_before)
                        pd.testing.assert_frame_equal(olivines, olivines_before)

    def test_collision_message_lists_names_from_both_inputs(self):
        for eq_tests in [False, True]:
            with self.subTest(eq_tests=eq_tests):
                liquids, olivines = _matching_test_inputs()
                liquids['T_K_calc'] = [9999.0, 9998.0]
                olivines['Ol_Fo_Meas'] = 123.0
                with self.assertRaises(ValueError) as error:
                    pt.calculate_ol_liq_temp_matching(
                        liq_comps=liquids, ol_comps=olivines,
                        equationT='T_Beatt93_ol', P=1.0, eq_tests=eq_tests)
                for name in ['T_K_calc', 'Ol_Fo_Meas']:
                    self.assertIn(name, str(error.exception))
                self.assertIn('rename', str(error.exception).lower())

    def test_only_names_calculated_in_selected_mode_are_reserved(self):
        for eq_tests, metadata in [(False, 'Kd Meas'), (True, 'Kd (Fe-Mg) Meas')]:
            with self.subTest(eq_tests=eq_tests):
                liquids, olivines = _matching_test_inputs()
                liquids[metadata] = 993.5
                result = pt.calculate_ol_liq_temp_matching(
                    liq_comps=liquids, ol_comps=olivines,
                    equationT='T_Beatt93_ol', P=1.0, eq_tests=eq_tests)
                self.assertEqual(result[metadata].tolist(), [993.5] * 6)


if __name__ == '__main__':
     unittest.main()
