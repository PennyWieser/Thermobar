import contextlib
import io
import unittest

import numpy as np
import pandas as pd

import Thermobar as pt
from Unit_tests.test_liq_ol_sp import LiqT, OlT


def _matching_inputs():
    liquids = pd.concat([LiqT.copy(), LiqT.copy()], ignore_index=True)
    liquids['NiO_Liq'] = [0.04, 0.08]
    for column, value in {'SiO2_Liq': 54, 'Al2O3_Liq': 18,
                          'FeOt_Liq': 6.4, 'MgO_Liq': 6.2,
                          'Na2O_Liq': 3.9, 'K2O_Liq': 1.0,
                          'H2O_Liq': 2.0, 'Fe3Fet_Liq': 0.2}.items():
        liquids.loc[1, column] = value
    liquids['Sample_ID_liq'] = ['melt-A', 'melt-B']
    liquids['Notes_Liq'] = ['first liquid', 'second liquid']
    liquids.index = [11, 27]

    olivines = pd.concat([OlT.copy()] * 3, ignore_index=True)
    olivines['NiO_Ol'] = [0.25, 0.3, 0.4]
    for column, values in {'SiO2_Ol': [40.2, 40.7],
                           'FeOt_Ol': [15.1, 9.0],
                           'MgO_Ol': [44.7, 50.0]}.items():
        olivines.loc[[1, 2], column] = values
    olivines['Sample_ID_Ol'] = ['ol-A', 'ol-B', 'ol-C']
    olivines.index = [101, 203, 307]
    return liquids, olivines


def _matching_call(liquids, olivines, eq_tests, equation='T_Put2008_eq22',
                   **overrides):
    with contextlib.redirect_stdout(io.StringIO()):
        return pt.calculate_ol_liq_temp_matching(
            liq_comps=liquids, ol_comps=olivines, equationT=equation,
            P=1.0, eq_tests=eq_tests, **overrides)


def _override_cases():
    return [
        ('none', {'H2O_Liq': None, 'Fe3Fet_Liq': None}),
        ('zero_water', {'H2O_Liq': 0}),
        ('zero_ferric', {'Fe3Fet_Liq': 0}),
        ('ferric', {'Fe3Fet_Liq': 0.5}),
        ('both_scalars', {'H2O_Liq': 3.25, 'Fe3Fet_Liq': 0.35}),
        ('aligned_series', {
            'H2O_Liq': pd.Series([0.25, 4.0], index=[11, 27]),
            'Fe3Fet_Liq': pd.Series([0.2, 0.5], index=[11, 27])}),
        ('reordered_series', {
            'H2O_Liq': pd.Series([3.0, 0.0], index=[27, 11]),
            'Fe3Fet_Liq': pd.Series([0.5, 0.0], index=[27, 11])}),
    ]


class _OverrideAssertions(unittest.TestCase):
    def setUp(self):
        self.liquids, self.olivines = _matching_inputs()

    def assert_override_equivalence(self, overrides, eq_tests,
                                    equation='T_Put2008_eq22'):
        liquids_before = self.liquids.copy(deep=True)
        olivines_before = self.olivines.copy(deep=True)
        series_before = {key: value.copy(deep=True)
                         for key, value in overrides.items()
                         if isinstance(value, pd.Series)}
        explicit = self.liquids.copy(deep=True)
        for key, value in overrides.items():
            if value is not None:
                explicit[key] = value

        expected = _matching_call(explicit, self.olivines, eq_tests,
                                  equation)
        actual = _matching_call(self.liquids, self.olivines, eq_tests,
                                equation, **overrides)
        pd.testing.assert_frame_equal(self.liquids, liquids_before)
        pd.testing.assert_frame_equal(self.olivines, olivines_before)
        for key, value in series_before.items():
            pd.testing.assert_series_equal(overrides[key], value)
        self.assertEqual(len(actual), 6)
        self.assertTrue(actual.columns.is_unique)
        self.assertEqual(actual['Sample_ID_liq'].tolist(),
                         ['melt-A', 'melt-B'] * 3)
        self.assertEqual(actual['Sample_ID_Ol'].tolist(),
                         ['ol-A', 'ol-A', 'ol-B', 'ol-B', 'ol-C', 'ol-C'])
        self.assertEqual(actual['ID_liq'].tolist(), [11.0, 27.0] * 3)
        self.assertEqual(actual['ID_Ol'].tolist(),
                         [101, 101, 203, 203, 307, 307])
        self.assertEqual(actual['Notes_Liq'].tolist(),
                         ['first liquid', 'second liquid'] * 3)

        pd.testing.assert_frame_equal(actual, expected, check_exact=False,
                                      rtol=1e-12, atol=1e-9)
        for column in ('H2O_Liq', 'Fe3Fet_Liq'):
            np.testing.assert_allclose(
                actual[column].to_numpy(dtype=float),
                explicit[column].to_numpy(dtype=float).tolist() * 3,
                rtol=0, atol=1e-12, equal_nan=True)

        if equation not in ('T_Pu2017', 'T_Pu2021'):
            # Independent component arithmetic with the declared masses.
            mg = explicit['MgO_Liq'].to_numpy(dtype=float) / 40.3044
            fe = (explicit['FeOt_Liq'].to_numpy(dtype=float) * (
                1 - explicit['Fe3Fet_Liq'].to_numpy(dtype=float)) / 71.844)
            expected_mg_number = mg / (mg + fe)
            np.testing.assert_allclose(
                actual['Mg_Number_Liq_Fe3'].to_numpy(dtype=float),
                expected_mg_number.tolist() * 3, rtol=0, atol=1e-12,
                equal_nan=True)
        return actual, expected, explicit


class TestOlivineLiquidMatchingOverrides(_OverrideAssertions):
    def check_case(self, name):
        overrides = dict(_override_cases())[name]
        for eq_tests in (False, True):
            with self.subTest(eq_tests=eq_tests):
                self.assert_override_equivalence(overrides, eq_tests)

    def test_none_preserves_input_columns(self):
        self.check_case('none')

    def test_zero_water(self):
        self.check_case('zero_water')

    def test_zero_ferric_fraction(self):
        self.check_case('zero_ferric')

    def test_scalar_ferric_fraction(self):
        self.check_case('ferric')

    def test_both_scalar_overrides(self):
        self.check_case('both_scalars')

    def test_series_on_original_liquid_labels(self):
        self.check_case('aligned_series')

    def test_reordered_series_with_zero_values(self):
        self.check_case('reordered_series')

    def test_extra_series_labels_are_ignored(self):
        overrides = {
            'H2O_Liq': pd.Series([3.0, 999.0, 0.0], index=[27, 999, 11]),
            'Fe3Fet_Liq': pd.Series([0.5, 0.8, 0.0], index=[27, 999, 11]),
        }
        for eq_tests in (False, True):
            with self.subTest(eq_tests=eq_tests):
                self.assert_override_equivalence(overrides, eq_tests)

    def test_missing_series_labels_keep_pandas_nan_semantics(self):
        overrides = {'H2O_Liq': pd.Series([3.0], index=[27]),
                     'Fe3Fet_Liq': pd.Series([0.5], index=[27])}
        for eq_tests in (False, True):
            with self.subTest(eq_tests=eq_tests):
                self.assert_override_equivalence(overrides, eq_tests)

    def test_overrides_supply_missing_input_columns(self):
        self.liquids = self.liquids.drop(columns=['H2O_Liq', 'Fe3Fet_Liq'])
        for eq_tests in (False, True):
            with self.subTest(eq_tests=eq_tests):
                self.assert_override_equivalence(
                    {'H2O_Liq': 0.0, 'Fe3Fet_Liq': 0.5}, eq_tests)

    def test_ni_preparation_respects_overrides(self):
        for equation in ('T_Pu2017', 'T_Pu2021'):
            for eq_tests in (False, True):
                with self.subTest(equation=equation, eq_tests=eq_tests):
                    self.assert_override_equivalence(
                        {'H2O_Liq': 0.0, 'Fe3Fet_Liq': 0.5}, eq_tests,
                        equation=equation)
