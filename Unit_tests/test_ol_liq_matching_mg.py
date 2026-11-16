import unittest

import numpy as np
import pandas as pd

import Thermobar as pt


LIQUIDS = {
    "SiO2_Liq": [
        51.0,
        48.0
    ],
    "TiO2_Liq": [
        0.48,
        1.1
    ],
    "Al2O3_Liq": [
        19.0,
        16.0
    ],
    "FeOt_Liq": [
        5.3,
        8.2
    ],
    "MnO_Liq": [
        0.1,
        0.15
    ],
    "MgO_Liq": [
        4.5,
        8.5
    ],
    "CaO_Liq": [
        9.0,
        11.0
    ],
    "Na2O_Liq": [
        4.2,
        2.8
    ],
    "K2O_Liq": [
        0.1,
        0.6
    ],
    "Cr2O3_Liq": [
        0.11,
        0.07
    ],
    "P2O5_Liq": [
        0.11,
        0.18
    ],
    "NiO_Liq": [
        0.05,
        0.08
    ],
    "CoO_Liq": [
        0.01,
        0.02
    ],
    "H2O_Liq": [
        2.0,
        2.0
    ],
    "Fe3Fet_Liq": [
        0.1,
        0.1
    ],
    "Sample_ID_Liq": [
        1001.0,
        1002.0
    ],
    "Sample_ID_liq": [
        2001.0,
        2002.0
    ]
}

OLIVINES = {
    "SiO2_Ol": [
        40.5,
        41.0,
        39.0
    ],
    "TiO2_Ol": [
        0.02,
        0.01,
        0.03
    ],
    "Al2O3_Ol": [
        0.08,
        0.06,
        0.1
    ],
    "FeOt_Ol": [
        12.4,
        10.0,
        15.5
    ],
    "MnO_Ol": [
        0.17,
        0.12,
        0.2
    ],
    "MgO_Ol": [
        47.4,
        49.0,
        44.0
    ],
    "CaO_Ol": [
        0.3,
        0.25,
        0.28
    ],
    "Na2O_Ol": [
        0.0,
        0.0,
        0.0
    ],
    "K2O_Ol": [
        0.0,
        0.0,
        0.0
    ],
    "Cr2O3_Ol": [
        0.03,
        0.04,
        0.05
    ],
    "P2O5_Ol": [
        0.01,
        0.02,
        0.015
    ],
    "NiO_Ol": [
        0.35,
        0.25,
        0.45
    ],
    "Sample_ID_Ol": [
        3001.0,
        3002.0,
        3003.0
    ]
}

NI_COMPONENT_COLUMNS = [
    "SiO2_Liq_mol_frac",
    "MgO_Liq_mol_frac",
    "MnO_Liq_mol_frac",
    "FeOt_Liq_mol_frac",
    "CaO_Liq_mol_frac",
    "Al2O3_Liq_mol_frac",
    "Na2O_Liq_mol_frac",
    "K2O_Liq_mol_frac",
    "TiO2_Liq_mol_frac",
    "P2O5_Liq_mol_frac",
    "Cr2O3_Liq_mol_frac",
    "NiO_Liq_mol_frac",
    "CoO_Liq_mol_frac",
    "SiO2_Liq",
    "TiO2_Liq",
    "Al2O3_Liq",
    "FeOt_Liq",
    "MnO_Liq",
    "MgO_Liq",
    "CaO_Liq",
    "Na2O_Liq",
    "K2O_Liq",
    "Cr2O3_Liq",
    "P2O5_Liq",
    "NiO_Liq",
    "CoO_Liq",
    "H2O_Liq",
    "Fe3Fet_Liq",
    "Sample_ID_Liq",
    "Sample_ID_liq",
    "ID_liq",
    "SiO2_Ol_mol_frac",
    "MgO_Ol_mol_frac",
    "FeOt_Ol_mol_frac",
    "CaO_Ol_mol_frac",
    "Al2O3_Ol_mol_frac",
    "Na2O_Ol_mol_frac",
    "K2O_Ol_mol_frac",
    "MnO_Ol_mol_frac",
    "TiO2_Ol_mol_frac",
    "NiO_Ol_mol_frac",
    "SiO2_Ol",
    "TiO2_Ol",
    "Al2O3_Ol",
    "FeOt_Ol",
    "MnO_Ol",
    "MgO_Ol",
    "CaO_Ol",
    "Na2O_Ol",
    "K2O_Ol",
    "Cr2O3_Ol",
    "P2O5_Ol",
    "NiO_Ol",
    "Sample_ID_Ol",
    "ID_Ol"
]

BEATTIE_COMPONENT_COLUMNS = [
    "SiO2_Liq",
    "TiO2_Liq",
    "Al2O3_Liq",
    "FeOt_Liq",
    "MnO_Liq",
    "MgO_Liq",
    "CaO_Liq",
    "Na2O_Liq",
    "K2O_Liq",
    "Cr2O3_Liq",
    "P2O5_Liq",
    "NiO_Liq",
    "CoO_Liq",
    "H2O_Liq",
    "Fe3Fet_Liq",
    "Sample_ID_Liq",
    "Sample_ID_liq",
    "ID_liq",
    "SiO2_Liq_mol_frac",
    "MgO_Liq_mol_frac",
    "MnO_Liq_mol_frac",
    "FeOt_Liq_mol_frac",
    "CaO_Liq_mol_frac",
    "Al2O3_Liq_mol_frac",
    "Na2O_Liq_mol_frac",
    "K2O_Liq_mol_frac",
    "TiO2_Liq_mol_frac",
    "P2O5_Liq_mol_frac",
    "Cr2O3_Liq_mol_frac",
    "Si_Liq_cat_frac",
    "Mg_Liq_cat_frac",
    "Mn_Liq_cat_frac",
    "Fet_Liq_cat_frac",
    "Ca_Liq_cat_frac",
    "Al_Liq_cat_frac",
    "Na_Liq_cat_frac",
    "K_Liq_cat_frac",
    "Ti_Liq_cat_frac",
    "P_Liq_cat_frac",
    "Cr_Liq_cat_frac",
    "Mg_Number_Liq_NoFe3",
    "Mg_Number_Liq_Fe3",
    "SiO2_Ol",
    "TiO2_Ol",
    "Al2O3_Ol",
    "FeOt_Ol",
    "MnO_Ol",
    "MgO_Ol",
    "CaO_Ol",
    "Na2O_Ol",
    "K2O_Ol",
    "Cr2O3_Ol",
    "P2O5_Ol",
    "NiO_Ol",
    "Sample_ID_Ol",
    "ID_Ol",
    "Si_Ol_cat_frac",
    "Mg_Ol_cat_frac",
    "Fet_Ol_cat_frac",
    "Ca_Ol_cat_frac",
    "Al_Ol_cat_frac",
    "Na_Ol_cat_frac",
    "K_Ol_cat_frac",
    "Mn_Ol_cat_frac",
    "Ti_Ol_cat_frac",
    "DMg_Meas",
    "CNML",
    "CSiO2L",
    "NF",
    "Den_Beat93"
]

EQ_METRICS = [
    "T_K_calc",
    "Kd Meas",
    "Kd calc (Toplis)",
    "ΔKd, Toplis (M-P)",
    "ΔKd, Roeder (M-P)",
    "ΔKd, Matzen (M-P)",
    "ΔKd, Shea (M-P)"
]

EXPECTED_FO = [
    0.8720226623296171,
    0.8720226623296171,
    0.8972717540908396,
    0.8972717540908396,
    0.8349861967742973,
    0.8349861967742973
]

EXPECTED_KD = [
    0.24679563729002468,
    0.3013047008107212,
    0.19252984212552945,
    0.23505337094187267,
    0.3323327615780446,
    0.4057341709780734
]


class TestOlivineMgMatching(unittest.TestCase):
    def compositions(self):
        liquids = pd.DataFrame(LIQUIDS, index=[101, 303], dtype="float64")
        liquids.index.name = "liquid_rows"
        olivines = pd.DataFrame(OLIVINES, index=[7, 13, 29], dtype="float64")
        olivines.index.name = "olivine_rows"
        return liquids, olivines

    def assert_frame(self, actual, expected):
        pd.testing.assert_frame_equal(
            actual, expected, check_exact=False, rtol=1e-12, atol=1e-12,
        )

    def direct_reference(self, equation, eq_tests, liquids, olivines):
        before_liquids = liquids.copy(deep=True)
        before_olivines = olivines.copy(deep=True)
        try:
            result = pt.calculate_ol_liq_temp(
                equationT=equation, liq_comps=liquids, ol_comps=olivines,
                P=5.0, H2O_Liq=2.0, Fe3Fet_Liq=0.1, eq_tests=eq_tests,
            )
        finally:
            pd.testing.assert_frame_equal(liquids, before_liquids, check_exact=True)
            pd.testing.assert_frame_equal(olivines, before_olivines, check_exact=True)
        # Compare the independently enumerated pairs by position. Do not reset
        # the matching result or change either external input frame.
        return result.reset_index(drop=True)

    def assert_matching(self, equation, eq_tests, *, comps=None, expected_nan_rows=()):
        liquids, olivines = self.compositions() if comps is None else comps
        before_liquids = liquids.copy(deep=True)
        before_olivines = olivines.copy(deep=True)
        # Enumerate the Cartesian product independently of matching preparation.
        paired_liquids = pd.concat(
            [liquids.iloc[[j]] for i in range(3) for j in range(2)], ignore_index=True,
        )
        paired_olivines = pd.concat(
            [olivines.iloc[[i]] for i in range(3) for j in range(2)], ignore_index=True,
        )
        full_reference = self.direct_reference(
            equation, True, paired_liquids, paired_olivines,
        )
        self.assertEqual(list(full_reference.columns[:7]), EQ_METRICS)
        reference = full_reference if eq_tests else self.direct_reference(
            equation, False, paired_liquids, paired_olivines,
        )
        try:
            actual = pt.calculate_ol_liq_temp_matching(
                equationT=equation, liq_comps=liquids, ol_comps=olivines,
                P=5.0, H2O_Liq=2.0, Fe3Fet_Liq=0.1, eq_tests=eq_tests,
            )
        finally:
            pd.testing.assert_frame_equal(liquids, before_liquids, check_exact=True)
            pd.testing.assert_frame_equal(olivines, before_olivines, check_exact=True)

        self.assertIs(type(actual), pd.DataFrame)
        pd.testing.assert_index_equal(actual.index, pd.RangeIndex(6), exact=True)
        self.assertIsNone(actual.columns.name)
        self.assertTrue(actual.columns.is_unique)
        components = (
            BEATTIE_COMPONENT_COLUMNS if equation == "T_Beatt93_ol" else NI_COMPONENT_COLUMNS
        )
        metrics = (
            [EQ_METRICS[0], "Ol_Fo_Meas", *EQ_METRICS[1:]] if eq_tests
            else ["T_K_calc", "Ol_Fo_Meas", "Kd (Fe-Mg) Meas"]
        )
        self.assertEqual(list(actual.columns), metrics + components)
        self.assertEqual(actual.dtypes.astype(str).tolist(), ["float64"] * len(actual.columns))
        self.assert_frame(actual.loc[:, reference.columns], reference)
        # The direct true-mode result supplies every prepared component, including
        # those returned by matching with eq_tests=False, rather than only T/Kd.
        reference_components = full_reference.iloc[:, 7:]
        self.assert_frame(actual.loc[:, reference_components.columns], reference_components)
        for name, values in [
            ("ID_Ol", [7.0, 7.0, 13.0, 13.0, 29.0, 29.0]),
            ("ID_liq", [101.0, 303.0, 101.0, 303.0, 101.0, 303.0]),
            ("Ol_Fo_Meas", EXPECTED_FO),
            ("Kd Meas" if eq_tests else "Kd (Fe-Mg) Meas", EXPECTED_KD),
        ]:
            pd.testing.assert_series_equal(
                actual[name], pd.Series(values, name=name, dtype="float64"),
                check_exact=False, rtol=1e-12, atol=1e-12,
            )
        mask = [i in expected_nan_rows for i in range(6)]
        self.assertEqual(actual["T_K_calc"].isna().tolist(), mask)
        self.assertEqual(np.isfinite(actual["T_K_calc"]).tolist(), [not x for x in mask])
        return actual

    def test_mg_matching_without_equilibrium_tests(self):
        self.assert_matching("T_Pu2017_Mg", False)

    def test_mg_matching_with_equilibrium_tests(self):
        self.assert_matching("T_Pu2017_Mg", True)

    def test_beattie_matching_without_equilibrium_tests(self):
        self.assert_matching("T_Beatt93_ol", False)

    def test_beattie_matching_with_equilibrium_tests(self):
        self.assert_matching("T_Beatt93_ol", True)

    def test_ni_2017_matching_without_equilibrium_tests(self):
        self.assert_matching("T_Pu2017", False)

    def test_ni_2017_matching_with_equilibrium_tests(self):
        self.assert_matching("T_Pu2017", True)

    def test_ni_2021_matching_without_equilibrium_tests(self):
        self.assert_matching("T_Pu2021", False)

    def test_ni_2021_matching_with_equilibrium_tests(self):
        self.assert_matching("T_Pu2021", True)

    def test_existing_ni_zero_negative_nan_policy(self):
        for side in ("liquid", "olivine"):
            for label, value in (("zero", 0.0), ("negative", -0.02), ("nan", float("nan"))):
                for eq_tests in (False, True):
                    with self.subTest(side=side, ni=label, eq_tests=eq_tests):
                        liquids, olivines = self.compositions()
                        if side == "liquid":
                            liquids.loc[101, "NiO_Liq"] = value
                            nan_rows = (0, 2, 4)
                        else:
                            olivines.loc[13, "NiO_Ol"] = value
                            nan_rows = (2, 3)
                        self.assert_matching(
                            "T_Pu2017", eq_tests, comps=(liquids, olivines),
                            expected_nan_rows=nan_rows,
                        )
