"""Liquid labels must survive the Cartesian mineral/liquid matching wrappers."""

import numpy as np
import pandas as pd
import pytest

import Thermobar as pt


@pytest.fixture
def compositions():
    liq = dict(SiO2_Liq=51.0, TiO2_Liq=0.48, Al2O3_Liq=19.0,
               FeOt_Liq=5.3, MnO_Liq=0.1, MgO_Liq=4.5, CaO_Liq=9.0,
               Na2O_Liq=4.2, K2O_Liq=0.1, Cr2O3_Liq=0.11,
               P2O5_Liq=0.11, H2O_Liq=5.0, Fe3Fet_Liq=0.1, NiO_Liq=0.015)
    ol = dict(SiO2_Ol=40.5, TiO2_Ol=0.02, Al2O3_Ol=0.08, FeOt_Ol=12.4,
              MnO_Ol=0.17, MgO_Ol=47.4, CaO_Ol=0.30, Na2O_Ol=0.0,
              K2O_Ol=0.0, Cr2O3_Ol=0.03, NiO_Ol=0.35)
    plag = dict(SiO2_Plag=57.3, TiO2_Plag=0.09, Al2O3_Plag=26.6,
                FeOt_Plag=0.43, MnO_Plag=0.0, MgO_Plag=0.03,
                CaO_Plag=8.83, Na2O_Plag=6.11, K2O_Plag=0.49,
                Cr2O3_Plag=0.0)
    kspar = dict(SiO2_Kspar=65.5, TiO2_Kspar=0.0, Al2O3_Kspar=19.6,
                 FeOt_Kspar=0.07, MnO_Kspar=0.0, MgO_Kspar=0.0,
                 CaO_Kspar=0.75, Na2O_Kspar=4.81, K2O_Kspar=9.36,
                 Cr2O3_Kspar=0.0)
    frames = {
        "liq": pd.DataFrame([liq, dict(liq, MgO_Liq=5.2, FeOt_Liq=5.7,
                                       NiO_Liq=0.018, H2O_Liq=4.2)], index=[11, 29]),
        "ol": pd.DataFrame([ol, dict(ol, FeOt_Ol=11.9, MgO_Ol=47.9,
                                    NiO_Ol=0.32)], index=[11, 29]),
        "plag": pd.DataFrame([plag, dict(plag, SiO2_Plag=57.9,
                                        TiO2_Plag=1.09)], index=[11, 29]),
        "kspar": pd.DataFrame([kspar, dict(kspar, K2O_Kspar=9.1,
                                          Na2O_Kspar=5.0)], index=[11, 29]),
    }
    frames["liq"]["User_Trace"] = [7.25, 9.5]
    for phase, suffix in [("ol", "Ol"), ("plag", "Plag"), ("kspar", "Kspar")]:
        frames[phase]["Sample_ID_" + suffix] = [phase + "-A", phase + "-B"]
    return frames


# Ordinary, Ni-based, equilibrium and both supported feldspar branches.
MODES = [
    ("ol_temp", "T_Beatt93_ol", False),
    ("ol_temp", "T_Beatt93_ol", True),
    ("ol_temp", "T_Pu2017", False),
    ("ol_temp", "T_Pu2017", True),
    ("ol_temp", "T_Pu2021", False),
    ("ol_temp", "T_Pu2021", True),
    ("ol_hygr", "H_Gavr2016", False),
    ("ol_hygr", "H_Gavr2016", True),
    ("ol_hygr", "H_Pu2017", False),
    ("ol_hygr", "H_Pu2017", True),
    ("plag_temp", "T_Put2008_eq24a", False),
    ("kspar_temp", "T_Put2008_eq24b", False),
    ("plag_temp_hygr", "T_Put2008_eq23", False),
]
BASIC_MODES = [MODES[0], MODES[6], MODES[10], MODES[12]]
CASES = [(mode, "canonical") for mode in MODES]
CASES += [(mode, state) for state in ("absent", "lower-only", "both", "zero-null")
          for mode in BASIC_MODES]


def match(mode, frames):
    api, equation, equilibrium = mode
    liq = frames["liq"]
    if api == "ol_temp":
        result = pt.calculate_ol_liq_temp_matching(
            liq_comps=liq, ol_comps=frames["ol"], equationT=equation,
            P=15 if equation == "T_Pu2021" else 5, eq_tests=equilibrium)
    elif api == "ol_hygr":
        result = pt.calculate_ol_liq_hygr_matching(
            liq_comps=liq, ol_comps=frames["ol"], equationH=equation,
            T=1300, P=5, eq_tests=equilibrium)
    elif api == "plag_temp_hygr":
        result = pt.calculate_fspar_liq_temp_hygr_matching(
            liq, frames["plag"], equationT=equation, equationH="H_Waters2015",
            P=5, iterations=3, Ab_An_P2008=False)
    else:
        phase = "plag" if api == "plag_temp" else "kspar"
        result = pt.calculate_fspar_liq_temp_matching(
            liq_comps=liq, equationT=equation, P=5, Ab_An_P2008=False,
            **{phase + "_comps": frames[phase]})
    return result if isinstance(result, dict) else {"result": result}


def without_liquid_labels(frame):
    # Include averaged outputs, whose label columns have Mean_/Std_ prefixes.
    return frame.loc[:, [c for c in frame if "sample_id_liq" not in c.lower()]]


@pytest.mark.parametrize("mode,state", CASES, ids=[
    "-".join((mode[0], mode[1], "eq" if mode[2] else "no-eq", state))
    for mode, state in CASES
])
def test_matching_preserves_liquid_labels(mode, state, compositions):
    liq = compositions["liq"]
    if state in ("canonical", "both"):
        liq["Sample_ID_Liq"] = ["liq-A", "liq-B"]
    elif state == "zero-null":
        liq["Sample_ID_Liq"] = [0, None]
    if state in ("lower-only", "both", "zero-null"):
        # A caller's differently cased field is ordinary metadata, not an alias.
        liq["Sample_ID_liq"] = [701.0, 702.0]
    originals = {name: frame.copy(deep=True) for name, frame in compositions.items()}

    tables = match(mode, compositions)
    pair_key = "All_PTs" if "All_PTs" in tables else "All_HTs" if "All_HTs" in tables else "result"
    pairs = tables[pair_key]
    assert len(pairs) == 4
    assert list(pairs["ID_liq"]) == [11, 29, 11, 29]
    phase_key = "ID_Ol" if mode[0].startswith("ol_") else "ID_Fspar"
    assert list(pairs[phase_key]) == [11, 11, 29, 29]
    assert list(pairs.columns).count("Sample_ID_Liq") == 1
    expected = liq["Sample_ID_Liq"] if "Sample_ID_Liq" in liq else pd.Series(
        liq.index.astype("float64"), index=liq.index)
    pd.testing.assert_series_equal(
        pairs["Sample_ID_Liq"].reset_index(drop=True),
        expected.iloc[[0, 1, 0, 1]].reset_index(drop=True),
        check_names=False, check_dtype=False, check_exact=True)
    if "Sample_ID_liq" in liq:
        assert list(pairs["Sample_ID_liq"]) == [701.0, 702.0, 701.0, 702.0]
    else:
        assert "Sample_ID_liq" not in pairs
    assert list(pairs["User_Trace"]) == [7.25, 9.5, 7.25, 9.5]
    primary = "T_K_calc" if "T_K_calc" in pairs else "H2O_calc"
    assert np.isfinite(pairs[primary].to_numpy(dtype=float)).all()
    for name, original in originals.items():
        pd.testing.assert_frame_equal(original, compositions[name], check_exact=True)

    # Changing only labels must leave every scientific/iteration table unchanged.
    reference = {name: frame.copy(deep=True) for name, frame in originals.items()}
    reference["liq"]["Sample_ID_Liq"] = ["reference-A", "reference-B"]
    reference_tables = match(mode, reference)
    assert tables.keys() == reference_tables.keys()
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(
            without_liquid_labels(frame), without_liquid_labels(reference_tables[name]),
            check_exact=True)
