import pandas as pd
import numpy as np

Average_DMM_WH2005 = {
    "Rb": 0.05, "Ba": 0.563, "Th": 0.0079, "U": 0.0032, "Nb": 0.1485, "Ta": 0.0096,
    "La": 0.192, "Ce": 0.55, "Pb": 0.018, "Pr": 0.107, "Nd": 0.581, "Sr": 7.664,
    "Zr": 5.082, "Hf": 0.157, "Sm": 0.239, "Eu": 0.096, "Ti": 716.3, "Gd": 0.358,
    "Tb": 0.07, "Dy": 0.505, "Ho": 0.115, "Y": 3.328, "Er": 0.348, "Yb": 0.365, "Lu": 0.058
}


NMORB_Gale13 = {
    "Ba": 19.6, "Be": 0.64, "Ce": 12.42, "Co": 42.3, "Cr": 263, "Cs": 0.024, "Cu": 73, "Dy": 5.5, "Er": 3.42, "Eu": 1.26,
    "Ga": 17, "Gd": 4.55, "Hf": 2.46, "Ho": 1.18, "La": 4.19, "Li": 6.1, "Lu": 0.48, "Mo": 0.36, "Nb": 3.62, "Nd": 10.66,
    "Ni": 100, "Pb": 0.51, "Pr": 1.98, "Rb": 1.84, "Sc": 38.4, "Sm": 3.48, "Sn": 0.8, "Sr": 128, "Ta": 0.24, "Tb": 0.82, "Th": 0.252,
    "Tl": 0.017, "Tmb": 0.52, "U": 0.083, "V": 280, "W": 0.08, "Y": 33.2, "Yb": 3.28, "Zn": 85.3, "Zr": 101.9
}

import pandas as pd

import pandas as pd


def prepare_trace_elements(df, reference_source=None):
    """Cleans, normalizes a dataframe, and returns the list of plotted elements.

    Parameters:
    -----------
    df : pd.DataFrame
        The input geological dataset.
    reference_source : dict or pd.Series, optional
        The reference values. Defaults to 'NMORB_Gale13' if available globally.

    Returns:
    --------
    normalized_df : pd.DataFrame
        The ready-to-plot normalized data.
    plot_elems : list
        The filtered list of elements used for setting plot x-ticks.
    """
    elements = [
        "Rb",
        "Ba",
         "U",
        "Th",
        "Nb",
        "Ta",
        "La",

        "Pb",
         "Ce",
      #  "Pr",
       "Sr",
        "Nd",
        "Zr",
       # "Hf",
        "Sm",
        "Eu",
      #  "Ti",
        "Gd",
        "Y",
        "Er",
        "Yb",
        "Lu"
    ]

    # 1. Clean data
    cols_present = [c for c in elements if c in df.columns]
    cleaned_df = df.copy()
    if cols_present:
        cleaned_df[cols_present] = (
            cleaned_df[cols_present]
            .apply(lambda s: s.replace(r"[^0-9eE\.\+\-]", "", regex=True))
            .apply(pd.to_numeric, errors="coerce")
        )

    # 2. Setup Reference
    if reference_source is None:
        try:
            reference_source = NMORB_Gale13
        except NameError:
            raise NameError(
                "NMORB_Gale13 is not defined globally. Please pass it explicitly."
            )

    ref_full = pd.Series(reference_source, dtype="float64")
    ref = ref_full.reindex(elements).dropna()
    ref = ref[ref != 0.0]

    plot_elems = ref.index.tolist()

    # 3. Normalize
    aligned_df = cleaned_df.reindex(columns=plot_elems)
    normalized_df = aligned_df.divide(ref, axis=1)

    return normalized_df, plot_elems


## Steves melting model
