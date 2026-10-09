""" Thermobar web app: thermobarometry and hygrometry from paired mineral and liquid compositions. Based on other streamlit apps built for mineralML and PyIRoGlass. // @author: Sarah Shi """

import io
import inspect
import warnings
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import Thermobar as pt

st.set_page_config(page_title="Thermobar", page_icon="🌡️", initial_sidebar_state=350)  # sidebar width, px
st.set_page_config(initial_sidebar_state="expanded")  # additive: keeps the width, and opens the sidebar

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "docs" / "Examples"
MAX_ROWS = 50_000
SAMPLE_COLS = ["Sample_ID", "Sample", "SampleID", "Sample Name", "Sample ID", "label", "Label"]

PHASE_NAMES = {"Liq": "liquid", "Ol": "olivine", "Cpx": "clinopyroxene", "Opx": "orthopyroxene",
               "Plag": "plagioclase", "Kspar": "alkali feldspar", "Amp": "amphibole"}
COMPS_KW = {p: p.lower() + "_comps" for p in PHASE_NAMES}  # Liq -> liq_comps, as Thermobar's functions expect
IMPORT_KEY = {p: p + "s" for p in PHASE_NAMES}  # import_excel output keys: Liqs, Cpxs, ...
OXIDES = ["SiO2", "TiO2", "Al2O3", "FeOt", "MnO", "MgO", "CaO", "Na2O", "K2O", "Cr2O3", "P2O5"]

# Registered in Thermobar but failing as of v1.0.73; re-check on each release. Cpx-only: Nimis_1999.py sets
# Si_recalc=2 (int), which pandas 3 will not overwrite with floats. Cpx-liquid: the registered function is a stub.
BROKEN = {"P_Nimis1999_BA"}

# Machine-learning cpx models live in the separate Thermobar_onnx package. Only the _onnx versions are offered:
# they give the same values as the pickled ones, but do not break when scikit-learn is upgraded.
try:
    import Thermobar_onnx
    HAS_ML = True
except ImportError:
    HAS_ML = False
ML_MODELS = ("Petrelli", "Jorgenson")

SOLVE, FIXED_P, FIXED_T, SOLVE_TH = ("Solve P and T together", "Temperature at a set pressure",
                                     "Pressure at a set temperature", "Temperature and H₂O together")

# Each system: phases in every row, and Thermobar's (equation registry, function) for T, P, P-T, and T-H2O.
# Menus list every equation in the registry; ML models only in their _onnx form, and only if Thermobar_onnx is installed.
# default: the equation selected first, as used in Thermobar's example notebook for that system (Thermobar has no
# built-in defaults). extra: equations the calculate_ function accepts but that are missing from the registry.
SYSTEMS = {
    "Liquid only": dict(
        phases=("Liq",), T=("Liquid_only_funcs", pt.calculate_liq_only_temp), default=dict(T="T_Put2008_eq15"),
        example=("Liquid_Ol_Liq_Themometry/Liquid_only_Thermometry.xlsx", "Liquid_only")),
    "Olivine-liquid": dict(
        phases=("Ol", "Liq"), T=("Liquid_olivine_funcs", pt.calculate_ol_liq_temp), default=dict(T="T_Put2008_eq22"),
        example=("Liquid_Ol_Liq_Themometry/Liquid_only_Thermometry.xlsx", "Ol-Liq")),
    "Clinopyroxene-liquid": dict(
        phases=("Cpx", "Liq"), T=("Cpx_Liq_T_funcs", pt.calculate_cpx_liq_temp),
        P=("Cpx_Liq_P_funcs", pt.calculate_cpx_liq_press), PT=pt.calculate_cpx_liq_press_temp,
        default=dict(T="T_Put2008_eq33", P="P_Put2008_eq30"),
        example=("Cpx_Cpx_Liq_Thermobarometry/Cpx_Liq_Example.xlsx", "Sheet1")),
    "Clinopyroxene only": dict(
        phases=("Cpx",), T=("Cpx_only_T_funcs", pt.calculate_cpx_only_temp),
        P=("Cpx_only_P_funcs", pt.calculate_cpx_only_press), PT=pt.calculate_cpx_only_press_temp,
        default=dict(T="T_Put2008_eq32d", P="P_Put2008_eq32a"),  # the notebooks' eq32b needs a melt H2O
        example=("Cpx_Cpx_Liq_Thermobarometry/Cpx_Liq_Example.xlsx", "Sheet1")),
    "Orthopyroxene-liquid": dict(
        phases=("Opx", "Liq"), T=("Opx_Liq_T_funcs", pt.calculate_opx_liq_temp),
        P=("Opx_Liq_P_funcs", pt.calculate_opx_liq_press), PT=pt.calculate_opx_liq_press_temp,
        default=dict(T="T_Put2008_eq28a", P="P_Put2008_eq29a"),
        example=("Opx_and_Opx_Liq_Thermobarometry/Opx_Liq_Example.xlsx", "Paired_Opx_Liq")),
    "Orthopyroxene only": dict(
        phases=("Opx",), P=("Opx_only_P_funcs", pt.calculate_opx_only_press),
        exclude={"P_Put2008_eq29cnoCr"},  # registered, but calculate_opx_only_press only accepts eq29c
        example=("Opx_and_Opx_Liq_Thermobarometry/Opx_Liq_Example.xlsx", "Paired_Opx_Liq")),
    "Two pyroxene": dict(
        phases=("Cpx", "Opx"), T=("Cpx_Opx_T_funcs", pt.calculate_cpx_opx_temp),
        P=("Cpx_Opx_P_funcs", pt.calculate_cpx_opx_press), PT=pt.calculate_cpx_opx_press_temp,
        default=dict(T="T_Put2008_eq36", P="P_Put2008_eq39"),
        example=("Two_Pyroxene_Thermobarometry/Two_pyroxene_input.xlsx", "Paired_Cpx_Opx")),
    "Plagioclase-liquid": dict(
        phases=("Plag", "Liq"), T=("plag_liq_T_funcs", pt.calculate_fspar_liq_temp),
        P=("plag_liq_P_funcs", pt.calculate_fspar_liq_press), PT=pt.calculate_fspar_liq_press_temp,
        H=("plag_liq_H_funcs", pt.calculate_fspar_liq_temp_hygr),
        default=dict(T="T_Put2008_eq23", P="P_Put2008_eq25", H="H_Waters2015"),
        example=("Feldspar_Thermobarometry/Feldspar_Liquid.xlsx", "Plag_Liquid")),
    "Alkali feldspar-liquid": dict(
        phases=("Kspar", "Liq"), T=("Kspar_Liq_T_funcs", pt.calculate_fspar_liq_temp),
        example=("Feldspar_Thermobarometry/Feldspar_Liquid.xlsx", "Kspar_Liquid")),
    "Two feldspar": dict(
        phases=("Plag", "Kspar"), T=("Plag_Kspar_T_funcs", pt.calculate_plag_kspar_temp),
        example=("Feldspar_Thermobarometry/Two_Feldspar_input.xlsx", "Paired_Plag_Kspar")),
    "Amphibole only": dict(
        phases=("Amp",), T=("Amp_only_T_funcs", pt.calculate_amp_only_temp),
        P=("Amp_only_P_funcs", pt.calculate_amp_only_press), PT=pt.calculate_amp_only_press_temp,
        exclude={"T_Put2016_eq4a_amp_sat"},  # needs a liquid; it is under Amphibole-liquid
        default=dict(T="T_Ridolfi2012", P="P_Ridolfi2021"),
        extra=dict(P={"P_Ridolfi2021", "P_Mutch2016"}),
        example=("Amphibole/Amphibole_Liquids.xlsx", "Amp-Liq")),
    "Amphibole-liquid": dict(
        phases=("Amp", "Liq"), T=("Amp_Liq_T_funcs", pt.calculate_amp_liq_temp),
        P=("Amp_Liq_P_funcs", pt.calculate_amp_liq_press), PT=pt.calculate_amp_liq_press_temp,
        default=dict(T="T_Put2016_eq4b", P="P_Put2016_eq7a"),
        example=("Amphibole/Amphibole_Liquids.xlsx", "Amp-Liq")),
}

CITATION = (
    "Wieser, P. E., Petrelli, M., Lubbers, J., Wieser, E., Özaydın, S., Kent, A. J. R. and Till, C. B. (2022) "
    "Thermobar: An open-source Python3 tool for thermobarometry and hygrometry. Volcanica, 5(2), pp. 349-384. "
    "doi: 10.30909/vol.05.02.349384."
)
BIBTEX = (
    "@article{Wieseretal2022,\n"
    "  doi     = {10.30909/vol.05.02.349384},\n"
    "  url     = {https://doi.org/10.30909/vol.05.02.349384},\n"
    "  year    = {2022},\n"
    "  volume  = {5},\n"
    "  number  = {2},\n"
    "  pages   = {349-384},\n"
    "  author  = {Wieser, Penny E. and Petrelli, Maurizio and Lubbers, Jordan and Wieser, Eric and "
    "{\\\"O}zayd{\\i}n, Sinan and Kent, Adam J. R. and Till, Christy B.},\n"
    "  title   = {Thermobar: An open-source Python3 tool for thermobarometry and hygrometry},\n"
    "  journal = {Volcanica}\n"
    "}"
)


# %% ----------------------------------------------------------------
# helpers

def registry(name):
    return getattr(pt, name + "_by_name")


def equations(spec, key):
    """Equation names for one quantity of a system, from Thermobar's own registry."""
    reg, _ = spec[key]
    skip = spec.get("exclude", set()) | BROKEN
    def offered(n):
        if n in skip or "Test" in n:
            return False
        if any(m in n for m in ML_MODELS):
            return HAS_ML and n.endswith("_onnx")
        return True
    return sorted(filter(offered, set(registry(reg)) | spec.get("extra", {}).get(key, set())))


def choose(label, spec, key):
    eqs = equations(spec, key)
    default = spec.get("default", {}).get(key)
    return st.selectbox(label, eqs, index=eqs.index(default) if default in eqs else 0)


def needs(spec, key, name, arg):
    """True if an equation needs P, T or H2O_Liq as input: Thermobar's equations take those with no default."""
    fn = registry(spec[key][0]).get(name)
    if fn is None:  # an extra: P_Ridolfi2021 and P_Mutch2016 take no P, T or H2O
        return False
    param = inspect.signature(fn).parameters.get(arg)
    return param is not None and param.default is inspect.Parameter.empty


def has_eq_tests(fn):
    return "eq_tests" in inspect.signature(fn).parameters


def clean(df):
    """Drops pandas index and note columns, and empty rows."""
    df.columns = df.columns.astype(str).str.strip()  # import_excel warns about leading spaces; strip them instead
    df = df.loc[:, ~df.columns.str.startswith(("Unnamed", "You "))]  # "You only need these columns..." notes in the examples
    return df.dropna(how="all").reset_index(drop=True)


def read_upload(uploaded, sheet=None):
    if uploaded.name.lower().endswith(".csv"):
        return clean(pd.read_csv(uploaded, encoding="utf-8-sig"))  # utf-8-sig strips Excel's byte-order mark
    return clean(pd.read_excel(uploaded, sheet_name=sheet or 0))


def template(phases):
    cols = ["Sample_ID"] + [f"{ox}_{p}" for p in phases for ox in OXIDES]
    if "Ol" in phases:
        cols += ["NiO_Ol", "NiO_Liq"]
    if "Liq" in phases:
        cols += ["H2O_Liq", "Fe3Fet_Liq"]
    return pd.DataFrame(columns=cols)


def call(fn, **kw):
    """Calls a Thermobar function with only the arguments it accepts, capturing its warnings and printed messages."""
    params = inspect.signature(fn).parameters
    kw = {k: v for k, v in kw.items() if k in params and v is not None}
    if "eq_tests" in params:
        kw["eq_tests"] = True
    printed = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(printed):
        warnings.simplefilter("always")
        out = fn(**kw)
    msgs = sorted({str(w.message) for w in caught if issubclass(w.category, UserWarning)})
    return out, msgs, printed.getvalue()


@st.cache_data(show_spinner=False, max_entries=16)
def run(system, df, mode, eqT, eqP, eqH, P, T, H2O, dNNO):
    spec = SYSTEMS[system]
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")  # lower-case suffix warnings etc. are reported below from the raw columns
        data = pt.import_excel(df=df)
    kw = {COMPS_KW[p]: data[IMPORT_KEY[p]] for p in spec["phases"]}
    kw.update(H2O_Liq=H2O, deltaNNO=dNNO)

    if mode == SOLVE:
        out, msgs, printed = call(spec["PT"], equationT=eqT, equationP=eqP, **kw)
    elif mode == FIXED_P:
        out, msgs, printed = call(spec["T"][1], equationT=eqT, P=P, **kw)
    elif mode == FIXED_T:
        out, msgs, printed = call(spec["P"][1], equationP=eqP, T=T, **kw)
    else:  # SOLVE_TH, plagioclase-liquid only
        out, msgs, printed = call(spec["H"][1], equationT=eqT, equationH=eqH, P=P, **kw)
        out = out["T_H_calc"]
        printed = ""  # the T-H2O solver prints its iteration arrays; not useful here

    if isinstance(out, pd.Series):
        out = out.to_frame("T_K_calc" if mode == FIXED_P else "P_kbar_calc")
    out = out.loc[:, ~out.columns.duplicated()].reset_index(drop=True)  # two-feldspar output repeats "sum"
    if mode == FIXED_T and "P_kbar_calc" not in out and "P_kbar" in out:
        out = out.rename(columns={"P_kbar": "P_kbar_calc"})  # cpx-only pressures with eq_tests come back as P_kbar
    # Some eq_tests outputs echo the set P or T under a _calc name; relabel so it is not reported as calculated.
    out = out.rename(columns={FIXED_P: {"P_kbar_calc": "P_kbar_input"}, FIXED_T: {"T_K_calc": "T_K_input"}}.get(mode, {}))
    if "T_K_calc" in out:
        out.insert(out.columns.get_loc("T_K_calc") + 1, "T_C_calc", out["T_K_calc"] - 273.15)
    if "H2O_calc" in out:
        out = out.rename(columns={"H2O_calc": "H2O_Liq_calc"})
    sample = next((c for c in SAMPLE_COLS if c in df.columns), None)
    if sample is not None:
        out.insert(0, sample, df[sample].values)
    printed = sorted({ln.strip() for ln in printed.splitlines() if ln.strip() and "independent function" not in ln})
    return out, msgs, printed


def key_columns(res):
    """Calculated values and equilibrium tests: everything before the echoed input compositions."""
    for i, c in enumerate(res.columns):
        if any(c.startswith(ox + "_") for ox in OXIDES):
            return list(res.columns[:i])
    return list(res.columns)


@st.cache_data(show_spinner=False, max_entries=4)
def to_excel(results, df_in):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        results.to_excel(xw, sheet_name="Results", index=False)
        df_in.to_excel(xw, sheet_name="Input", index=False)
    return buf.getvalue()


def pt_chart(res):
    """T (°C) against P (kbar), pressure increasing downward as in the crust."""
    has_T, has_P = "T_C_calc" in res, "P_kbar_calc" in res
    vals = [c for c in ("T_C_calc", "P_kbar_calc", "H2O_Liq_calc") if c in res]
    first = [c for c in res.columns[:1] if c not in vals]  # sample name, if any
    base = alt.Chart(res[first + vals]).properties(height=380)
    tip = first + [alt.Tooltip(c, format=".1f") for c in vals]
    if has_T and has_P:
        return base.mark_circle(size=70, opacity=0.8, stroke="white", strokeWidth=1).encode(
            x=alt.X("T_C_calc:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
            y=alt.Y("P_kbar_calc:Q", title="Pressure (kbar)", scale=alt.Scale(reverse=True, zero=False)),
            tooltip=tip,
        )
    col, title = ("T_C_calc", "Temperature (°C)") if has_T else ("P_kbar_calc", "Pressure (kbar)")
    return base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=title),
        y=alt.Y("count():Q", title="Number of analyses"),
        tooltip=[alt.Tooltip(f"{col}:Q", bin=True, title=title, format=".0f"), alt.Tooltip("count():Q", title="Analyses")],
    )


def show_results(res, msgs, printed, df_in):
    for m in msgs:
        st.warning(m)

    st.header("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analyses", f"{len(res):,}")
    if "T_C_calc" in res:
        t = res["T_C_calc"]
        c2.metric("Median temperature", f"{t.median():.0f} °C" if t.notna().any() else "–",
                  help=f"Range {t.min():.0f} to {t.max():.0f} °C" if t.notna().any() else None)
    if "P_kbar_calc" in res:
        p = res["P_kbar_calc"]
        c3.metric("Median pressure", f"{p.median():.1f} kbar" if p.notna().any() else "–",
                  help=f"Range {p.min():.1f} to {p.max():.1f} kbar" if p.notna().any() else None)
    if "H2O_Liq_calc" in res:
        c4.metric("Median H₂O in liquid", f"{res['H2O_Liq_calc'].median():.2f} wt%")
    if "PH2O_kbar_calc" in res:  # Krawczynski et al. (2012) at a set temperature
        c4.metric("Median P(H₂O)", f"{res['PH2O_kbar_calc'].median():.2f} kbar")
    n_bad = int(res[[c for c in ("T_K_calc", "P_kbar_calc") if c in res]].isna().any(axis=1).sum())
    if n_bad:
        st.info(f"{n_bad:,} analyses returned no value (NaN): usually a missing oxide the equation needs, "
                "or a composition outside the equation's calibration.")
    if res[[c for c in ("T_C_calc", "P_kbar_calc") if c in res]].notna().any().any():
        st.altair_chart(pt_chart(res), width="stretch")

    st.header("Results")
    key = key_columns(res)
    fmt = {c: st.column_config.NumberColumn(format="%.2f") for c in key if pd.api.types.is_float_dtype(res[c])}
    st.dataframe(res[key], column_config=fmt, hide_index=True)
    st.caption("Calculated values and equilibrium tests. T_K_calc in K, T_C_calc in °C, P_kbar_calc in kbar; "
               "_input columns repeat the pressure or temperature you set. "
               "The downloads also include your input compositions and the calculated components.")
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button(
        "Download Excel", lambda: to_excel(res, df_in), file_name="Thermobar_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="A Results sheet with every calculated column, plus your input.",
    )
    d2.download_button("Download CSV", res.to_csv(index=False).encode(), file_name="Thermobar_results.csv", mime="text/csv")
    with st.expander("All columns"):
        st.dataframe(res, hide_index=True)
    if printed:
        with st.expander("Messages from Thermobar"):
            st.code("\n".join(printed), language=None)


def fixed_input(kind, columns, key, prefer_column=False):
    """A single value or a column from the file for a set P (kbar) or T (returned in K)."""
    label = "Pressure" if kind == "P" else "Temperature"
    src = st.radio(f"{label} source", ["Single value", "Column in file"], horizontal=True, key=f"{key}_src",
                   index=1 if prefer_column and columns else 0,
                   disabled=not columns, help=None if columns else "Your file has no numeric columns to use.")
    if src == "Single value" or not columns:
        if kind == "P":
            return st.number_input("Pressure (kbar)", 0.0, 100.0, 5.0, step=0.5, key=f"{key}_val")
        return st.number_input("Temperature (°C)", 500.0, 2000.0, 1100.0, step=10.0, key=f"{key}_val") + 273.15
    guess = {"P": ["P_kbar"], "T": ["T_K", "Temperature_C", "T_C"]}[kind]
    col = st.selectbox("Column", columns, key=f"{key}_col",
                       index=next((columns.index(c) for c in guess if c in columns), 0))
    if kind == "P":
        st.caption("Values in kbar." + (" A P_GPa column is converted." if col == "P_GPa" else ""))
        return col, "GPa" if col == "P_GPa" else "kbar"
    unit = st.radio("Units", ["K", "°C"], horizontal=True, key=f"{key}_unit",
                    index=0 if col.upper().startswith("T_K") else 1)
    return col, unit


def resolve(value, df):
    """Turns a (column, unit) choice from fixed_input into a Series in kbar or K."""
    if not isinstance(value, tuple):
        return value
    col, unit = value
    s = pd.to_numeric(df[col], errors="coerce")
    return {"GPa": s * 10, "°C": s + 273.15}.get(unit, s)


def page_header(intro, cite_note):
    st.title("Thermobar")
    st.caption('Streamlit App developed by Penny Wieser and Sarah Shi, with help from Claude Code. We welcome feedback! Note, the more computationally intensive calculations such as melt-matching are not available, because of streamlit memory limitations')
    st.markdown(intro)
    st.info(
        "**Please cite Thermobar in the main text of your paper**, not only the supplement, along with the original "
        "calibration paper for each equation you use. See **How to cite** below."
    )
    with st.expander("How to cite"):
        st.markdown("If you use Thermobar in your work, please cite it in the main text:")
        st.code(CITATION, language=None, wrap_lines=True)
        st.markdown("BibTeX:")
        st.code(BIBTEX, language="bibtex")
        st.markdown(cite_note)


def sidebar_about():
    st.header("About")
    st.markdown(
        "[Documentation](https://thermobar.readthedocs.io) · "
        "[GitHub](https://github.com/PennyWieser/Thermobar) · "
        "[Paper](https://doi.org/10.30909/vol.05.02.349384)"
    )
    st.caption(
        f"Thermobar v{pt.__version__}, by Penny Wieser and coauthors. Questions, bugs, or a model to add? "
        "Open a [GitHub issue](https://github.com/PennyWieser/Thermobar/issues)."
    )


def hist_chart(res, col, title):
    return alt.Chart(res[[col]]).properties(height=320).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=title),
        y=alt.Y("count():Q", title="Number of analyses"),
        tooltip=[alt.Tooltip(f"{col}:Q", bin=True, title=title, format=".2f"), alt.Tooltip("count():Q", title="Analyses")],
    )


# %% ----------------------------------------------------------------
# other functions: melt viscosity, redox conversions, pressure to depth, oxide to element wt%

TOOLS = ("Thermobarometry (no melt matching)", "Other functions")
OTHER_TOOL = TOOLS[1]
OTHER_DOCS = "https://thermobar.readthedocs.io/en/latest/Examples/Other_features/"
FEO_PER_FE2O3 = 2 * 71.844 / 159.688  # wt% FeO equivalent of 1 wt% Fe2O3

KRESS = "Kress, V. C. and Carmichael, I. S. E. (1991) Contributions to Mineralogy and Petrology, 108, 82-92."
PUT2016 = "Putirka, K. D. (2016) American Mineralogist, 101, 819-840."
FROST = "Frost, B. R. (1991) Reviews in Mineralogy, 25, 1-9."
VISC, FO2_BUF, FO2_FE, FE_FO2, DEPTH, OX_EL, CALI = (
    "Melt viscosity (Giordano et al. 2008)",
    "fO₂ to buffer position (ΔQFM, ΔNNO)",
    "fO₂ or buffer to melt Fe³⁺/ΣFe",
    "Melt Fe³⁺/ΣFe to fO₂",
    "Pressure to depth",
    "Oxide wt% to element wt%",
    "Compare with a model's calibration dataset",
)
# notebook: worked example in the docs; example: (file in docs/Examples/Other_features, sheet);
# typed: a starting table for values typed into the app.
OTHER = {
    VISC: dict(
        notebook="Calculating_Viscosity", example=("Other_features/Viscoity_Giordano.xlsx", "MELTSTest"),
        about="Viscosity of a silicate melt from its composition, including H₂O and optionally F, with the model of "
              "Giordano et al. (2008). Returns the VFT parameters `A`, `B` and `C` and, at a temperature, the viscosity "
              "`n_melt` (Pa s) and `logn_melt` (log₁₀ Pa s).\n\n"
              "Input: one liquid per row, oxides in wt% as `SiO2_Liq`, `H2O_Liq`... (or `SiO2`, `H2O`...).",
        cite=["Giordano, D., Russell, J. K. and Dingwell, D. B. (2008) Earth and Planetary Science Letters, 271, 123-134."]),
    FO2_BUF: dict(
        notebook="fo2_to_BufferPosition",
        typed=pd.DataFrame({"Sample_ID": [f"Blatter2017_{i}" for i in range(1, 5)],
                            "logfO2": [-12.3, -12.06, -12.35, -12.33], "T_C": [847.0, 859.0, 810.0, 816.0],
                            "P_kbar": [0.0, 0.0, 0.0, 0.0]}),
        about="Converts an absolute fO₂ (bar) at a temperature and pressure to ΔQFM and ΔNNO, using the buffers of "
              "Frost (1991), with QFM switching to the α-quartz equation below the α-β quartz transition. Also gives "
              "ΔQFM relative to O'Neill (1987).\n\n"
              "Type values into the table, or upload a file with fO₂, temperature and pressure columns.",
        cite=[FROST]),
    FO2_FE: dict(
        notebook="fo2_to_Fe3FeT", example=("Other_features/Redox_Conversions.xlsx", "Liqs"),
        about="Splits FeOt in a liquid into FeO and Fe₂O₃ at a set fO₂, given as an offset from the QFM or NNO buffer "
              "or as an absolute value. Returns `Fe3Fet_Liq` (Fe³⁺/ΣFe), `FeO_Liq` and `Fe2O3_Liq`.\n\n"
              "Input: one liquid per row, oxides in wt% as `SiO2_Liq`... (or `SiO2`...).",
        cite=[KRESS, PUT2016, FROST]),
    FE_FO2: dict(
        notebook="fo2_to_Fe3FeT", example=("Other_features/Redox_Conversions.xlsx", "Liqs"),
        about="The reverse: fO₂, ΔQFM and ΔNNO (Frost 1991) from the Fe³⁺/ΣFe of a liquid. Fe³⁺/ΣFe comes from a "
              "`Fe3Fet_Liq` column, from `FeO_Liq` and `Fe2O3_Liq` columns, or a single value you set.\n\n"
              "Input: one liquid per row, oxides in wt% as `SiO2_Liq`... (or `SiO2`...).",
        cite=[KRESS, PUT2016, FROST]),
    DEPTH: dict(
        notebook="Pressure_to_depth",
        typed=pd.DataFrame({"Sample_ID": [f"P{i}" for i in range(1, 6)], "P_kbar": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        about="Converts pressure (kbar) to depth (km) with a constant crustal density, a stepped density profile, or "
              "a published density profile. Some published profiles return no depth (NaN) beyond the depth range "
              "they were fitted to.\n\n"
              "Type pressures into the table, or upload a file with a `P_kbar` column (e.g. Thermobar results).",
        cite=["The source of the density profile you use (listed in the Thermobar documentation)."]),
    OX_EL: dict(
        notebook="oxide_to_wtperc", example=("Other_features/Ox_to_element.xlsx", "Plag"),
        about="Converts oxide wt% to element wt%, either with oxygen or as cations only, normalized to 100%.\n\n"
              "Input: one analysis per row, oxides in wt% with a phase suffix (`SiO2_Plag`...) or none (`SiO2`...).",
        cite=[]),
    CALI: dict(
        notebook="Calibration_Plot_Examples",
        about="Plots your analyses among the calibration dataset of a model, to check whether your compositions (and "
              "calculated pressures and temperatures) fall within the range the model was calibrated on. Your file is "
              "run through the same component calculations as the calibration data (amphibole sites after Ridolfi, "
              "clinopyroxene components, plagioclase An and Ab), so you can plot any column the two share.\n\n"
              "Input: one analysis per row, with columns like `SiO2_Amp`, `SiO2_Cpx`, `SiO2_Plag` or `SiO2_Liq`. "
              "To plot pressure or temperature, upload Thermobar results (e.g. with `P_kbar_calc` and `T_K_calc` "
              "columns) and pick those columns under Settings.",
        cite=["The paper for the model whose calibration dataset you plot."]),
}
# Calibration datasets in Thermobar's return_cali_dataset: label -> (model, phase, example, default x and y).
# HelzThornber87 is listed by Thermobar but its file is missing from the package, so it is left out.
_EX_LPA = "Other_features/Example_Liq_Px_Amp.xlsx"
CALI_MODELS = {
    "Amphibole: Ridolfi (2021)": ("Ridolfi2021", "Amp", (_EX_LPA, "Amphibole"), ("Mgno_FeT", "Al2O3_Amp")),
    "Amphibole: Mutch et al. (2016)": ("Mutch2016", "Amp", (_EX_LPA, "Amphibole"), ("Mgno_FeT", "Al2O3_Amp")),
    "Amphibole: Putirka (2016)": ("Putirka2016", "Amp", (_EX_LPA, "Amphibole"), ("Mgno_FeT", "Al2O3_Amp")),
    "Amphibole: Zhang et al. (2017)": ("Zhang2017", "Amp", (_EX_LPA, "Amphibole"), ("Mgno_FeT", "Al2O3_Amp")),
    "Clinopyroxene: Putirka (2008), whole Cpx-Liq database": ("Putirka2008", "Cpx", (_EX_LPA, "Pyroxene"), ("Mgno_Cpx", "Na2O_Cpx")),
    "Clinopyroxene: Neave and Putirka (2017)": ("Neave2017", "Cpx", (_EX_LPA, "Pyroxene"), ("Mgno_Cpx", "Na2O_Cpx")),
    "Clinopyroxene: Masotta et al. (2013)": ("Masotta2013", "Cpx", (_EX_LPA, "Pyroxene"), ("Mgno_Cpx", "Na2O_Cpx")),
    "Clinopyroxene: Brugman and Till (2019)": ("Brugman2019", "Cpx", (_EX_LPA, "Pyroxene"), ("Mgno_Cpx", "Na2O_Cpx")),
    "Clinopyroxene: Petrelli et al. (2020)": ("Petrelli2020", "Cpx", (_EX_LPA, "Pyroxene"), ("Mgno_Cpx", "Na2O_Cpx")),
    "Clinopyroxene: Wang et al. (2021)": ("Wang2021", "Cpx", (_EX_LPA, "Pyroxene"), ("Mgno_Cpx", "Na2O_Cpx")),
    "Clinopyroxene: Jorgenson et al. (2022)": ("Jorgenson2022", "Cpx", (_EX_LPA, "Pyroxene"), ("Mgno_Cpx", "Na2O_Cpx")),
    "Plagioclase: Waters and Lange (2015)": ("Waters2015", "Plag",
        ("Feldspar_Thermobarometry/Feldspar_Liquid.xlsx", "Plag_Liquid"), ("XAn", "SiO2_Liq")),
    "Plagioclase: Masotta et al. (2019)": ("Masotta2019", "Plag",
        ("Feldspar_Thermobarometry/Feldspar_Liquid.xlsx", "Plag_Liquid"), ("XAn", "SiO2_Liq")),
    "Liquid: Shea et al. (2022)": ("Shea2022", "Liq", (_EX_LPA, "Liquid"), ("MgO_Liq", "FeOt_Liq")),
}
CALI_SKIP = ("Sample_ID", "Input_Check", "Sum_input", "Jd_from", "oxy_renorm", "O=F,Cl", "Charge", "Unnamed")
CALI_LABEL = ("Reference", "Data Source", "Source", "Ref.", "Sample_ID_Cpx", "Sample I.D.", "Exp", "Expename")
REDOX_MODELS = {"Kress and Carmichael (1991)": "Kress1991", "Putirka (2016), eq. 6b": "Put2016_eq6b"}
DENSITY_MODELS = {
    "Constant density": None,
    "Two-step profile": "two-step",
    "Three-step profile": "three-step",
    "Four-step profile": "four-step",
    "Ryan (1987), via Lerner et al. (2021): Hawaii": "ryan_lerner",
    "Denlinger and Flinders (2022), via Lerner et al. (2024)": "denlinger_lerner",
    "Rasmussen et al. (2022)": "rasmussen",
    "Mavko and Thompson (1983), DeBari and Greene (2011)": "mavko_debari",
    "Hill and Zucca (1987)": "hill_zucca",
    "Prezzi et al. (2009): Andes": "prezzi",
}
STEP_DEFAULTS = {2: ([14.0], [2700.0, 3100.0]), 3: ([5.0, 14.0], [2500.0, 2900.0, 3100.0]),
                 4: ([5.0, 14.0, 20.0], [2500.0, 2700.0, 2900.0, 3100.0])}


def as_series(x, n):
    """A per-row Series from a single value or a Series (Thermobar's redox functions need matching Series)."""
    return x.astype(float) if isinstance(x, pd.Series) else pd.Series(np.full(n, float(x)))


def guess_index(columns, names):
    return next((columns.index(c) for c in names if c in columns), 0)


def value_or_column(label, columns, key, value, guess, **num):
    """A single number, or ("column", None) for a column in the file; turn it into values with resolve()."""
    src = st.radio(f"{label}", ["Single value", "Column in file"], horizontal=True, key=f"{key}_src",
                   disabled=not columns)
    if src == "Single value" or not columns:
        return st.number_input(label, value=value, key=f"{key}_val", label_visibility="collapsed", **num)
    return st.selectbox("Column", columns, key=f"{key}_col", index=guess_index(columns, guess)), None


def liquids(df):
    """Liquid compositions as Thermobar expects them, from columns named SiO2_Liq... or SiO2..."""
    has_suffix = any(f"{ox}_Liq" in df.columns for ox in OXIDES)
    if not has_suffix and not any(ox in df.columns for ox in OXIDES):
        raise ValueError(f"No liquid oxide columns found (e.g. SiO2_Liq or SiO2). Columns in your file: {list(df.columns)}")
    if not has_suffix:  # add _Liq, leaving any column that already has it (e.g. Fe3Fet_Liq)
        df = df.rename(columns=lambda c: c if c.endswith("_Liq") else c + "_Liq")
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return pt.import_excel(df=df)["Liqs"]


def add_T_C(out, after="T_K"):
    out = out.loc[:, ~out.columns.duplicated()]  # the viscosity output repeats T_K
    if after in out:
        out.insert(out.columns.get_loc(after) + 1, "T_C", out[after] - 273.15)
    return out


def form_viscosity(df, cols, spec):
    T = None
    if st.checkbox("Calculate viscosity at a temperature", value=True,
                   help="Without a temperature, only the VFT parameters A, B and C are returned."):
        T = fixed_input("T", cols, "visc_T")
    H2O = None
    if st.checkbox("Use a fixed value of H₂O for all analyses", value=False,
                   help="Overrides any H2O_Liq column. Viscosity is very sensitive to H₂O."):
        H2O = st.number_input("H₂O in liquid (wt%)", 0.0, 20.0, 1.0, step=0.1)
    F_ppm = st.number_input("F in liquid (ppm)", 0.0, 50_000.0, 0.0, step=100.0,
                            help="Applied to all analyses; converted to F₂O for the Giordano et al. (2008) model.")

    def job():
        out, msgs, printed = call(pt.calculate_viscosity_giordano_2008, liq_comps=liquids(df), T=resolve(T, df),
                                  H2O_Liq=H2O, F2O_content=pt.convert_F_to_F2O(F_ppm=F_ppm))
        return add_T_C(out), msgs, printed, ["n_melt", "logn_melt", "T_K", "T_C", "A", "B", "C"], \
            ("logn_melt", "Viscosity (log₁₀ Pa s)")
    return job


def form_fo2_buffer(df, cols, spec):
    fcol = st.selectbox("fO₂ column", cols, index=guess_index(cols, ["logfO2", "logfo2", "log_fO2", "fo2_bars", "fO2", "fo2"]),
                        disabled=not cols)
    funit = st.radio("fO₂ units", ["log₁₀ fO₂ (bar)", "fO₂ (bar)"], horizontal=True,
                     index=0 if str(fcol).lower().startswith("log") else 1)
    T = fixed_input("T", cols, "buf_T", prefer_column=True)
    P = fixed_input("P", cols, "buf_P", prefer_column=True)

    def job():
        if fcol is None:
            raise ValueError("No numeric column for fO₂ in the input.")
        n = len(df)
        f = pd.to_numeric(df[fcol], errors="coerce").astype(float)
        fo2 = 10 ** f if funit.startswith("log") else f
        out, msgs, printed = call(pt.convert_fo2_to_buffer, fo2=fo2, T_K=as_series(resolve(T, df), n),
                                  P_kbar=as_series(resolve(P, df), n))
        out.insert(out.columns.get_loc("fo2") + 1, "logfO2", np.log10(out["fo2"]))
        key = ["DeltaQFM_Frost1991", "DeltaNNO_Frost1991", "DeltaQFM_ONeill1987", "QFM_equation_Choice",
               "logfO2", "T_K", "T_C", "P_kbar"]
        return add_T_C(out), msgs, printed, key, ("DeltaQFM_Frost1991", "ΔQFM (Frost 1991)")
    return job


def form_fo2_fe(df, cols, spec):
    how = st.radio("fO₂ given as", ["Offset from a buffer", "Absolute fO₂"], horizontal=True)
    if how == "Offset from a buffer":
        buffer = st.selectbox("Buffer", ["QFM", "NNO"])
        offset = value_or_column("Offset from buffer (log units)", cols, "fe_off", 0.0,
                                 ["DeltaQFM", "deltaQFM", "dQFM", "DeltaNNO", "deltaNNO", "dNNO"],
                                 min_value=-10.0, max_value=10.0, step=0.5)
    else:
        logfo2 = value_or_column("log₁₀ fO₂ (bar)", cols, "fe_abs", -8.0, ["logfO2", "logfo2", "log_fO2"],
                                 min_value=-40.0, max_value=5.0, step=0.5)
    T = fixed_input("T", cols, "fe_T")
    P = fixed_input("P", cols, "fe_P")
    model = REDOX_MODELS[st.selectbox("Model", list(REDOX_MODELS))]
    renorm = st.checkbox("Renormalize other oxides", value=False,
                         help="Following K. Iacovino's spreadsheet: rescales the other oxides for the change in "
                              "total when FeOt is split into FeO and Fe₂O₃.")

    def job():
        n = len(df)
        if how == "Offset from a buffer":
            fo2_kw = dict(fo2=buffer, fo2_offset=as_series(resolve(offset, df), n))
        else:
            fo2_kw = dict(fo2=10 ** as_series(resolve(logfo2, df), n))
        out, msgs, printed = call(pt.convert_fo2_to_fe_partition, liq_comps=liquids(df), model=model, renorm=renorm,
                                  T_K=as_series(resolve(T, df), n), P_kbar=as_series(resolve(P, df), n), **fo2_kw)
        return out, msgs, printed, ["Fe3Fet_Liq", "FeO_Liq", "Fe2O3_Liq"], ("Fe3Fet_Liq", "Fe³⁺/ΣFe")
    return job


def form_fe_fo2(df, cols, spec):
    has_fe = df is not None and (any(c in df for c in ("Fe3Fet_Liq", "Fe3Fet")) or
                                 (any(c in df for c in ("FeO_Liq", "FeO")) and any(c in df for c in ("Fe2O3_Liq", "Fe2O3"))))
    src = st.radio("Fe³⁺/ΣFe from", ["File columns", "Single value"], horizontal=True, index=0 if has_fe else 1,
                   help="File columns: Fe3Fet_Liq, or FeO_Liq and Fe2O3_Liq (also without the _Liq suffix).")
    fe3 = st.number_input("Fe³⁺/ΣFe", 0.0, 1.0, 0.15, step=0.01) if src == "Single value" else None
    T = fixed_input("T", cols, "fe2_T")
    P = fixed_input("P", cols, "fe2_P")
    model = REDOX_MODELS[st.selectbox("Model", list(REDOX_MODELS))]

    def job():
        n = len(df)
        liq = liquids(df)
        fe = fe3
        if fe is None:
            feo = next((c for c in ("FeO_Liq", "FeO") if c in df), None)
            fe2o3 = next((c for c in ("Fe2O3_Liq", "Fe2O3") if c in df), None)
            if feo and fe2o3:  # import_excel folds these into FeOt_Liq, so take the ratio from the raw columns
                fe_as_feo = pd.to_numeric(df[fe2o3], errors="coerce").fillna(0) * FEO_PER_FE2O3
                fe = fe_as_feo / (pd.to_numeric(df[feo], errors="coerce").fillna(0) + fe_as_feo)
            elif not (liq["Fe3Fet_Liq"] > 0).any():
                raise ValueError("No Fe³⁺/ΣFe in the input: add a Fe3Fet_Liq column, or FeO_Liq and Fe2O3_Liq "
                                 "columns, or choose Single value.")
        out, msgs, printed = call(pt.convert_fe_partition_to_fo2, liq_comps=liq, model=model, Fe3Fet_Liq=fe,
                                  T_K=as_series(resolve(T, df), n), P_kbar=as_series(resolve(P, df), n))
        with np.errstate(divide="ignore"):
            out.insert(out.columns.get_loc("fo2_calc") + 1, "logfO2_calc", np.log10(out["fo2_calc"]))
        key = ["DeltaQFM_Frost1991", "DeltaNNO_Frost1991", "fo2_calc", "logfO2_calc", "Fe3Fet_Liq"]
        return out, msgs, printed, key, ("DeltaQFM_Frost1991", "ΔQFM (Frost 1991)")
    return job


def form_depth(df, cols, spec):
    P = fixed_input("P", cols, "d_P", prefer_column=True)
    label = st.selectbox("Density model", list(DENSITY_MODELS))
    model = DENSITY_MODELS[label]
    kw = {}
    if model is None:
        kw["crust_dens_kgm3"] = st.number_input("Crustal density (kg/m³)", 1000.0, 4000.0, 2700.0, step=50.0)
    elif model.endswith("-step"):
        kw["model"] = model
        n_layers = {"two": 2, "three": 3, "four": 4}[model.split("-")[0]]
        depths, rhos = STEP_DEFAULTS[n_layers]
        for i, d in enumerate(depths, 1):
            kw[f"d{i}"] = st.number_input(f"Depth of step {i} (km)", 0.0, 200.0, d, step=1.0)
        for i, r in enumerate(rhos, 1):
            where = "surface to step 1" if i == 1 else f"below step {i - 1}"
            kw[f"rho{i}"] = st.number_input(f"Density, {where} (kg/m³)", 1000.0, 4000.0, r, step=50.0)
    else:
        kw["model"] = model

    def job():
        p = as_series(resolve(P, df), len(df))
        depth, msgs, printed = call(pt.convert_pressure_to_depth, P_kbar=p, **kw)
        out = pd.DataFrame({"P_kbar": p, "Depth_km": np.asarray(depth, dtype=float)})
        return out, msgs, printed, ["P_kbar", "Depth_km"], ("Depth_km", "Depth (km)")
    return job


def form_ox_el(df, cols, spec):
    found = [] if df is None else sorted({c[len("SiO2"):] for c in df.columns if c.startswith("SiO2_")})
    if df is not None and "SiO2" in df.columns:
        found.append("")
    suffix = st.selectbox("Phase suffix", found, format_func=lambda s: s or "No suffix",
                          help="The oxide columns to convert, e.g. SiO2_Plag.", disabled=not found)
    without_O = st.radio("Output", ["Elements and oxygen", "Cations only, normalized to 100%"]) != "Elements and oxygen"

    def job():
        if suffix is None:
            raise ValueError(f"No oxide columns found (e.g. SiO2_Plag or SiO2). Columns in your file: {list(df.columns)}")
        out, msgs, printed = call(pt.convert_oxide_percent_to_element_weight_percent, df=df,
                                  suffix=suffix or None, without_oxygen=without_O)
        return out, msgs, printed, list(out.columns), None
    return job


@st.cache_data(show_spinner=False, max_entries=16)
def cali_dataset(model):
    with contextlib.redirect_stdout(io.StringIO()):  # return_cali_dataset prints the file path
        cali = pt.return_cali_dataset(model=model)
    if "P_kbar" not in cali and "P (kbar)" in cali:  # Putirka (2016)
        cali["P_kbar"] = pd.to_numeric(cali["P (kbar)"], errors="coerce")
    return cali


@st.cache_data(show_spinner=False, max_entries=16)
def cali_user_data(df, phase):
    """Your analyses with the same components the calibration datasets carry (as in generic_cali_plot)."""
    if not any(f"{ox}_{phase}" in df.columns for ox in OXIDES):
        raise ValueError(f"No {PHASE_NAMES[phase]} columns found (e.g. SiO2_{phase}). "
                         f"Columns in your file: {list(df.columns)}")
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        data = pt.import_excel(df=df)
        comps = data[IMPORT_KEY[phase]]
        if phase == "Amp":
            parts = [comps, pt.calculate_sites_ridolfi(amp_comps=comps)]
        elif phase == "Cpx":
            parts = [pt.calculate_clinopyroxene_components(cpx_comps=comps)]
        elif phase == "Plag":
            cat = pt.calculate_cat_fractions_plagioclase(plag_comps=comps)
            parts = [comps, cat[["An_Plag", "Ab_Plag"]].rename(columns={"An_Plag": "XAn", "Ab_Plag": "XAb"})]
        else:
            parts = [comps]
    if phase != "Liq" and any(f"{ox}_Liq" in df.columns for ox in OXIDES):  # for the Cpx-Liq and Plag-Liq calibrations
        parts.append(data["Liqs"])
    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def form_cali(df, cols, spec):
    label = spec["cali_model"]
    model, phase, _, (dx, dy) = CALI_MODELS[label]
    opts = ["None"] + cols
    P_col = st.selectbox("Pressure column (kbar)", opts, help="Optional, e.g. P_kbar_calc from Thermobar results. "
                         "Plotted as P_kbar.", index=next((opts.index(c) for c in ("P_kbar_calc", "P_kbar") if c in opts), 0))
    T_col = st.selectbox("Temperature column (K)", opts, help="Optional, e.g. T_K_calc from Thermobar results. "
                         "Plotted as T_K.", index=next((opts.index(c) for c in ("T_K_calc", "T_K") if c in opts), 0))

    user, err, shared = None, None, []
    if df is not None:
        try:
            user = cali_user_data(df, phase).copy()
        except Exception as e:
            err = e
    if user is not None:
        if P_col != "None":
            user["P_kbar"] = pd.to_numeric(df[P_col], errors="coerce").values
        if T_col != "None":
            user["T_K"] = pd.to_numeric(df[T_col], errors="coerce").values
        cali = cali_dataset(model)
        numeric = lambda d: {c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])}
        shared = sorted(c for c in numeric(user) & numeric(cali) if not any(k in c for k in CALI_SKIP))
        shared = [c for c in ("P_kbar", "T_K") if c in shared] + [c for c in shared if c not in ("P_kbar", "T_K")]
    x = st.selectbox("x axis", shared, index=shared.index(dx) if dx in shared else 0, disabled=not shared)
    y = st.selectbox("y axis", shared, index=shared.index(dy) if dy in shared else min(1, len(shared) - 1),
                     disabled=not shared)
    cali_top = st.checkbox("Draw calibration data on top", value=False)

    def job():
        if err is not None:
            raise err
        if not shared:
            raise ValueError(f"Your data and the {model} calibration dataset share no numeric columns to plot.")
        cali_plot(model, phase, user, cali, x, y, cali_top, df)
        return None  # drawn already
    return job


def cali_plot(model, phase, user, cali, x, y, cali_top, df_in):
    sample = next((c for c in SAMPLE_COLS + [f"Sample_ID_{phase}"] if c in df_in.columns), None)
    cname = f"{model} calibration"
    num = lambda s: pd.to_numeric(s, errors="coerce").values
    u = pd.DataFrame({"x": num(user[x]), "y": num(user[y]), "Dataset": "Your data",
                      "Label": df_in[sample].astype(str).values if sample else [str(i + 1) for i in range(len(user))]})
    lab = next((c for c in CALI_LABEL if c in cali.columns), None)
    c = pd.DataFrame({"x": num(cali[x]), "y": num(cali[y]), "Dataset": cname,
                      "Label": cali[lab].astype(str).values if lab else ""})
    u, c = u.dropna(subset=["x", "y"]), c.dropna(subset=["x", "y"])

    st.header("Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Your analyses plotted", f"{len(u):,}", help=None if len(u) == len(user) else
              f"{len(user) - len(u):,} rows have no value for {x} or {y}.")
    m2.metric("Calibration analyses", f"{len(c):,}")
    if len(u) and len(c):
        inside = u["x"].between(c["x"].min(), c["x"].max()) & u["y"].between(c["y"].min(), c["y"].max())
        m3.metric("Inside calibration range", f"{inside.mean():.0%}",
                  help="Share of your analyses whose x and y both fall within the minimum to maximum of the "
                       "calibration data. A rough check: a point can be inside both ranges but away from the "
                       "calibration points.")

    dom = ["Your data", cname]
    enc = dict(
        x=alt.X("x:Q", title=x.replace("_", " "), scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title=y.replace("_", " "), scale=alt.Scale(zero=False)),
        color=alt.Color("Dataset:N", scale=alt.Scale(domain=dom, range=["#d62728", "#8c8c8c"]),
                        legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("Dataset:N", scale=alt.Scale(domain=dom, range=["triangle-up", "circle"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Dataset", "Label", alt.Tooltip("x:Q", title=x, format=".4g"), alt.Tooltip("y:Q", title=y, format=".4g")],
    )
    layer_c = alt.Chart(c).mark_point(filled=True, size=45, opacity=0.6, stroke="#333333", strokeWidth=0.5).encode(**enc)
    layer_u = alt.Chart(u).mark_point(filled=True, size=110, opacity=0.85, stroke="black", strokeWidth=0.8).encode(**enc)
    chart = alt.layer(layer_u, layer_c) if cali_top else alt.layer(layer_c, layer_u)
    st.altair_chart(chart.properties(height=460).interactive(), width="stretch")
    st.caption("Scroll to zoom and drag to pan; double-click to reset. Hover over a point for its sample or reference.")

    yours = user.copy()
    if sample is not None and sample not in yours:
        yours.insert(0, sample, df_in[sample].values)
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download calibration dataset (CSV)", cali.to_csv(index=False).encode(),
                       file_name=f"Thermobar_{model}_calibration.csv", mime="text/csv")
    d2.download_button("Download your data with components (CSV)", yours.to_csv(index=False).encode(),
                       file_name=f"Thermobar_{model}_your_data.csv", mime="text/csv")
    with st.expander(f"{model} calibration dataset"):
        st.dataframe(cali, hide_index=True)


OTHER_FORMS = {VISC: form_viscosity, FO2_BUF: form_fo2_buffer, FO2_FE: form_fo2_fe, FE_FO2: form_fe_fo2,
               DEPTH: form_depth, OX_EL: form_ox_el, CALI: form_cali}


def other_input(spec, name):
    """Sidebar input: values typed into a table (drawn on the page), an uploaded file, or the example file."""
    st.header("Input data")
    sources = (["Type values"] if "typed" in spec else []) + ["Upload a file"] + \
              (["Example data"] if "example" in spec else [])
    src = st.radio("Input", sources, horizontal=True, label_visibility="collapsed", key=f"{name}_input")
    if src == "Type values":
        return "typed", None
    if src == "Example data":
        path, sheet = spec["example"]
        return (clean(pd.read_excel(EXAMPLES / path, sheet_name=sheet)),
                f"Example data: `docs/Examples/{path}`, sheet `{sheet}`.")
    uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"], key=f"{name}_file",
                                help="One analysis per row; see About this function.")
    if uploaded is None:
        return None, None
    sheet = None
    if not uploaded.name.lower().endswith(".csv"):
        sheets = pd.ExcelFile(uploaded).sheet_names
        sheet = st.selectbox("Sheet", sheets, key=f"{name}_sheet") if len(sheets) > 1 else sheets[0]
    try:
        return read_upload(uploaded, sheet), None
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()


def show_other(name, res, msgs, printed, key, chart, df_in):
    for m in msgs:
        st.warning(m)
    sample = next((c for c in SAMPLE_COLS if c in df_in.columns), None)
    if sample is not None and sample not in res:
        res.insert(0, sample, df_in[sample].values)
    key = ([sample] if sample else []) + [c for c in key if c in res and c != sample]

    st.header("Summary")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.metric("Analyses", f"{len(res):,}")
    if chart and chart[0] in res:
        vals = pd.to_numeric(res[chart[0]], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().any():
            c2.metric(f"Median {chart[1]}", f"{vals.median():.3g}", help=f"Range {vals.min():.3g} to {vals.max():.3g}")
            st.altair_chart(hist_chart(res.assign(**{chart[0]: vals}), chart[0], chart[1]), width="stretch")
        n_bad = int(vals.isna().sum())
        if n_bad:
            st.info(f"{n_bad:,} analyses returned no value (NaN or infinite): usually a missing input, or a value "
                    "outside the model's range.")

    st.header("Results")
    fmt = {c: st.column_config.NumberColumn(format="%.4g") for c in key if pd.api.types.is_float_dtype(res[c])}
    st.dataframe(res[key], column_config=fmt, hide_index=True)
    fname = "Thermobar_" + "".join(ch if ch.isalnum() else "_" for ch in name.split(" (")[0]).strip("_")
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button(
        "Download Excel", lambda: to_excel(res, df_in), file_name=f"{fname}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="A Results sheet with every calculated column, plus your input.",
    )
    d2.download_button("Download CSV", res.to_csv(index=False).encode(), file_name=f"{fname}.csv", mime="text/csv")
    if len(key) < len(res.columns):
        with st.expander("All columns"):
            st.dataframe(res, hide_index=True)
    printed = sorted({ln.strip() for ln in printed.splitlines() if ln.strip()})
    if printed:
        with st.expander("Messages from Thermobar"):
            st.code("\n".join(printed), language=None)


def other_functions():
    """The Other functions page: draws its own sidebar and results."""
    with st.sidebar:
        name = st.selectbox("Function", list(OTHER))
        spec = OTHER[name]
        if name == CALI:  # the dataset decides the phase, and so the example file
            cali_label = st.selectbox("Calibration dataset", list(CALI_MODELS))
            spec = dict(spec, cali_model=cali_label, example=CALI_MODELS[cali_label][2])
    page_header(
        "Other calculations in Thermobar: melt viscosity, redox conversions between fO₂, buffer position and "
        "Fe³⁺/ΣFe, pressure to depth, oxide to element wt%, and comparing your data with model calibration datasets. Choose the function and its inputs in the menu at left.",
        "Also cite the original paper for each model you use; see **About this function**.",
    )
    with st.expander("About this function", expanded=True):
        st.markdown(spec["about"])
        if spec["cite"]:
            st.markdown("Please also cite:\n" + "".join(f"\n* {c}" for c in spec["cite"]))
        st.markdown(f"Worked example in the documentation: [{spec['notebook']}]({OTHER_DOCS}{spec['notebook']}.html)")

    with st.sidebar:
        df_in, note = other_input(spec, name)
    if isinstance(df_in, str):  # typed values
        st.header("Input values")
        df_in = clean(st.data_editor(spec["typed"], num_rows="dynamic", hide_index=True, key=f"{name}_table"))
        st.caption("Edit the values, or add rows at the bottom of the table.")
    elif note:
        st.caption(note)
    cols = [] if df_in is None else [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])]

    with st.sidebar:
        st.header("Settings")
        job = OTHER_FORMS[name](df_in, cols, spec)
        sidebar_about()

    if df_in is None:
        st.info("Upload a CSV or Excel file in the menu at left" + (", or choose **Example data**." if "example" in spec else "."))
        st.stop()
    if df_in.empty:
        st.error("The input has no rows.")
        st.stop()
    if len(df_in) > MAX_ROWS:
        st.error(f"{len(df_in):,} rows exceeds the {MAX_ROWS:,}-row web limit. Use `pip install Thermobar` for larger datasets.")
        st.stop()
    try:
        with st.spinner(f"Calculating {len(df_in):,} rows..."):
            out = job()
    except Exception as e:
        st.error(f"Thermobar could not run this calculation: {type(e).__name__}: {e}")
        st.stop()
    if out is None:  # the calibration plot draws its own results
        return
    res, msgs, printed, key, chart = out
    res = res.loc[:, ~res.columns.duplicated()].reset_index(drop=True)
    show_other(name, res, msgs, printed, key, chart, df_in)


# %% ----------------------------------------------------------------
# sidebar

with st.sidebar:
    tool = st.selectbox("Calculation type", TOOLS)

if tool == OTHER_TOOL:
    other_functions()  # draws its own sidebar and page
    st.stop()

with st.sidebar:
    system = st.selectbox("Which phases?", list(SYSTEMS), index=2)
    spec = SYSTEMS[system]
    phases = spec["phases"]

    st.header("Input data")
    use_example = st.checkbox("Use example data", value=False)
    uploaded, sheet = None, None
    if not use_example:
        uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"],
                                    help="One analysis per row. Column headings like SiO2_Cpx and SiO2_Liq; see Input format.")
        if uploaded is not None and not uploaded.name.lower().endswith(".csv"):
            sheets = pd.ExcelFile(uploaded).sheet_names
            sheet = st.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]
    st.download_button("Download template CSV", template(phases).to_csv(index=False).encode(),
                       file_name=f"Thermobar_template_{system.replace(' ', '_')}.csv", mime="text/csv")

    st.header("Calculation")
    modes = ([SOLVE] if "PT" in spec else []) + ([FIXED_P] if "T" in spec else []) + \
            ([FIXED_T] if "P" in spec else []) + ([SOLVE_TH] if "H" in spec else [])
    mode = st.radio("Calculate", modes, label_visibility="collapsed") if len(modes) > 1 else modes[0]
    eqT = choose("Thermometer", spec, "T") if mode != FIXED_T else None
    eqP = choose("Barometer", spec, "P") if mode in (SOLVE, FIXED_T) else None
    eqH = choose("Hygrometer", spec, "H") if mode == SOLVE_TH else None
    # The inputs that need the file's columns are drawn after it is read, in the placeholder below.
    fixed_box = st.container()

    H2O = None
    if "Liq" in phases and mode != SOLVE_TH:
        if st.checkbox("Use a fixed value of H₂O in the liquid for all analyses", value=False,
                       help="Overrides any H2O_Liq column. Many thermometers and barometers are H₂O-sensitive."):
            H2O = st.number_input("H₂O in liquid (wt%)", 0.0, 20.0, 1.0, step=0.1)
    elif any(eq and needs(spec, k, eq, "H2O_Liq") for k, eq in (("T", eqT), ("P", eqP))):  # a few cpx-only equations
        H2O = st.number_input("H₂O in the melt (wt%)", 0.0, 20.0, 0.0, step=0.1,
                              help="H₂O content of the melt the clinopyroxene grew from. Thermobar assumes 0 if not set.")
    dNNO = None
    if eqP == "P_Kraw2012":
        dNNO = st.number_input("ΔNNO (log units)", -5.0, 5.0, 0.0, step=0.5, help="Krawczynski et al. (2012) needs fO₂.")

    sidebar_about()

# %% ----------------------------------------------------------------
# main

page_header(
    "Thermobarometry, chemometry and hygrometry for liquid, olivine, pyroxene, feldspar and amphibole, "
    "with equilibrium tests, using more than 100 published equations. Choose the phases, equations and "
    "calculation in the menu at left.",
    "Also cite the calibration for each equation (e.g. Putirka 2008 for `T_Put2008_eq33`). "
    "The [Thermobar documentation](https://thermobar.readthedocs.io) lists the reference for every equation.",
)

with st.expander("Input format"):
    names = ", ".join(PHASE_NAMES[p] for p in phases)
    st.markdown(
        f"One analysis per row, oxides in wt%, with a suffix naming the phase ({names} for {system.lower()}): "
        + ", ".join(f"`SiO2_{p}`" for p in phases) + ", and so on.\n\n"
        "* Oxides: SiO₂, TiO₂, Al₂O₃, FeOt, MnO, MgO, CaO, Na₂O, K₂O, Cr₂O₃, P₂O₅. Missing oxides are treated as 0.\n"
        "* Iron must be total FeO, as `FeOt_Cpx` etc. For liquids, `FeO_Liq` and `Fe2O3_Liq` are combined automatically.\n"
        "* Suffixes are case sensitive: `_Cpx`, not `_cpx`.\n"
        + ("* Liquids can have `H2O_Liq` (wt%) and `Fe3Fet_Liq` (Fe³⁺/ΣFe) columns.\n" if "Liq" in phases else "")
        + ("* The Pu et al. (2017, 2021) thermometers also need `NiO_Ol` and `NiO_Liq`.\n" if "Ol" in phases else "")
        + "* Each row is one pair: the phases in a row are assumed to be in equilibrium. Check the equilibrium "
        "tests in the results.\n"
        "* A `Sample_ID` column is optional but recommended. `P_kbar` or `T_K` columns can be used as a set "
        "pressure or temperature. Other columns are ignored."
    )
    st.dataframe(template(phases), hide_index=True)
    st.markdown("To test every combination of separate mineral and liquid analyses (melt matching), use "
                "`pip install Thermobar` and the `_matching` functions; see the documentation.")

with st.expander("Choosing a calculation"):
    st.markdown(
        f"""
* **{SOLVE}**: iterates a thermometer and a barometer until both converge. Use when neither is known.
* **{FIXED_P}** or **{FIXED_T}**: runs one equation at a pressure or temperature you set, as a single value
  or from a column in your file. The P or T input only appears when the chosen equation depends on it.
* **{SOLVE_TH}** (plagioclase-liquid): iterates a thermometer and a hygrometer at a set pressure.
""" + (
"""* **Machine-learning models** for clinopyroxene (Petrelli et al. 2020, Jorgenson et al. 2022; names ending in `_onnx`)
  are extra-trees models that take no P or T input. They return one value per analysis, without the voting spread
  of the pickled versions in the Python package.
""" if HAS_ML else
"""* Machine-learning models (Petrelli et al. 2020, Jorgenson et al. 2022) need the separate Thermobar_onnx
  package, which is not installed here.
""")
    )

# %% ----------------------------------------------------------------
# read input and run

df_in = None
if use_example:
    path, ex_sheet = spec["example"]
    df_in = clean(pd.read_excel(EXAMPLES / path, sheet_name=ex_sheet))
    st.caption(f"Example data: `docs/Examples/{path}`, sheet `{ex_sheet}`.")
elif uploaded is not None:
    try:
        df_in = read_upload(uploaded, sheet)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

numeric_cols = [] if df_in is None else [c for c in df_in.columns if pd.api.types.is_numeric_dtype(df_in[c])
                                         and not any(c.endswith("_" + ph) for ph in PHASE_NAMES)]  # not compositions
P = T = None
with fixed_box:
    # Equilibrium tests need P and T even when the equation does not, so ask whenever the function runs them.
    if mode == SOLVE_TH or (mode == FIXED_P and (needs(spec, "T", eqT, "P") or has_eq_tests(spec["T"][1]))):
        if mode == FIXED_P and not needs(spec, "T", eqT, "P"):
            st.caption(f"{eqT} does not depend on pressure; this pressure is used only for the equilibrium tests.")
        P = fixed_input("P", numeric_cols, "P")
    elif mode == FIXED_P:
        st.caption(f"{eqT} does not depend on pressure.")
    if mode == FIXED_T and (needs(spec, "P", eqP, "T") or has_eq_tests(spec["P"][1])):
        if not needs(spec, "P", eqP, "T"):
            st.caption(f"{eqP} does not depend on temperature; this temperature is used only for the equilibrium tests.")
        T = fixed_input("T", numeric_cols, "T")
    elif mode == FIXED_T:
        st.caption(f"{eqP} does not depend on temperature.")

if df_in is None:
    st.info("Upload a CSV or Excel file in the menu at left, or tick **Use example data**. See **Input format** above.")
    st.stop()
if df_in.empty:
    st.error("The file has no rows.")
    st.stop()
if len(df_in) > MAX_ROWS:
    st.error(f"{len(df_in):,} rows exceeds the {MAX_ROWS:,}-row web limit. Use `pip install Thermobar` for larger datasets.")
    st.stop()

lower = [c for c in df_in.columns if any(c.endswith("_" + p.lower()) for p in PHASE_NAMES)]
if lower:
    st.warning(f"Column suffixes must be capitalized (`_Cpx`, not `_cpx`). These columns are ignored: {lower}")
missing = [p for p in phases if not any(f"{ox}_{p}" in df_in.columns for ox in OXIDES)]
if missing:
    st.error(f"No {', '.join(PHASE_NAMES[p] for p in missing)} columns found (e.g. `SiO2_{missing[0]}`). "
             f"Columns in your file: {list(df_in.columns)}")
    st.stop()

try:
    with st.spinner(f"Calculating {len(df_in):,} analyses..."):
        res, msgs, printed = run(system, df_in, mode, eqT, eqP, eqH, resolve(P, df_in), resolve(T, df_in), H2O, dNNO)
except Exception as e:
    st.error(f"Thermobar could not run this calculation: {type(e).__name__}: {e}")
    st.stop()

show_results(res, msgs, printed, df_in)
