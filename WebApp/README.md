# Thermobar web app

Browser front end for Thermobar: upload a CSV/Excel file of paired mineral and liquid compositions (or use the example data), choose the phases and equations, and get temperatures, pressures, H₂O contents and equilibrium tests, with Excel/CSV downloads. No Python needed for users.

Systems: liquid only, olivine-liquid, clinopyroxene-liquid, clinopyroxene only, orthopyroxene-liquid, orthopyroxene only, two pyroxene, plagioclase-liquid (including T-H₂O iteration), alkali feldspar-liquid, two feldspar, amphibole only, amphibole-liquid. Equation menus are read from Thermobar's own registries, so new equations appear automatically after a release. The machine-learning clinopyroxene models (Petrelli et al. 2020, Jorgenson et al. 2022) are offered in their ONNX form when `Thermobar_onnx` is installed (it is in `requirements.txt`; about 330 MB from GitHub), and hidden otherwise. Melt matching (all mineral-liquid combinations) is left to the Python package.

## Run locally

```
pip install -r webapp/requirements.txt
streamlit run webapp/streamlit_app.py
```

## Deploy (free)

1. Push to GitHub.
2. Go to https://share.streamlit.io, sign in with GitHub, and create an app.
3. Repository, branch `main`, main file path `webapp/streamlit_app.py`.

The example data come from `docs/Examples` in this repository. The app installs Thermobar from PyPI (pinned in `requirements.txt`); after a new release, bump the pin and push.
