[![PyPI](https://badgen.net/pypi/v/Thermobar)](https://pypi.org/project/Thermobar/)
[![Build Status](https://github.com/PennyWieser/Thermobar/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/PennyWieser/Thermobar/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/PennyWieser/Thermobar/branch/main/graph/badge.svg)](https://codecov.io/gh/PennyWieser/Thermobar/branch/main)

Thermobar is a python tool for thermobarometry, chemometry and mineral equilibrium.

Thermobar is written in the open-source language Python3. 
Thermobar allows users to easily choose between more than 100 popular parameterizations involving liquid, olivine-liquid, olivine-spinel, 
pyroxene only, pyroxene-liquid, two pyroxene, feldspar-liquid, two feldspar, amphibole and amphibole-liquid, garnet and biotite equilibrium. 

Thermobar is also the first open-source tool for assessing equilibrium, and calculating pressures and temperatures for all possible pairs of phases
 from a given sample/volcanic center (e.g., clinopyroxene-liquid, orthopyroxene-liquid, two-pyroxene, feldspar-liquid, two feldspar, amphibole-liquid).
Thermobar also contains a number of functions allowing users to propagate errors using Monte-Carlo methods, plot mineral classification diagrams 
and assess mineral-melt equilibrium (e.g. olivine-melt Rhodes diagrams), calculate liquid viscosities, and convert between different measures for 
oxygen fugacity and Fe speciation. Finally, in order to perform its calculations, Thermobar contains a number of functions 
for calculating molar and cation fractions, cation site allocations, and mineral components. These can be leveraged alongside various statistical 
and machine learning packages in Python to easily produce new thermobarometry, hygrometry or chemometry calibration. 
Thermobar can be downloaded via pip, on Github (you are here!), and extensive documentation and 
example videos and Jupyter Notebooks are available at https://thermobar.readthedocs.io/en/latest/index.html

We hope to submit Thermobar in the fall, for now please cite my GSA abstract:
Penny Wieser, Maurizio Petrelli, Jordan Lubbers, Eric Wieser, Adam Kent, Christy Till. 
Thermobar: A critical evaluation of mineral-melt thermobarometry and hygrometry in arc magmas using a new open-source Python3 tool.
Geological Society of America Abstracts with Programs. Vol 53, No. 6. https://doi.org/10.1130/abs/2021AM-367080.

________________________________
Want your model in Thermobar?
________________________________
Getting your model into Thermobar will hopefully help to increase usage. 
I am happy to help you with this. You will need to supply me with your scripts or excel spreadsheet showing how the model works, 
your calibration dataset, and some example calculations for benchmarking. 

For Machine Learning models, please see the read the docs page. 