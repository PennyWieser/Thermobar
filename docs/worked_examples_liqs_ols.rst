================
Worked examples
================

This page summarizes the different examples available. If you want a specific example, we can always include more. 

Liquids and Olivines
-----------------------------------

This :doc:`example <Examples/Liquid_Ol_Liq_Themometry/Liquid_only_Thermometry>` shows various options for calculating liquid-only temperatures, 
as well as sensitivity to temperature and pressure. 

This :doc:`example <Examples/Liquid_Ol_Liq_Themometry/Olivine_Liquid_thermometry>` shows:
- how to perform olivine-liquid thermometry, 
both for paired olivines and liquids, and showing how to consider all possible pairings. 
- How to calculate equilibrium olivine-liquid
contents for different liquid compositions using a variety of Kd models (Roeder, Matzen, Toplis, Putirka)
- How to plot olivines and liquids on Rhodes diagrams
- How to calculate Fe3/FeT from a buffer position. 

This :doc:`example <Examples/Liquid_Ol_Liq_Themometry/Olivine_Liquid_hygrometry>` shows how to perform olivine-liquid hygrometry calculations.

This :doc:`example <Examples/Liquid_Ol_Liq_Themometry/Olivine_MatrixGlass_Mg_Fe_Eq_MultipleSamples>` shows how to plot a Rhodes diagram 
for a single sample with an olivine population.

This :doc:`example <Examples/Liquid_Ol_Liq_Themometry/Olivine_MatrixGlass_Mg_Fe_Eq_MultipleSamples>` shows how to plot a Rhodes diagram when you have
different matrix glass/whole rock samples, and an olivine population from each sample. 

This :doc:`example <Examples/Liquid_Ol_Liq_Themometry/CHOMPI_Liq_calcs>` shows how to perform CHOMPI calculations following Blundy (2022).

Clinopyroxene (Cpx)
-----------------------------------

This :doc:`example <Examples/Cpx_Cpx_Liq_Thermobarometry/Cpx_Liq_Thermobarometry>` shows how to perform Cpx-Liq and Cpx-only calculations when you have
either just Cpx, or paired Cpx-Liqs. Melt matching is covered below. It also shows how to plot Cpx Liq comps on a Rhodes diagram. 

This :doc:`example <Examples/Cpx_Cpx_Liq_Thermobarometry/MachineLearning_Cpx_Liq_Thermobarometry>` shows how to perform Cpx-Liq and Cpx-only calculations when you have
using the Petrelli and Jorgenson machine learning models. There are two options, using onnx and using the voting method of Jorgenson.
Note, if you use these models, you have to install the .pkl and .onnx files from a zip from github (instructions in notebook) and check your sklearn
installation matches the setup.py (should check automatically but may need manual upgrading).

This :doc:`example <Examples/General_Plotting/Pyroxene_Classification_Kilauea>` shows how to plot a pyroxene ternary diagram for pyroxene
data from Kilauea

This :doc:`example <Examples/Cpx_Cpx_Liq_Thermobarometry/Cpx_Liquid_melt_matching/Cpx_MeltMatch1_Gleeson2020>` is a simple introduction to 
melt matching. It shows how to combine all possible liquid and cpx compositions using various equilibrium tests. 

This :doc:`example <Examples/Cpx_Cpx_Liq_Thermobarometry/Cpx_Liquid_melt_matching/Cpx_MeltMatch2_ScruggsPutirka2018>` is a more complex example
following the method of Scruggs and Putirka (2018) - if you dont have many liquids (perhaps just mafic and silicic end members) it shows how to make synthetic liquids.

Orthopyroxene (Opx)
-----------------------------------
This :doc:`example <Examples/Opx_and_Opx_Liq_Thermobarometry/Pyroxene_Ternary_Opx_Example>` shows how to plot pyroxene compositions on a ternary diagram

This :doc:`example <Examples/Opx_and_Opx_Liq_Thermobarometry/Opx_Liq_Matching>` shows how to consider matches between all possible liq-opx pairs. 

Two Pyroxene (Opx-Cpx)
-----------------------------------
This :doc:`example <Examples/Two_Pyroxene_Thermobarometry/Two_Pyroxene_Thermobarometry>` shows how to perform Opx-Cpx calcs on pre-matched pairs. 

This :doc:`example <Examples/Two_Pyroxene_Thermobarometry/Two_Pyroxene_Matching>` shows how to consider all possible Cpx-Opx pairs using equilibrium tests to calculate P and T.

Amphiboles
-----------------------------------
This :doc:`example <Examples/Amphibole/Amphibole_Thermobarometry_Chemometry>` shows how to perform Amp-only and Amp-Liq
thermobarometry and chemometry calculations. 

This :doc:`example <Examples/Amphibole/Amp_Liq_Melt_Matching>` shows how to do Amp-Liq melt matching.

This :doc:`example <Examples/Amphibole/Amphibole_Classification_Diagrams>` shows how to plot amphibole classification diagrams.


Feldspars
-----------------------------------
This :doc:`example <Examples/Feldspar_Thermobarometry/Feldspar_Liquid_Thermobarometry>` shows how to perform feldspar-liquid thermobarometry.

This :doc:`example <Examples/Feldspar_Thermobarometry/Feldspar_Liquid_Thermobarometry>` shows how to perform plagioclase-liquid hygrometry calculations.


This :doc:`example <Examples/Feldspar_Thermobarometry/Two_Feldspar_All_Possible_Pairs> shows how to perform two feldspar thermobarometry. 


This :doc:`example <Examples/Feldspar_Thermobarometry/Fspar_Liq_Matching> shows how to perform Kspar-liq and plag-liq melt matching. 


This :doc:`example <Examples/Feldspar_Thermobarometry/Fspar_Ternary_Plot> shows how to plot a feldspar ternary diagram.

This :doc:`example <Examples/General_Plotting/Plagioclase_Classification_Kilauea> shows how to plot a plagioclase ternary diagram segmented by sample. 



Garnet
-----------------------------------
This :doc:`example <Examples/Garnet_Geotherms/Garnet_Functions> shows how to perform garnet thermobarometry and plot a garnet compositional section with a geotherm etc. 

This :doc:`example <Examples/Garnet_Geotherms/Geotherm_functions> shows how to calculate a garnet geotherm




Plotting Mineral classification diagrams. 
-----------------------------------
This :doc:`example <Examples/Opx_and_Opx_Liq_Thermobarometry/Pyroxene_Ternary_Opx_Example.ipynb>` shows how to plot pyroxene compositions on a ternary diagram

This :doc:`example <Examples/General_Plotting/Pyroxene_Classification_Kilauea>` shows how to plot a pyroxene ternary diagram for pyroxene data from Kilauea

This :doc:`example <Examples/Amphibole/Amphibole_Classification_Diagrams>` shows how to plot amphibole classification diagrams.


This :doc:`example <Examples/Feldspar_Thermobarometry/Fspar_Ternary_Plot> shows how to plot a feldspar ternary diagram.

This :doc:`example <Examples/General_Plotting/Plagioclase_Classification_Kilauea> shows how to plot a plagioclase ternary diagram segmented by sample. 


Error Propagation
-----------------------------------




