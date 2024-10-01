================================================
Change Log
================================================
Version 1.0.47 - Sept 30th, 2024
=====================================
Fixed Pu et al. so it works for integers, floats and series. Also put in filter so if one of the log results in Nan, it returns Nan, rather than killing the function.

Version 1.0.46 - Aug 11, 2024
=====================================
Fixed some pandas future warnings

Version 1.0.45 - July 11th, 2024
=====================================
Based on issue from Jordan lubbers, changed import_excel function to allow you to supply a path using pathlib. Added as dependency as well. Old path functionality should be preserved.


Version 1.0.43 - June 29th 2024
=====================================
Added functionality for melt matching for Pu et al (upon user request). Also, updated FAQs.


Version 1.0.42 - June 20th 2024
=====================================
Fixed Pu et al. (2017) and (2021) thermometers - bug in NiO allocation that returned error.



Version 1.0.37 - Feb 26th, 2024
=====================================
For cpx-only press-temp used to only return Cpx composition if eq_tests=True, now also return Cpx components (e.g. Jd, Mg#, etc) for ease of plotting and workflows subdividing by textural types


Version 1.0.36 - Jan 21st, 2024
=====================================
After numerous issues with peoples old (2020-2021 sklearn issues, have required 1.3 onwards in setup.py file now. )
This meant that had to change python to be 3.7+, as skitlearn only supports 3.8-3.11. People with older installations can pip install earlier versions



Version 1.0.31 to Version 1.0.34 - Dec, 2023
=====================================
Fixing additional issues related to pandas2 and shift to csv file paths.


Version 1.0.31 - November 17th, 2023
=====================================
Sorted issues with .pkl incompatability for calibration datasets by resaving all as csv files and loading those.

Version 1.0.31 - November 17th, 2023
=====================================
Sorted issues with .pkl incompatability for calibration datasets by resaving all as csv files and loading those.

Version 1.0.30 - October 6th, 2023
=====================================
Issue with Amp-Liq melt mathcing, particularly if non unique values for liqiud names. Fixed so dont end up with a duplicate column name that errors out.


Version 1.0.28 - October 6th, 2023
=====================================
Saved pkl files for Jorgenson and Petrelli were failing with Sklearn 1.3. retrained models using this version + released new version on github for Thermobar_onnx.
Sklearn also changed 'mse' to now be called 'squared_error'
For Cpx_all functions, as doesnt do voting anyway, swapped to onnx versions, so at least those work when people have wrong versions.

================================================
Version 1.0.27 - Augst 23rd, 2023
=====================================
Fixed bug with Jorgenson - was doing normalizatoin using the Sample_ID_Liq_num column for MonteCarlo simulations, resulted in liquid getting less and less data as you go to higher iterations. Added if statement
to drop column before normalization stage.


Version 1.0.25 - July 6th, 2023
==============================
Added CHOMPI from Blundy (2022)
function calculate_CHOMPI


Version 1.0.20 Mayn 30th, 2023
==============================
Cpx matching function thros error if one column all Nans, have done a check if column there first. Thanks Divya!



Version 1.0.19 April 1st, 2023
==============================
Raises warning if people enter non equal length dataframes into functions which dont melt match. Add melt match for plag-fspar (Thanks Jordan!)


Version 1.0.18 April 1st, 2023
==============================
Support for non numerical sample names for Opx-Cpx, Opx-Liq matching, Fspar-Liq, Amp-Liq.

Version 1.0.16 April 1st, 2023
==============================
Added ability to do confidence intervals in matplotlib

Version 1.0.14 March 13th, 2023
================================
Felix boschetty showed that viscosity doesnt work if sample name is a string. Have changed to drop the sample column during the norm step.

Conversion between fo2 and melt redox only used high T QFM value. Have now changed to allocate based on entered T_K


Version 1.0.12, Feb 13th, 2023
================================
If a user entered Kspar comps into plag-liq functions, returned Nan for eq tests, broke loop. Have fixed to add warning if you enter An<0.05, and also to not return yes or no if T is Nan
Have also updated examples to get users to plot input fspar on a ternary diagram.

Version 1.0.11, Feb 11th, 2023
================================
Fixed bug, in old version, T<1300 K, code returned 'Pass' for An-Ab in plag-liq, even if fail. Kd value was correct, just string was wrong. Thanks to Bryant platt for spotting this one.

Version 0.12 Nov 23rd, 2021
================================

Added in capability for Plag-Liq, Kspar-Liq temperature calculations for melt matching,
and Plag-Liq hygrometers.

Added in warning if users specify Suffix, but already have the suffix.

Version 0.11 Nov 15th, 2021
================================

Fixed issue with indent on Opx-Liq melt matching.


Previous changes before detailed change log started:
====================================================
Removed Wang et al. (2021) barometer eq. 3 and thermometer eq 2, updated
eq 1 and 2 to reflect those in the final version of the manuscript.

Changed order of Cpx-Liq oxides when training Petrelli (2020) to get perfect benchmark

