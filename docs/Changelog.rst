================================================
Change Log
================================================

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

