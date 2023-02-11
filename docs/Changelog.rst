================================================
Change Log
================================================

Version 1.0.11, Feb 22th, 2023
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

