========================
FAQs and Troubleshooting
========================

IAs with most software packages - Thermobar will contain bugs. However, most emails I get are user errors. So please go through these FAQs first.

If your answer is not found, when you email me make sure you include:
1) the version of thermobar you are using (pt.__version__)
2) Attach your Jupyter notebook and all the files it pulls from. If possible, please simplify your code to just emphasize the cell that isnt working (e.g. I dont need to see all your calculations.
3) Screenshots of what the error was - In python, the most useful info is at the bottom of the error message, so please dont just screenshot the top!

General - For any error
==================================================================
Check what version of Thermobar you are on - if it doesn't match the number here, https://pypi.org/project/Thermobar/, first try upgrading.
Make sure you restart your kernel before you try again.


Thermobar doesn't match the Putirka (2008) spreadsheets
==================================================================
Q: Thermobar doesn't match the number from Putirka (2008) spreadsheets

A: First, double check the spreadsheets are working how you think they are. If you are asking Thermobar to iterate equation 33 with say equation 32b, check the spreadsheet is doing that. By default, the Cpx-Liq spreadsheet uses pressure from Neave and Putirka (2017) to iterate with equation 33, and for 32c uses a therometer from 1996. If you change the cells iterating each other, your results should match thermobar. If you've double checked this and there is still a problem, let me know, if its a real bug in Thermobar i'll buy you a coffee at a conference!

My temperatures are really hot!
================================
Q: Why are my temperatures so hot?

A: Remember, Thermobar outputs temperature in Kelvin, not celcius.


Importing Data Issues
======================

Q: My columns are filled with zeros that you expect to be filled with numbers.
A: Check for special characters, e.g., zeros rather than capital 0s, spaces at the start or end of the word, incorrect phase identifiers (e.g., typos like oxps)


Q: What do I do about H$_2$O?.
A: If you know H2O content, have a column in your input spreadsheet named H2O_Liq. If not, it assumes its 0 wt%. You can then overwrite this in various functions to investigate how much H2O affects your results.