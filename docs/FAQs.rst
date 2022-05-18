========================
FAQs and Troubleshooting
========================

It is very possible that there are errors in Thermobar., and I really appreciate people alerting me to them. However, I get an awful lot of emails with the same problems, so please have a look through these FAQs first!

If your problem isnt covered here, it would be great if you could raise an issue on GitHub, as it might help future people with the same question.
https://github.com/PennyWieser/Thermobar/issues
Select "New Issue"
Make sure you give me enough information to troubleshoot- E.g., attach your .ipynb notebook with the problem, and some input data, screenshots of error messsages etc. If you have a problem of Thermobar vs. an existing tool, attach the other spreadsheet you are comparing it too.


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

Q: Columns are filled with zeros that you expect to be filled with numbers.

A: Check for special characters, e.g., zeros rather than capital 0s, spaces at the start or end of the word, incorrect phase identifiers (e.g., typos like oxps)

Q: What do I do about H$_2$O?.

A: If you know H2O content, have a column in your input spreadsheet named H2O_Liq. If not, it assumes its 0 wt%. You can then overwrite this in various functions to investigate how much H2O affects your results.