============
FAQs
============

Importing Data
============

Q: Columns are filled with zeros that you expect to be filled with numbers. A: Check for special characters, e.g., zeros rather than capital 0s, spaces at the start or end of the word, incorrect phase identifiers (e.g., typos like oxps)

Q: What do I do about H$_2$O?. A: If you know H2O content, have a column in your input spreadsheet named H2O_Liq. If not, it assumes its 0 wt%. You can then overwrite this in various functions to investigate how much H2O affects your results.