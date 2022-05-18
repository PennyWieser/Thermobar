==============================
Adding new models
==============================

Getting your model into Thermobar will hopefully help to increase usage, and citations

I am happy to help you with this. You will need to supply me with your scripts or excel spreadsheet showing how the model works,
your calibration dataset, and some example calculations for benchmarking.

For Machine Learning models, there are lots of complications about how to distribute models in a reproducable way.

One option is to save the regressors and scalars as .pkl files (pickles). The main problem with this is, overtime, as the version
of sklearn/scipy changes, you get different answers within the standard error of the model. You also start to get depreciation warnings.

A second option is to make machine learning pipelines, and release them as .onnx files.
These are resistant to change, but they are also large! This means I already hit the 100 MB PyPI file size limit
(when they say file size, they mean release size, not the individual .pkl, .onnx files ).
I have requested an extension up to 500 MB which accomadates the inclusion of Jorgenson and Petrelli models.
However, I will eventually hit the 10GB project limit(sum of all releases). Deleting releases is not ideal, as it
means there is less reproducability.

Thus, the plan now on is that creaters of models will be responsible for releasing pkl and onnx files for their model on PyPI as separate packages.
I will then produce code for Thermobar to download thes pkl and onnx files, and integrate them into existing functions.









