This collection of codes is for you to create your own random forest model
We suggest you only do this if you have specific chemical/pressure filters you absolutely need or 
if you have a new calibration dataset. 

As a note: all files used in these scripts must be kept in the same folder. 

The work flow goes like this:
1 - Preprocessing - this calculates the cations
2 - Filtering         - this creates filters for cations, kd (Fe-Mg), P ranges etc. If you want a compositional filter we suggest you do it here
3 - Grid Search	      - this code extracts 200 test/ train data sets which you will use in #4
4 - Determine SEE     - this runs 200 iterations of the model (with the 200 test/train data sets) which you can extract the mean (or modal) SEE from
5 - Final Model Train - this creates the final model you will use from all the calibration data, it has the SEE you determined in 4
6 - Filter your data  - this is similar to script 1 and 2 but is designed for user data
7 - Run the model     - this calls in the model you created in 5 and allows you to put in your data and get PT estimates. Please note your data should already be filtered for poor cations totals (you can use code #6 for this)

Good luck!
-CJ and OH 