These workbooks and excel spreadsheets benchmark Thermobar to existing tools for Cpx-only and Cpx-Liquid thermobarometry. 
Small discrepencies (on 3rd -4th decimal place) reflect the fact that many studies use slightly different molar masses. 

Wang_2021_Cpx_only.ipynb compares the results of Thermobar to the supporting spreadsheet provided by the authors (Wang2021_SupportingCalculator.xlsx)

Cpx_only_Putirka2008_Benchmark.ipynb compares the results of Thermobar to the supporting spreadsheet of Putirka (2008), 
results from his spreadsheet are pasted into columns in Python_Liq_CpxBarometers_Test.xlsx

Petrelli_Benchmark.ipynb compares results from Thermobar to the 3 supporting .py files from Petrelli (2020), pasted into GlobalDataset_Final_rev9_TrainValidation.xlsx

Cpx_Liq_Benchmark_Putirka_Neave_Brugman_Masotta.ipynb compares results from Thermobar to Cpx-Liq thermobarometers of Masotta et al. (2013), Putirka (2008) 
and his previous work in the spreadsheets on the Fresno state website,  and the results from Neave and Putirka (2017) Supporting Information. 
columns from various existing spreadsheets pasted into Python_Liq_CpxBarometers_Test.xlsx. 

 Melt_Matching_Benchmark_toNeave.ipynb compares results of our melt matching algorithm and equilibrium tests to the supporting spreadsheet of Neave et al. (2019).
Reads from Melt_Matching_Example_Neave.xlsx

The folder Petrelli_2020_MachineLearning does the calibration of the scalar and decision trees needed to get these expressions into thermobar. 