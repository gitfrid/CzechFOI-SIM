### CzechFOI-SIM 
**Czech FOI Simulation Analysis** 
<br>
<br>**Investigates whether there is a reliable statistical way to determine the dAEFI rate when the baseline is unknown (real world).**
<br>
Simulates dAEFIs to analyse the impact on the curve and back-calculate the dAEFIs rate (comparing known and unknown baseline).
Uses real Czech FOI (Freedom of Information) data, or generates d, dvx, duvx data in modulated sine wave form.

Simulated Date can be used to check for calculation errors in your code, it is possible to create a CSV file with the data of all Plot curves (from day 1-1534).

The [Python Scripts](https://github.com/gitfrid/CzechFOI-SIM/tree/main/Py%20Scripts) process and visualize CSV data from the [TERRA folder](https://github.com/gitfrid/CzechFOI-SIM/tree/main/TERRA), generating interactive HTML plots. <br>Each plot compares two age groups. To interact with the plots, click on a legend entry to show/hide curves.

Download the processed plots for analysis from the [Plot Results Folder](https://github.com/gitfrid/CzechFOI-SIM/tree/main/Plot%20Results/dAEFI). Or simply adapt and run the [Python script](https://github.com/gitfrid/CzechFOI-SIM/blob/main/Py%20Scripts/AB%29%20backcalc%20dAEFI%20simulation.py) to meet your own analysis requirements!

Dates are counted as the number of days since [January 1, 2020](https://github.com/gitfrid/CzechFOI-SIM/blob/main/Plot%20Results/Days%20to%20Date%20Translation%20Day%20Date%20Translation/Days%20to%20Date%20Translation%20Day%20Date%20Translation.png), for easier processing. "AGE_2023" represents age on January 1, 2023. <br>The data can optionally be normalized per 100,000 for comparison.

Access the original Czech FOI data from a [Freedom of Information request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz). To learn how the Pivot CSV files in the TERRA folder were created, see the [wiki](https://github.com/gitfrid/CzechFOI-DA/wiki)

<br>**Abbreviations:** The figures are per age group from the CSV files in the TERRA folder:
| **Deaths**        | **Definition**                                             | **Population/Doses**  | **Definition**                                        |
|-------------------|------------------------------------------------------------|-----------------------|-------------------------------------------------------|
| NUM_D             | Number deaths                                              | NUM_POP               | Total people                                          |
| NUM_DUVX          | Number unvaxed deaths                                      | NUM_UVX               | Number of unvaxed people                              |
| NUM_DVX           | Number vaxed deaths                                        | NUM_VX                | Number of vaxed people                                |
| NUM_DVD1-DVD7     | Number deaths doses 1 - 7                                  | NUM_VD1-VD7           | Number of vax doses 1 - 7                             |
| NUM_DVDA          | Number deaths from all doses                               | NUM_VDA               | Total number of all vax doses (sum)                   |
| dAEFI             | simulated death Adverse Events following imun.             |                       |                                                       |
<br>

_________________________________________
**dAEFI simulation known Basline. <br>One dAEFI per 5000 Doses RAND_DAY_RANGE 1-250 AVG_WND 14: AG_50-54**
<br>

<img src=https://github.com/gitfrid/CzechFOI-SIM/blob/main/Plot%20Results/dAEFI/AB)%20backcalc%20dAEFI%20simulation%20known%20basline%20DAY_RNG_250%20WD_14%20%20AG_50-54.png width="1280" height="auto">
<br>

**If the baseline is known (which is not the case in practice), the estimated dAEFIs per dose are quite accurate, e.g., 4408 vs. 5000.** .

_________________________________________
**dAEFI simulation known Basline. <br>One dAEFI per 5000 Doses RAND_DAY_RANGE 1-250 AVG_WND 14: AG_75-79**
<br>

<img src=https://github.com/gitfrid/CzechFOI-SIM/blob/main/Plot%20Results/dAEFI/AB)%20backcalc%20dAEFI%20simulation%20known%20basline%20DAY_RNG_250%20WD_14%20%20AG_75-79.png width="1280" height="auto">
<br>

**The estimated dAEFIs per dose, e.g., 4179 vs. 5000.** .
_________________________________________
**dAEFI simulation unknown Basline real world. <br>One dAEFI per 5000 Doses RAND_DAY_RANGE 1-250 AVG_WND 14: AG_50-54**
<br>

<img src=https://github.com/gitfrid/CzechFOI-SIM/blob/main/Plot%20Results/dAEFI/AB)%20backcalc%20dAEFI%20simulation%20unknown%20real%20world%20basline%20DAY_RNG_250%20WD_14%20%20AG_50-54.png width="1280" height="auto">
<br>

**If the baseline is unknown (which is the case in practice), the estimated dAEFI per dose are not reliable , e.g., 136 vs. 5000.** .

_________________________________________
**dAEFI simulation known Basline. <br>One dAEFI per 5000 Doses RAND_DAY_RANGE 1-250 AVG_WND 14: AG_75-79**
<br>

<img src=https://github.com/gitfrid/CzechFOI-SIM/blob/main/Plot%20Results/dAEFI/AB)%20backcalc%20dAEFI%20simulation%20unknown%20real%20world%20basline%20DAY_RNG_250%20WD_14%20%20AG_75-79.png width="1280" height="auto">
<br>

**The estimated dAEFIs per dose, e.g.,  39 vs. 5000.** .
_________________________________________

**D, DVX, DUVX plots added dAEFIs (1/5000 Doses) vs non AEFIs: AG_50-54 vs 75-79**
<br>
<br>
**As you can see, the added dAEFIs have little impact on the top D-curves for age group 75-79, making it hard to detect a signal without knowing the baseline.
I struggled to find a reliable method to back-calculate the dAEFIs ratio using only the moving average as the baseline (real world). This is particularly true for the older age groups.**
<br>

<img src=https://github.com/gitfrid/CzechFOI-SIM/blob/main/Plot%20Results/dAEFI/AB)%20backcalc%20dAEFI%20simulation%20d-duvx-dvx%20known%20basline%20DAY_RNG_250%20WD_14%20%20AG_50-54.png width="1280" height="auto">
<br>
<img src=https://github.com/gitfrid/CzechFOI-SIM/blob/main/Plot%20Results/dAEFI/AB)%20backcalc%20dAEFI%20simulation%20d-duvx-dvx%20known%20basline%20DAY_RNG_250%20WD_14%20%20AG_75-79.png width="1280" height="auto">
<br>
_________________________________________

### Software Requirements:
- [Python 3.12.5](https://www.python.org/downloads/) to run the scripts.
- [Visual Studio Code 1.92.2](https://code.visualstudio.com/download) to edit and run scripts.
- [Optional - DB Browser for SQLite 3.13.0](https://sqlitebrowser.org/dl/) for database creation, SQL queries, and CSV export.

### Disclaimer:
**The results have not been checked for errors. Neither methodological nor technical checks or data cleansing have been performed.**
