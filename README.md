### CzechFOI-DA

**Czech FOI Simulation Analysis** 
<br>
<br>**Investigates whether there is a reliable statistical way to determine the dAEFI rate when the baseline is unknown (ral world).**
<br>Simulates dAEFIs to analyse the impact on the curve and back-calculate the dAEFIs rate (comparing known and unknown baseline).
<br>Uses real Czech FOI (Freedom of Information) data, or generates d, dvx, duvx data in modulated sine wave form.
Requires some IT knowledge.

The [Python Scripts](https://github.com/gitfrid/CzechFOI-DA/tree/main/Py%20Scripts) process and visualize CSV data from the [TERRA folder](https://github.com/gitfrid/CzechFOI-DA/tree/main/TERRA), generating interactive HTML plots. <br>Each plot compares two age groups. To interact with the plots, click on a legend entry to show/hide curves.

Download the processed plots for analysis from the [Plot Results Folder](https://github.com/gitfrid/CzechFOI-DA/tree/main/Plot%20Results). Or simply adapt and run the [Python scripts](https://github.com/gitfrid/CzechFOI-DA/blob/main/Py%20Scripts/AH%29%202D%206-Axis%20age-compare%20rolling-mean%20significance-1D-2D%20different-scale.py) to meet your own analysis requirements!

Dates are counted as the number of days since [January 1, 2020](https://github.com/gitfrid/CzechFOI-DA/blob/main/Plot%20Results/Days%20to%20Date%20Translation%20Day%20Date%20Translation/Days%20to%20Date%20Translation%20Day%20Date%20Translation.png), for easier processing. "AGE_2023" represents age on January 1, 2023. <br>The data can optionally be normalized per 100,000 for comparison.

Access the original Czech FOI data from a [Freedom of Information request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz). To learn how the Pivot CSV files in the TERRA folder were created, see the [wiki](https://github.com/gitfrid/CzechFOI-DA/wiki)

<br>**Abbreviations:** The figures are per age group from the CSV files in the TERRA folder:
| **Deaths**        | **Definition**                                                       | **Population/Doses**  | **Definition**                                        |
|-------------------|----------------------------------------------------------------------|-----------------------|-------------------------------------------------------|
| NUM_D             | Number deaths                                                        | NUM_POP               | Total people                                          |
| NUM_DUVX          | Number unvaxed deaths                                                | NUM_UVX               | Number of unvaxed people                              |
| NUM_DVX           | Number vaxed deaths                                                  | NUM_VX                | Number of vaxed people                                |
| NUM_DVD1-DVD7     | Number deaths doses 1 - 7                                            | NUM_VD1-VD7           | Number of vax doses 1 - 7                             |
| NUM_DVDA          | Number deaths from all doses                                         | NUM_VDA               | Total number of all vax doses (sum)                   |
| dAEFI             | simulated death AdversEvents following imun.                         |                       |                                                       |
<br>

_________________________________________
**Interactive html plot simulation with the Czech FOI data. <br>Age group comparison simulation of dAEFI : AG_70-74 vs 75-79**
<br>

<img src=https://github.com/gitfrid/CzechFOI-DA/blob/main/Plot%20Results/AH)%202D%206-Axis%20age-compare%20rolling-mean%20significance-1D-2D%20same-scale%201D2D-MEAN%20POP-D%20N-CUM-D%20N%20AVG_30%20CORR_300/AH)%202D%206-Axis%20age-compare%20rolling-mean%20significance-1D-2D%20same-scale%201D2D-MEAN%20POP-D%20N-CUM-D%20N%20AVG_30%20CORR_300%20AG_70-74%20vs%2075-79.png width="1280" height="auto">
<br>

**The first derivative represents speed** because it measures how fast something changes over time.
**The second derivative represents acceleration** as it measures how fast the speed (or first derivative) changes over time, showing how quickly the speed is increasing or decreasing.

_________________________________________
**: AG_70-74 vs 75-79**
<br>

<img src= width="1280" height="auto">
<br>

The py script ending with **same-scale** uses the same y-axis scale for both age groups. Use this version to compare similar age groups. The file ending with **different-scale** uses different y-axes with different scales for each age group.
_________________________________________

**: AG_50-54 vs 75-79**
<br>Decay Time - calculates the number of days per day retroactively after a certain percentage has died
<br>

<img src= width="1280" height="auto">
<br>
_________________________________________

### Software Requirements:
- [Python 3.12.5](https://www.python.org/downloads/) to run the scripts.
- [Visual Studio Code 1.92.2](https://code.visualstudio.com/download) to edit and run scripts.
- [Optional - DB Browser for SQLite 3.13.0](https://sqlitebrowser.org/dl/) for database creation, SQL queries, and CSV export.

### Disclaimer:
**The results have not been checked for errors. Neither methodological nor technical checks or data cleansing have been performed.**
