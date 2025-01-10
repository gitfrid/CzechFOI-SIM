import pandas as pd
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
import numpy as np
from matplotlib import colors as mcolors 
import colorsys
import random
import math
import plotly.io as pio
import csv
import xarray as xr
import scipy.stats as stats
import copy

# As far as I know, 
# this (vital) problem is still waiting for the head that can solve it?

# The goal is to find a reliable method 
# for backcalculating the estimated dAEFIs 
# If the baseline is unknown (real world)
    
# This script processes data from pivot CSV files located in the TERRA folder, 
# which were generated from a Czech Freedom of Information request (Vesely_106_202403141131.csv). 
# The pivot files were created using DB Browser for SQLite.

# The script can simulate adding dAEFIs (deadly adverse events following immunization).
# It can also simulate modulated sine waves for d, duvx, dvx, and dvda (deaths from all doses).
# The script calculates the differences in doses and deaths for similar age bands, 
# as specified in the `age_band_compare` list.


plot_title_text="Sum of the differences between age groups of D-curves and doses, calculated for age groups from 15 to 85, including rolling and shift correlation" # Title of plot
annotation_text ="AG) calc dAEFI diff all-agegrp-in-same-plot.py / Curves without dAEFIS -vs- Curves with simulated dAEFIs / D-curves not normalized "
plotfile_name = "AG) calc dAEFI diff all-agegrp-in-same-plot" 
plot_name_append_text="dAEFI"   # apend text - to plotfile name and directory, to save in uniqe file location
normalize=False                 # normalize dvd values
normalize_cumulate_deaths=False # normalize cumulated deaths bevore cummulation
population_minus_death=True     # deducts the deceased from the total population
custom_legend_column = 10       # A change could require adjustments to the code. 
axis_y1_logrithmic=True         # show logarithmic y1 axis
savetraces_to_csv=False         # save calcualted results of all traces into a csv file
window_size_mov_average=30      # Define the window size for the rolling average (adjust this as needed)
calc_correlation=True           # clacualte and plot rolling and phase shift correlation
window_size_correl=280          # window size for rolling pearson correlation
max_phase_shift=300             # max phase shift for phase shift correlation
plot_trace=False                # bool - if true (default) trace is saved into the html file 


# Define the event threshold (1 in 10,000 VDA events triggers an event)
event_threshold = 5000          # Trigger simulates one  dAEFI (deadly adverse event following immunisation) per 5,000 VDA (vax dose all)
future_day_range = 250          # Random future day range (for example 1 to 14)
window_size = 14                # Define the window size for the rolling average dAEFI (adjust this as needed)

# simulation behavior
simulate_sinus=False            # uses modulated sin wave to simulate Death curve
simulate_proportinal_norm=False # simulate constant death curve adjusted to (uvx, vx , total) population (use real data or sin wave for each dey 1..1534)
simulate_dAEFI=False
real_world_baseline=True


def main():
    global dae

    # just for debugging - is faster
    dbg_age_band_compare = [
        ('15', '25'),
        ('25', '35'),
        ('35', '45'),
        ('45', '55'),
        ('55', '65'),
        ('65', '75'),    
    ]

    # List of tuples with the age bands you want to compare 
    # Uses Data from CSV-Files in TERRA folder.
    # SQL-Query "C:\CzechFOI-SIM\SQLQueries\All AG SQL Time.sql"  was used
    # to create the views in czech SQLite-DB and export them to CSV-Files in TERRA folder    
    age_band_compare = [ # for debugging
        ('15', '16'),
        ('16', '17'),
        ('17', '18'),
        ('18', '19'),
        ('19', '20'),
        ('20', '21'),
        ('21', '22'),
        ('22', '23'),
        ('23', '24'),
        ('24', '25'),
        ('25', '26'),
        ('26', '27'),
        ('27', '28'),
        ('28', '29'),
        ('29', '30'),
        ('30', '31'),
        ('31', '32'),
        ('32', '33'),
        ('33', '34'),
        ('34', '35'),
        ('35', '36'),
        ('36', '37'),
        ('37', '38'),
        ('38', '39'),
        ('39', '40'),
        ('40', '41'),
        ('41', '42'),
        ('42', '43'),
        ('43', '44'),
        ('44', '45'),
        ('45', '46'),
        ('46', '47'),
        ('47', '48'),
        ('48', '49'),
        ('49', '50'),
        ('50', '51'),
        ('51', '52'),
        ('52', '53'),
        ('53', '54'),
        ('54', '55'),
        ('55', '56'),
        ('56', '57'),
        ('57', '58'),
        ('58', '59'),
        ('59', '60'),
        ('60', '61'),
        ('61', '62'),
        ('62', '63'),
        ('63', '64'),
        ('64', '65'),
        ('65', '66'),
        ('66', '67'),
        ('67', '68'),
        ('68', '69'),
        ('69', '70'),
        ('70', '71'),
        ('71', '72'),
        ('72', '73'),
        ('73', '74'),
        ('74', '75'),
        ('75', '76'),
        ('76', '77'),
        ('77', '78'),
        ('78', '79'),
        ('79', '80'),
        ('80', '81'),
        ('81', '82'),
        ('82', '83'),
        ('83', '84'),
        ('84', '85')
    ]
    
    
    # CSV file pairs with age_band with death and population/doses data  
    csv_files_dvd = [
            r"C:\CzechFOI-SIM\TERRA\PVT_NUM_D.csv",
            r"C:\CzechFOI-SIM\TERRA\PVT_NUM_DUVX.csv",
            r"C:\CzechFOI-SIM\TERRA\PVT_NUM_DVX.csv",
            r"C:\CzechFOI-SIM\TERRA\PVT_NUM_DVDA.csv",
        ]

    csv_files_vd = [
            r"C:\CzechFOI-SIM\TERRA\PVT_NUM_POP.csv",
            r"C:\CzechFOI-SIM\TERRA\PVT_NUM_UVX.csv",
            r"C:\CzechFOI-SIM\TERRA\PVT_NUM_VX.csv",
            r"C:\CzechFOI-SIM\TERRA\PVT_NUM_VDA.csv",
        ]
    
  
    modulated_wave_params_ag1 = [
                    {'small_period': 3, 'large_period': 360, 'small_amplitude': 1, 'large_amplitude': 2, 'vertical_shift': 5, 'horizontal_shift': 0}, # Curve 1 DUVX - AG1
                    {'small_period': 3, 'large_period': 360, 'small_amplitude': 1, 'large_amplitude': 2, 'vertical_shift': 5, 'horizontal_shift': 0}, # Curve 2 DVX - AG1
        ]  
    
    # is not used
    modulated_wave_params_ag2 = [
                    {'small_period': 3, 'large_period': 360, 'small_amplitude': 5, 'large_amplitude': 20, 'vertical_shift': 50, 'horizontal_shift': 0}, # Curve 1 DUVX - AG2
                    {'small_period': 3, 'large_period': 360, 'small_amplitude': 5, 'large_amplitude': 20, 'vertical_shift': 50, 'horizontal_shift': 0}, # Curve 2 DVX - AG2
        ]

      
    try:
        dataframes_dvd = [pd.read_csv(file) for file in csv_files_dvd]
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")

    try:
        dataframes_vd = [pd.read_csv(file) for file in csv_files_vd]   
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")

    # Get the color shades for the current standard dataset traces (legends)
    # 11 rows x 10 columns
    color_palette = px.colors.qualitative.Dark24
    color_palette_r = px.colors.qualitative.Dark24_r
    # Generate shades for all color pairs
    color_shades = generate_color_shades(color_palette, n_pairs=11)
    color_shades_r = generate_color_shades(color_palette_r, n_pairs=11)
   
    # Create an instance of TraceManager
    trace_manager = TraceManager()   
    
    # loop will run twice (k=0, k=1)
    # k=0 for the RAW-Data,  k=1 for RAW-Data (extension RAW)  with the added (simulated) dAEFI Deaths (extension AEF) 
    for k in range(1):  
            
        # Set dae based on k and normalize
        if k == 0:
            dae = "n RAW" if normalize else "RAW"
            simulate_dAEFI=False
        elif k == 1:
            dae = "n AEF" if normalize else "AEF"
            simulate_dAEFI=True
            

        # Loop through each pair of age bands in the list
        for age_band_pair in age_band_compare:    
    
            for idx, age_band in enumerate(age_band_pair):

                # Use Plotly and Plotly_r color palette with 10 different colors 
                # for the two age groups and plot additional decay and correlation traces
                colors = px.colors.qualitative.Plotly                    
            
                # Define the days range (from 0 to 1530)
                max_days = len(dataframes_vd[0][age_band])
                days = np.linspace(0, max_days-1, max_days) 

                # proportion_uvx, proportion_vx, proportion_vda = calculate_proportions(dataframes_vd, age_band)

                # Generat D, DUVX, DVX, DVDA data as modulated sin curves, by using the two parameter sets 
                sin_curves = generate_modulated_wave(days, modulated_wave_params_ag1)

                # calculate cumulated population
                cum_pop_vd = calculate_cumulative_population(dataframes_vd, dataframes_dvd, age_band, population_minus_death)

                # Loop through each day (time point)           
                proportions, updated_sin_curves  = calculate_population_proportions(sin_curves, cum_pop_vd, dataframes_dvd, age_band, simulate_proportinal_norm, simulate_sinus)
                                                    
                # Call the function to simulate the dAEFI events
                curve_out, daefi_out = simulate_dAEFI_events(updated_sin_curves, dataframes_vd, age_band, window_size, event_threshold, future_day_range, simulate_dAEFI)
                updated_sin_curves[0] = curve_out

                # Not used - Call the function to calculate estimated dAEFI events
                #estimated_th_events, event_thresholds = calculate_estimated_dAEFI_events(
                #    dataframes_vd, age_band, window_size,raw_d_data_curve,
                #    baseline_d_without_dAEFI_unknown, baseline_d_without_dAEFI_known
                #)

                # Loop for each CSV File (NUM_D, NUM_DUVX, NUM_DVX, NUM_DVDA)
                for i in range(0, len(dataframes_dvd)):   
                    
                    # Simulate D-Data as sinus waves (optional) 
                    df_dvd = process_sine_data(
                        dataframes_dvd, 
                        cum_pop_vd, 
                        updated_sin_curves, 
                        age_band, 
                        normalize,  # or False if you don't want normalization
                        i  # Index 0-3 for the specific dataset (NUM_D, NUM_DUVX, NUM_DVX, NUM_DVDA or NUM_POP, NUM_UVX, NUM_VX, NUM_VDA)
                    )
                            
                    # Determine the color shades based on idx (age_band)
                    shades_1, shades_2 = (color_shades[i] if idx == 0 else color_shades_r[i])

                    # Add the traces with Deaths and all given Doses data to the plot 
                    add_traces_for_data(
                        trace_manager,
                        df_dvd, 
                        dataframes_vd, 
                        i, 
                        age_band, 
                        dae, 
                        window_size_mov_average, 
                        normalize, 
                        shades_1, 
                        shades_2, 
                        csv_files_dvd, 
                        csv_files_vd, 
                        cum_pop_vd
                    )

                                                                                    
            # Add agebands to Title-Text     
            # plot_title_text = f"{title_text} {age_band0} vs {age_band1}"                

            # Assign the plot traces-curves to the y-axis
            plot_layout(plot_title_text,trace_manager.get_fig(), px.colors.qualitative.Dark24 )


        # Loop through each pair of age bands in the age_band_compare list
        # calcualte difference of death curves for each age_band pair
        calculate_age_band_differences(f"Avg NUM_D {dae}", f"NUM_D {dae}", age_band_compare, trace_manager, window_size_mov_average, flip_color_axis=True)
        
        # Loop through each pair of age bands in the age_band_compare list
        # calcualte difference of all doses curves for each age_band pair
        calculate_age_band_differences(f"Avg NUM_VDA {dae}", f"NUM_VDA {dae}", age_band_compare, trace_manager, window_size_mov_average, flip_color_axis=False)

        
        D_curve = get_trace_by_name(f'Sum DIF-NUM_D {dae}', "", trace_manager)        
        y_values = D_curve.y  # # Step 1: Extract the y-values from the Scatter object, This will be a list of y-values        
        D_curve_in = pd.Series(y_values) # Step 2: Convert the y-values into a pandas Series

        Doses_curve = get_trace_by_name(f'Avg sum DIF-NUM_VDA {dae}', "", trace_manager) 
        y_values = Doses_curve.y
        Doses_curve_in = pd.Series(y_values)
        
        # Call the function to add simulated dAEFI events    
        simulate_dAEFI=True           
        D_curve_out, daefi_out = add_dAEFI_To_Trace(D_curve_in, Doses_curve_in, window_size, event_threshold, future_day_range, simulate_dAEFI)
        time_indices = np.arange(len(D_curve_out))  
        
        yaxis = 'y6'        
        trace_manager.add_trace(
            name=f'Sum DIF-NUM_D AEF',  # Name the sum trace appropriately
            x=time_indices,  # Same time indices for consistency
            y=D_curve_out,  # Cumulative sum of differences
            line=dict(dash='solid', width=1,  color='grey'),  # Customize line style and color for the sum
            secondary=True,  # Use secondary axis (if needed)
            axis_assignment=yaxis  # Assign to a different y-axis for clarity
        )

        # Calculate and add moving average trace for sum
        totalled_diff_series = pd.Series(D_curve_out) # Convert the NumPy array to a pandas Series                
        moving_average_totalled_diff = pd.Series(0.0, index=totalled_diff_series.index, dtype=float)
        # Calculate the rolling mean with center=True
        moving_average_totalled_diff_temp = totalled_diff_series.rolling(window=window_size_mov_average, center=True).mean()
        # Replace the valid moving averages with the calculated values (ignoring NaN values)
        moving_average_totalled_diff[moving_average_totalled_diff_temp.notna()] = moving_average_totalled_diff_temp[moving_average_totalled_diff_temp.notna()]

        trace_manager.add_trace(
            name=f'Avg sum DIF-NUM_D AEF',  # Name the sum trace appropriately
            x=time_indices,  # Same time indices for consistency
            y=moving_average_totalled_diff,  # Cumulative sum of differences
            line=dict(dash='solid', width=1,  color='orange'),  # Customize line style and color for the sum
            secondary=True,  # Use secondary axis (if needed)
            axis_assignment=yaxis  # Assign to a different y-axis for clarity
        )


        # Calculate correlation for the sum traces (vx doses all given, and deaths)
        # The sum traces were calculated by summing the differences for each age band in the age_band_compare list            
        if calc_correlation:    
            
            corr_band_pairs = [
                    (f'Sum cum DIF-NUM_VDA RAW', f'Avg sum DIF-NUM_D AEF')
            ]    
            plot_rolling_correlation_and_phase_shift_for_traces(trace_manager, corr_band_pairs,"", colors, 10, max_phase_shift) 

            corr_band_pairs = [
                    (f'Sum cum DIF-NUM_VDA RAW', f'Avg sum DIF-NUM_D RAW')
            ]    
            plot_rolling_correlation_and_phase_shift_for_traces(trace_manager, corr_band_pairs,"", colors, 10, max_phase_shift) 
        
            # Adding the red horizontal line at p = 0.05
            x_values = np.arange(len(dataframes_dvd[0]))  # Create x values from 0 to len
            trace_manager.add_trace(
                name=f'p = 0.05 significance level',
                x=x_values,
                y=[0.05] * len(x_values),  # Constant line at p = 0.05
                line=dict(color='red', width=1, dash='dash'),
                secondary=True,
                axis_assignment='y7'
            )

    # Assign the plot traces-curves to the y-axis
    plot_layout(plot_title_text,trace_manager.get_fig(), px.colors.qualitative.Dark24 )
        
    # Save the plot to an HTML file
    # If you want to automatically save the plots to prevent them from being overwritten, 
    # you can add the dependent variables here!
    html_file_path = rf'C:\CzechFOI-SIM\Plot Results\dAEFI\{plotfile_name} {plot_name_append_text} AG_15-85.html'
    trace_manager.save_to_html(html_file_path)

    # Extract the base filename without the .html extension
    file_name_without_extension, _ = os.path.splitext(html_file_path)

    # Saves the traces to a .csv file
    if savetraces_to_csv:
        save_traces_to_csv(trace_manager, file_name_without_extension)



# --- Start Function part --- 


# Loop through each pair of age bands calculate NUM_D and NUM_VDA difference 
def calculate_age_band_differences(tracename_prefix, tracename_new, age_band_compare, trace_manager,  window_size_mov_average, flip_color_axis):    
    totalled_diff = None

    for n, age_band_pair in enumerate(age_band_compare):
        # Initialize age bands
        age_band1, age_band2 = age_band_pair[0], age_band_pair[1]

        #print(f"Comparing age bands: {age_band1} and {age_band2}")

        # Set color palette based on index (even or odd)
        if n % 2 == 0: 
          colors = px.colors.qualitative.Plotly 
        else :
          colors = px.colors.qualitative.Plotly_r
        
        # Get the traces for the two age bands
        trace1 = get_trace_by_name(tracename_prefix, age_band1, trace_manager)
        trace2 = get_trace_by_name(tracename_prefix, age_band2, trace_manager)


        # Check if traces were successfully found
        if trace1 is None or trace2 is None:
            print(f"Warning: One or both traces for {age_band1} and {age_band2} not found.")
            continue  # Skip this iteration if either trace is missing

        # Check if the length of the traces match
        if len(trace1.y) != len(trace2.y):
            print(f"Warning: Trace lengths for {age_band1} and {age_band2} do not match.")
        
        # Calculate the difference and mid-difference between the traces
        df_diff = np.array(trace2.y) - np.array(trace1.y)  # Calculate the difference in the 'y' values
        df_diff = np.nan_to_num(df_diff, nan=0) # Replace NaN values with 0

         # Sum the difference across all traces
        if totalled_diff is None:
            totalled_diff = df_diff  # Initialize with the first difference trace
        else:
            totalled_diff += df_diff  # Add the current difference trace to the cumulative sum

        # Ensure time_indices is defined (same length as df_diff)
        time_indices = np.arange(len(df_diff))  

        # Determine which y-axis to use (y5 for even, y6 for odd)
        yaxis = 'y6' if n % 2 == 0 else 'y5'

        # Add the plot for the difference (df_diff)
        trace_manager.add_trace(
            name=f'DIF-{tracename_new} {age_band1}<br>{age_band2}',  # Name the trace appropriately
            x=time_indices,  # Use time_indices as the x-axis
            y=df_diff,  # df_diff is a NumPy array
            line=dict(dash='solid', width=1.5, color=colors[(n + 1) % len(colors)]),  # Customize line style and color
            secondary=True,  # Use secondary axis (if needed)
            axis_assignment=yaxis,  # Assign to the correct y-axis (adjust based on your figure setup)
            plot=False
        )  

        yaxis = 'y4' if n % 2 == 0 else 'y3'
        cum_df_diff = df_diff.cumsum()          
        # Add cumulative VD data trace on the secondary y-axis
        trace_manager.add_trace(
            name=f'Cum DIF-{tracename_new} {age_band1}<br>{age_band2}', 
            x=time_indices,  
            y=cum_df_diff,            
            line=dict(dash='dot', width=1.5, color=colors[(n + 2) % len(colors)]),
            secondary=True,
            axis_assignment=yaxis,
            plot=False
        )                                                        

        # Calculate and add moving average trace for DIF
        df_diff_series = pd.Series(df_diff)  # Convert the NumPy array to a pandas Series
        # Initialize the moving average with 0, but explicitly set the dtype as float
        moving_average_dif = pd.Series(0.0, index=df_diff_series.index, dtype=float)
        # Calculate the rolling mean with center=True
        moving_average_temp = df_diff_series.rolling(window=window_size_mov_average, center=True).mean()
        # Replace the valid moving averages with the calculated values (ignoring NaN values)
        moving_average_dif[moving_average_temp.notna()] = moving_average_temp[moving_average_temp.notna()]
        yaxis = 'y6' if n % 2 == 0 else 'y5'
        trace_manager.add_trace(
            name=f'Avg DIF-{tracename_new} {age_band1}<br>{age_band2}',  
            x=time_indices,  # Use time_indices as the x-axis
            y=moving_average_dif.values,  
            line=dict(dash='solid', width=1.5, color=colors[(n + 3) % len(colors)]),
            secondary=True,  
            axis_assignment=yaxis,
            plot=False 
        )
    
    if flip_color_axis:
        colors = px.colors.qualitative.Plotly 
        yaxis = 'y6'
    else:
        colors = px.colors.qualitative.Plotly_r 
        yaxis = 'y5' 

    # Plot the sum of all totalled differences    
    if totalled_diff is not None:
        trace_manager.add_trace(
            name=f'Sum DIF-{tracename_new}',  # Name the sum trace appropriately
            x=time_indices,  # Same time indices for consistency
            y=totalled_diff,  # Cumulative sum of differences
            line=dict(dash='solid', width=1.5,  color=colors[(n + 4) % len(colors)]),  # Customize line style and color for the sum
            secondary=True,  # Use secondary axis (if needed)
            axis_assignment=yaxis  # Assign to a different y-axis for clarity
        )
    else:
        print("Warning: No differences were calculated, so no cumulative sum was plotted.")

    # Calculate and add moving average trace for sum
    totalled_diff_series = pd.Series(totalled_diff) # Convert the NumPy array to a pandas Series                
    moving_average_totalled_diff = pd.Series(0.0, index=totalled_diff_series.index, dtype=float)
    # Calculate the rolling mean with center=True
    moving_average_totalled_diff_temp = totalled_diff_series.rolling(window=window_size_mov_average, center=True).mean()
    # Replace the valid moving averages with the calculated values (ignoring NaN values)
    moving_average_totalled_diff[moving_average_totalled_diff_temp.notna()] = moving_average_totalled_diff_temp[moving_average_totalled_diff_temp.notna()]

    trace_manager.add_trace(
        name=f'Avg sum DIF-{tracename_new}',  # Name the sum trace appropriately
        x=time_indices,  # Same time indices for consistency
        y=moving_average_totalled_diff,  # Cumulative sum of differences
        line=dict(dash='solid', width=1.5,  color=colors[(n + 5) % len(colors)]),  # Customize line style and color for the sum
        secondary=True,  # Use secondary axis (if needed)
        axis_assignment=yaxis  # Assign to a different y-axis for clarity
    )

    if flip_color_axis:
            yaxis = 'y4'
    else:
            yaxis = 'y3'             

    tot_diff_cum = totalled_diff.cumsum()
    trace_manager.add_trace(
        name=f'Sum cum DIF-{tracename_new}',  # Name the sum trace appropriately
        x=time_indices,  # Same time indices for consistency
        y=tot_diff_cum,  # Cumulative sum of differences
        line=dict(dash='solid', width=1.5,  color='green'),  # Customize line style and color for the sum
        secondary=True,  # Use secondary axis (if needed)
        axis_assignment=yaxis  # Assign to a different y-axis for clarity
    )
        

def get_trace_by_name(name0, age_band_extension, trace_manager):
    # Loop through all traces in the figure
    for trace in trace_manager.figData.data:
        # Check if the trace name starts with name0 and ends with the provided age_band_extension (if given)
        if trace.name.startswith(name0) and trace.name.endswith(age_band_extension):
            print(f"TR:{name0} * {age_band_extension}:{trace.name}:")
            # Check if the trace has valid data (y values are not None and the length is greater than 0)
            if trace.y is not None and len(trace.y) > 0:
                return trace
            else:
                print(f"Warning: Trace {trace.name} has no data or invalid data: {len(trace.y)}")
    # If no matching trace is found, return None or handle the case as needed
    print(f"Warning: No trace found with the name starting with {name0} and ending with {age_band_extension}")
    return None


def plot_daefi_curves(dataframes_vd, daefi_events, estimated_daefi_events, doses_per_daefi, window_size, trace_manager,tracename_extension , colors, simulate_dAEFI):
    if simulate_dAEFI:
        
        # Add dAEFI (rolling average)
        yaxis = 'y7'
        ave_daefi_events = pd.Series(daefi_events).rolling(window=window_size, min_periods=1).mean()
        trace_manager.add_trace(
            name=f'dAEFI {tracename_extension}', 
            x=dataframes_vd[0].iloc[:, 0],  # Assuming x-axis is the first column in dataframes_vd[0]
            y=ave_daefi_events,            
            line=dict(dash='solid', width=1.5, color=colors[0]),
            secondary=True,
            axis_assignment=yaxis
        )

        # Add horizontal line at the mean of ave_daefi_events
        mean_ave_daefi_events = ave_daefi_events.mean()
        trace_manager.add_trace(
            name=f'Mean dAEFI {tracename_extension}', 
            x=dataframes_vd[0].iloc[:, 0],  
            y=[mean_ave_daefi_events] * len(dataframes_vd[0].iloc[:, 0]),  # horizontal line
            line=dict(dash='solid', width=1.5, color=colors[1]), 
            secondary=True,
            axis_assignment=yaxis
        )
        
        # Add estimated threshold (= estimated dAEFI per x doses, rolling average)
        yaxis = 'y5'
        ave_estimated_th_events = pd.Series(estimated_daefi_events).rolling(window=window_size, min_periods=1).mean()
        trace_manager.add_trace(
            name=f'estim dAEFI {tracename_extension}', 
            x=dataframes_vd[0].iloc[:, 0],  
            y=ave_estimated_th_events,            
            line=dict(dash='solid', width=1.5, color=colors[2]),
            secondary=True,
            axis_assignment=yaxis
        )

        # Add horizontal line at the mean for estimated dAEFI per x doses
        mean_ave_estimated_th_events = ave_estimated_th_events.mean()
        trace_manager.add_trace(
            name=f'Mean estim dAEFI {tracename_extension}', 
            x=dataframes_vd[0].iloc[:, 0],  
            y=[mean_ave_estimated_th_events] * len(dataframes_vd[0].iloc[:, 0]),  # horizontal line
            line=dict(dash='solid', width=1.5, color=colors[3]), 
            secondary=True,
            axis_assignment=yaxis
        )

        # Add horizontal line at target doses_per_daefi value
        yaxis = 'y5'
        trace_manager.add_trace(
            name=f'Event Threshold {tracename_extension}', 
            x=dataframes_vd[0].iloc[:, 0],  
            y=[doses_per_daefi] * len(dataframes_vd[0].iloc[:, 0]),  # horizontal line at doses_per_daefi threshold
            line=dict(dash='solid', width=1.5, color='orange'),  # Orange color for the threshold line
            secondary=True,
            axis_assignment=yaxis
        )


def calculate_estimated_dAEFI_events(dataframes_vd, age_band, window_size,raw_d_data_curve,
                                     baseline_d_without_dAEFI_unknown, baseline_d_without_dAEFI_known):
    # Calculate the rolling moving average of VDA events
    mov_average_vd = dataframes_vd[3][age_band].rolling(window=window_size, min_periods=1,center=True).mean()

    # Initialize variables for tracking events and thresholds
    vda_events = mov_average_vd  # Raw VDA events for the age band
    cumulative_vda = 0
    d_add = 0  # Number of added D events
    additional_d_events = []  # List to track excess D events (those above baseline)
    event_thresholds = []  # List to store estimated event thresholds

    # Step 3: Calculate additional D events triggered by VDA
    estimated_th_events = [0] * len(raw_d_data_curve)  # List to store estimated events for each day

    # Loop through each day and calculate excess D events
    for day_idx in range(len(vda_events)):
        vda_value = vda_events[day_idx]

        # Calculate the D values if you know the baseline without VDA
        if not baseline_d_without_dAEFI_unknown: # test scenario (baseline without the dAEFIs known)
            d_value_with_vda = raw_d_data_curve[day_idx]
            baseline_d_value = baseline_d_without_dAEFI_known[day_idx]
        else:  # Real-world scenario (baseline without dAEFIs is unknown)
            d_value_with_vda = raw_d_data_curve[day_idx]
            baseline_d_value = baseline_d_without_dAEFI_known[day_idx]

        # Calculate the additional D events (difference between D with VDA and baseline D)
        additional_d = d_value_with_vda - baseline_d_value

        # If the additional D events are greater than zero, we count them
        if additional_d > 0:
            additional_d_events.append(additional_d)

            # Accumulate the VDA value over time
            cumulative_vda += vda_value

            # After accumulating enough D events, calculate the estimated threshold
            if len(additional_d_events) > 0:
                # Estimate how many VDA events per D event (average over the accumulated events)
                estimated_threshold = cumulative_vda / sum(additional_d_events)
                estimated_th_events[day_idx] += estimated_threshold
                event_thresholds.append(estimated_threshold)

    # Step 4: Calculate the average event threshold
    if event_thresholds:
        average_threshold = np.mean(event_thresholds)
        print(f"Estimated Event Threshold: {average_threshold}")
    else:
        print("No excess D events triggered.")

    # Return the estimated dAEFI events and the calculated thresholds
    return estimated_th_events, event_thresholds


def add_dAEFI_To_Trace(D_Curve, Doses_Curve, window_size, event_threshold, future_day_range, simulate_dAEFI):

    # Calculate the moving average of VDA using a rolling window
    mov_average_vd = Doses_Curve.rolling(window=window_size, min_periods=1, center=True).mean()
    
    # Initialize the cumulative VDA count and other variables
    cumulative_vda = 0
    d_add = 0
    rest = 0

    # Initialize output lists
    daefi_events_out = copy.deepcopy(D_Curve)
    D_curve_out = copy.deepcopy(D_Curve)

    # Loop through each day to check if an event should be triggered
    for day_idx in range(len(mov_average_vd)):
        # Get the current day's VDA value (moving average VDA value)
        vda_value = mov_average_vd[day_idx]                   

        # Check if the value is not NaN
        if math.isnan(vda_value):
            continue

        vda_value += rest
        rest = 0
        add_d_events = 0

        # Accumulate the VDA value
        cumulative_vda += vda_value

        # Check if the absolute value of cumulative VDA triggers an event
        if abs(cumulative_vda) >= event_threshold:
            # Randomly select a future day in the range [1, 25]
            future_day = random.randint(1, future_day_range)
            # Calculate the future index (ensure it's within bounds)
            future_idx = day_idx + future_day
            if future_idx < len(D_Curve):  # Ensure the index is within bounds for D_Curve
                # Calculate the number of dAEFI events to add or subtract based on cumulative VDA
                add_d_events = cumulative_vda // event_threshold
                d_add += add_d_events
                
                if simulate_dAEFI:  
                    print(f"add dAEFI:{cumulative_vda}>={event_threshold}->{add_d_events}")          
                    # Adjust the D_curve_out and daefi_events_out based on the sign of Doses_Curve value
                    if Doses_Curve[day_idx] > 0:
                        daefi_events_out[future_idx] += add_d_events  # Add dAEFI event
                        D_curve_out[future_idx] += add_d_events      # Add to D_curve
                    else:
                        daefi_events_out[future_idx] -= add_d_events  # Subtract dAEFI event
                        D_curve_out[future_idx] -= add_d_events      # Subtract from D_curve

            # Reset cumulative VDA count and store the rest for next accumulation
            rest = cumulative_vda % event_threshold
            cumulative_vda = 0

    # Return the updated D_curve_out and daefi_events_out
    return D_curve_out, daefi_events_out


def simulate_dAEFI_events(sin_curves, dataframes_vd, age_band, window_size, event_threshold, future_day_range, simulate_dAEFI):
    # Calculate the moving average of VDA using a rolling window
    mov_average_vd = dataframes_vd[3][age_band].rolling(window=window_size, min_periods=1,center=True).mean()
    
    # Initialize the cumulative VDA count and other variables
    cumulative_vda = 0
    d_add = 0
    rest = 0

    # Initialize output lists
    daefi_events_out = [0] * len(sin_curves[0])
    # sin_curve_out = sin_curves[0][:]  # Make a copy reference to sin_curves[0] which chnages the original!
    sin_curve_out = copy.deepcopy(sin_curves[0]) # Create a deep copy of sin_curves[0]

    # Loop through each day to check if an event should be triggered
    for day_idx in range(len(mov_average_vd)):
        # Get the current day's VDA value (moving average VDA value)
        vda_value = mov_average_vd[day_idx]                  

        # Check if the value is not NaN
        if math.isnan(vda_value):
            continue

        vda_value += rest
        rest = 0
        add_d_events = 0

        # Accumulate the VDA value
        cumulative_vda += vda_value

        # Check if the cumulative VDA value triggers an event
        if cumulative_vda >= event_threshold:
            # Randomly select a future day in the range [1, 25]
            future_day = random.randint(1, future_day_range)
            # Calculate the future index (ensure it's within bounds)
            future_idx = day_idx + future_day
            if future_idx < len(sin_curves[0]):  # Ensure the index is within bounds for sin_curve[0]
                # Add dAEFI event to sin_curves and daefi_events_out at the future day index
                add_d_events = cumulative_vda // event_threshold
                d_add += add_d_events
                if simulate_dAEFI:  
                    # print(f"add:{add_d_events}")             
                    daefi_events_out[future_idx] += add_d_events  # Save to plot later
                    sin_curve_out[future_idx] +=  add_d_events

            # Reset cumulative VDA count and store the rest for next accumulation
            rest = cumulative_vda % event_threshold
            cumulative_vda = 0

    # Return the updated sin_curve_out and daefi_events_out
    return sin_curve_out, daefi_events_out


# Process sine wave data, fill leading zeros, and assign appropriate color shades.
def process_sine_data(dataframes_dvd, cum_pop_vd, sin_curves,  
                                 age_band, normalize, i):
    
    # Initialize a copy of the dataframes_dvd to avoid modifying the original
    df_dvd = [df.copy() for df in dataframes_dvd]
    
    # Process sine waves and assign them to the appropriate columns
    leading_zeros = first_nonzero_index(df_dvd[i][age_band])
    sin_curves[i] = fill_leading_days_with_zeros(sin_curves[i], leading_zeros)

    df_dvd[i][age_band] = sin_curves[i]  # Update the appropriate curve (D, DUVX, DVX, DVDA)
    
    # Normalize the data per 100,000 if required
    if normalize:
        df_dvd[i][age_band] = (df_dvd[i][age_band] / cum_pop_vd[i]) * 100000
    #else:
    #    norm_data = df_dvd[i][age_band]
    
    return df_dvd


# Add traces for DVD and VD data to the plot.
def add_traces_for_data(trace_manager, df_dvd, df_vd, i, age_band, dae,  
                        window_size_mov_average, normalize, shades_1, shades_2, 
                        csv_files_dvd, csv_files_vd, cum_pop_vd):

    age_band_extension = age_band.split('-')[0]  
    
    # Already normalized by function -> process_sine_data!
    # Normalize the data per 100,000                 
    #norm_dataframes_dvd = (df_dvd[i][age_band] / cum_pop_vd[i]) * 100000
    #if normalize:                    
    #    plt_dataframes_dvd = norm_dataframes_dvd 
    #else :
    
    plt_dataframes_dvd = df_dvd[i][age_band]   

    # For DVD (i == 0 or i == 3)
    if i == 0 or i == 3:
        
        
        # Add traces for DVD on primary y-axis
        yaxis = 'y1'                
        trace_manager.add_trace(
            name=f'{os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {dae}{age_band_extension}',
            x=df_dvd[i].iloc[:, 0],
            y=plt_dataframes_dvd,
            mode='lines',
            line=dict(dash='solid', width=1, color=shades_1[0]),
            secondary=False,
            axis_assignment=yaxis,
            plot=plot_trace)

        # Add moving average trace for DVD
        yaxis = 'y1'
        moving_average_dvd = plt_dataframes_dvd.rolling(window=window_size_mov_average, center=True).mean()
        #moving_average_dvd = moving_average_dvd.rolling(window=window_size_mov_average, center=True).mean()
        moving_average_dvd.fillna(0)
        trace_manager.add_trace(
            name=f'Avg {os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {dae}{age_band_extension}',
            x=df_dvd[i].iloc[:, 0],
            y=moving_average_dvd,
            line=dict(dash='solid', width=1, color=shades_1[1]),
            secondary=False,
            axis_assignment=yaxis,
            plot=True)

        # Add cumulative DVD data trace
        cum_dataframes_dvd = df_dvd[i][age_band].cumsum()
        yaxis = 'y3'
        trace_manager.add_trace(
            name=f'cum {os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {dae}{age_band_extension}',
            x=df_dvd[i].iloc[:, 0],
            y=cum_dataframes_dvd,
            line=dict(dash='dot', width=1.5, color=shades_1[2]),
            secondary=True,
            axis_assignment=yaxis,
            plot=plot_trace)

        # Add trace for VD
        yaxis = 'y2'
        trace_manager.add_trace(
            name=f'{os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {dae}{age_band_extension}',
            x=df_vd[i].iloc[:, 0],
            y=df_vd[i][age_band],
            line=dict(dash='solid', width=1, color=shades_2[0]),
            secondary=False,
            axis_assignment=yaxis,
            plot=plot_trace)

        # Add moving average trace for VD
        yaxis = 'y2'
        moving_average_vd = df_vd[i][age_band].rolling(window=window_size_mov_average, center=True).mean()
        moving_average_vd.fillna(0)
        trace_manager.add_trace(
            name=f'Avg {os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {dae}{age_band_extension}',
            x=df_vd[i].iloc[:, 0],
            y=moving_average_vd,
            line=dict(dash='solid', width=1, color=shades_2[1]),
            secondary=False,
            axis_assignment=yaxis)

        # Add cumulative VD data trace
        yaxis = 'y4'
        trace_manager.add_trace(
            name=f'cum {os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {dae}{age_band_extension}',
            x=df_vd[i].iloc[:, 0],
            y=cum_pop_vd[i],
            line=dict(dash='dot', width=1.5, color=shades_2[2]),
            secondary=True,
            axis_assignment=yaxis,
            plot=plot_trace)



# Calculate population proportions for UVX, VX, and VDA, adjusting based on various simulation conditions.
def calculate_population_proportions(sin_curves, cum_pop_vd, dataframes_dvd, age_band, simulate_proportinal_norm, simulate_sinus):
 
    proportions = []


    # Loop through each time point (day)
    for t in range(len(sin_curves[0])):

        # Calculate population proportions for UVX and VX
        C = 1 / (1 + (cum_pop_vd[2][t] / cum_pop_vd[1][t])) if cum_pop_vd[1][t] != 0 else 0
        C2 = 1 / (1 + (cum_pop_vd[3][t] / cum_pop_vd[1][t])) if cum_pop_vd[1][t] != 0 else 0

        if simulate_proportinal_norm and simulate_sinus:
            # Scale the amplitudes of UVX and VX in proportion to their population share
            sin_curves[1][t] = C * sin_curves[0][t]  # Scale DUVX amplitude
            sin_curves[2][t] = sin_curves[0][t] - (sin_curves[0][t] * C)  # Scale DVX amplitude
            sin_curves[3][t] = sin_curves[0][t] - (sin_curves[0][t] * C2)  # Scale DVDA

        if not simulate_proportinal_norm and simulate_sinus:
            sin_curves[0][t] = sin_curves[0][t]  # D
            sin_curves[1][t] = sin_curves[1][t]  # DUVX
            sin_curves[2][t] = sin_curves[2][t]  # DVX
            sin_curves[3][t] = sin_curves[3][t]  # DVDA

        if simulate_proportinal_norm and not simulate_sinus:
            sin_curves[0][t] = dataframes_dvd[0][age_band][t]
            sin_curves[1][t] = C * dataframes_dvd[0][age_band][t]
            sin_curves[2][t] = dataframes_dvd[0][age_band][t] - (dataframes_dvd[0][age_band][t] * C)
            sin_curves[3][t] = dataframes_dvd[0][age_band][t] - (dataframes_dvd[0][age_band][t] * C2)

        if not simulate_proportinal_norm and not simulate_sinus:
            # Real data: simulate proportional deaths for VX, UVX, VDA
            sin_curves[0][t] = dataframes_dvd[0][age_band][t]
            sin_curves[1][t] = dataframes_dvd[1][age_band][t]
            sin_curves[2][t] = dataframes_dvd[2][age_band][t]
            sin_curves[3][t] = dataframes_dvd[3][age_band][t]

        # Sum of UVX and VX (checking consistency with total deaths sin[0])
        sum_sin_1_2 = sin_curves[1][t] + sin_curves[2][t]
        if not np.isnan(sum_sin_1_2) and not np.isnan(sin_curves[0][t]) and not np.isclose(sum_sin_1_2, sin_curves[0][t]):
            print(f"uvx + vx != D at time {t}: sin_1 + sin_2 = {sum_sin_1_2}, sin_0 = {sin_curves[0][t]}")

        # Append proportions to the list for plotting (as functions of t)
        proportions.append(C)

    return proportions, sin_curves


# Calculate the cumulative population, considering different factors like deaths and UVX/VX data
def calculate_cumulative_population(dataframes_vd, dataframes_dvd, age_band, population_minus_death=False):
   
    # Initialize the list to hold cumulative population data
    cum_pop_vd = []
    
    # Loop through each data group (POP, UVX, VX, etc.)
    for i in range(len(dataframes_vd)):
        if i == 0:  # For population data
            if population_minus_death:
                # Subtract cumulative deaths from population
                cum_dataframes_vd = dataframes_vd[0][age_band] - dataframes_dvd[0][age_band].cumsum()                          
            else:
                # Just use the population data as is
                cum_dataframes_vd = dataframes_vd[0][age_band]
        
        elif i == 1:  # For UVX data
            if population_minus_death:
                # Adjust population by subtracting cumulative deaths and adding cumulative UVX
                cum_dataframes_vd = dataframes_vd[0][age_band] - dataframes_dvd[1][age_band].cumsum() + dataframes_vd[1][age_band].cumsum()
            else:
                # Just use the population and cumulative UVX data
                cum_dataframes_vd = dataframes_vd[0][age_band] + dataframes_vd[1][age_band].cumsum()
        
        else:  # For other data groups like VX, VDA, etc.
            if population_minus_death:
                # Adjust population by subtracting cumulative deaths for each group
                cum_dataframes_vd = dataframes_vd[i][age_band].cumsum() - dataframes_dvd[i][age_band].cumsum()
            else:
                # Just use the cumulative data for each group
                cum_dataframes_vd = dataframes_vd[i][age_band].cumsum()

        # Append the resulting cumulative population data for the group
        cum_pop_vd.append(cum_dataframes_vd)
    
    return cum_pop_vd


# Calculate proportions of UVX, VX, and VDA for each time point based on the data provided.
def calculate_proportions(dataframes_vd, age_band):
    # Ensure age_band exists in all dataframes
    for i, df in enumerate(dataframes_vd):
        if age_band not in df:
            print(f"Error: 'age_band' not found in dataframes_vd[{i}]")
            return None

    # Initialize lists to hold the proportions
    proportion_uvx = []
    proportion_vx = []
    proportion_vda = []

    # Loop through each time point (assuming all age bands have the same length)
    for t in range(len(dataframes_vd[0][age_band])):
        # Ensure each DataFrame has the same length for the given age_band
        if (len(dataframes_vd[1][age_band]) != len(dataframes_vd[0][age_band]) or 
            len(dataframes_vd[2][age_band]) != len(dataframes_vd[0][age_band]) or 
            len(dataframes_vd[3][age_band]) != len(dataframes_vd[0][age_band])):
            print("Error: DataFrames have mismatched lengths for the given age_band")
            return None

        # Calculate proportion for UVX
        if dataframes_vd[0][age_band][t] != 0:  
            proportion_uvx.append(dataframes_vd[1][age_band][t] / dataframes_vd[0][age_band][t])
        else:
            proportion_uvx.append(0)

        # Calculate proportion for VX
        if dataframes_vd[0][age_band][t] != 0:  
            proportion_vx.append(dataframes_vd[2][age_band][t] / dataframes_vd[0][age_band][t])
        else:
            proportion_vx.append(0)

        # Calculate proportion for VDA
        if dataframes_vd[0][age_band][t] != 0:  
            proportion_vda.append(dataframes_vd[3][age_band][t] / dataframes_vd[0][age_band][t])
        else:
            proportion_vda.append(0)

    # Convert lists to numpy arrays for easier element-wise operations
    proportion_uvx = np.array(proportion_uvx)
    proportion_vx = np.array(proportion_vx)
    proportion_vda = np.array(proportion_vda)

    return proportion_uvx, proportion_vx, proportion_vda

# Function to generate shifted sine wave with given parameters 
def shifted_sin(days, period=30, amplitude=1, vertical_shift=0, horizontal_shift=0):
    # Apply the horizontal shift to the days array
    return amplitude * np.sin(2 * np.pi * (days + horizontal_shift) / period) + vertical_shift


# Function to create a modulated wave with given parameters
def create_modulated_wave(days, large_period, small_period, large_amplitude, small_amplitude, horizontal_shift=0, shift_y=0):
    # Generate the large and small sine waves with horizontal (x) and vertical (y) shifts
    large_wave = shifted_sin(days, period=large_period, amplitude=large_amplitude, vertical_shift=0, horizontal_shift=horizontal_shift)
    small_wave = shifted_sin(days, period=small_period, amplitude=small_amplitude, vertical_shift=0, horizontal_shift=horizontal_shift)
    
    # Modulate the large wave with the small wave
    modulated_wave = large_wave + small_wave
    
    # Apply the vertical shift to the modulated wave
    modulated_wave_shifted = modulated_wave + shift_y
    
    return modulated_wave_shifted


# Function to generate the modulated waves for D, UVX, VX, DVDA
def generate_modulated_wave(days, modulated_wave_params):
    
    # generate d, duvx, dvx, dvda modulated sin curve and return as List
    sin_curves = []
    for n in range(4):  # We need curves 0 to 3
        # Determine the parameter set for each curve
        if n == 0:  # Curve 0 (D) = Curve 1 (UVX) + Curve 2 (VX)
            # Generate Curve 1 (UVX) and Curve 2 (VX)
            sin_curve_uvx = create_modulated_wave(
                days,
                large_period=modulated_wave_params[0]['large_period'],
                small_period=modulated_wave_params[0]['small_period'],
                large_amplitude=modulated_wave_params[0]['large_amplitude'],
                small_amplitude=modulated_wave_params[0]['small_amplitude'],
                horizontal_shift=modulated_wave_params[0]['horizontal_shift'],
                shift_y=modulated_wave_params[0]['vertical_shift']
            )
            sin_curve_vx = create_modulated_wave(
                days,
                large_period=modulated_wave_params[1]['large_period'],
                small_period=modulated_wave_params[1]['small_period'],
                large_amplitude=modulated_wave_params[1]['large_amplitude'],
                small_amplitude=modulated_wave_params[1]['small_amplitude'],
                horizontal_shift=modulated_wave_params[1]['horizontal_shift'],
                shift_y=modulated_wave_params[1]['vertical_shift']
            )
            # Curve 0 (D) is the sum of UVX and VX
            sin_curve = sin_curve_uvx + sin_curve_vx
        else:
            # Determine which parameters to use for the other curves
            if n == 1:  # Curve 1 (UVX)
                mod_params = modulated_wave_params[0]
            elif n == 2:  # Curve 2 (VX)
                mod_params = modulated_wave_params[1]
            elif n == 3:  # Curve 3 (DVDA) = Curve 2 (VX)
                mod_params = modulated_wave_params[1]

            # Generate the sine curve using the selected parameters
            sin_curve = create_modulated_wave(
                days,
                large_period=mod_params['large_period'],
                small_period=mod_params['small_period'],
                large_amplitude=mod_params['large_amplitude'],
                small_amplitude=mod_params['small_amplitude'],
                horizontal_shift=mod_params['horizontal_shift'],
                shift_y=mod_params['vertical_shift']
            )

        # Append the generated sine curve to the list of curves
        sin_curves.append(sin_curve)

    return sin_curves

        
# Function to fill the leading days with NaN based on zero_day_prefix
def fill_leading_days_with_zeros(sin_curve, zero_day_prefix):
    # Overwrite the first `zero_day_prefix` values with NaN
    sin_curve[:zero_day_prefix] = np.nan
    return sin_curve

 
# Helper function for identifying the first valid (non-zero, non-NaN) index in a series
def first_nonzero_index(series, threshold=1e-6):
    # Mask out NaNs and values close to zero (below the threshold)
    valid_series = series.where((series != 0) & (~np.isnan(series)) & (np.abs(series) > threshold))
    
    # Find the first valid (non-zero, non-NaN) index
    non_zero_indices = np.nonzero(~np.isnan(valid_series.values))[0]
    
    if len(non_zero_indices) > 0:
        return non_zero_indices[0]
    else:
        return 0  # Return 0 if no valid data exists


def generate_age_band_pairs(age_band_pair, age_band_pairs):
    # Create an empty list to store the updated pairs
    updated_age_band_pairs = []
    
    # Extract the lower bounds of the single age band pair
    age1_lower = age_band_pair[0].split('-')[0]  # Get the lower bound of the first age band
    age2_lower = age_band_pair[1].split('-')[0]  # Get the lower bound of the second age band
    
    # Loop through the age_band_pairs and add the age information
    for pair in age_band_pairs:
        # For each variable, add the corresponding age bounds
        updated_pair = (f'{pair[0]} n AG {age1_lower}-{age2_lower}', f'{pair[1]} n AG {age1_lower}-{age2_lower}')
        updated_age_band_pairs.append(updated_pair)

    # Return the updated pairs
    return updated_age_band_pairs


# Calculate and plot rolling correlation, significance, and phase shift for each pair of traces.
def plot_rolling_correlation_and_phase_shift_for_traces(trace_manager, pairs, age_band_extension, colors, n_color=0, max_phase_shift=300):
    # Returns: time_indices_list (list): List of time indices for each pair.
    # start_idx_list (list): List of the starting indices for each pair.

    # Create y_series from the trace data
    y_series = {}
    z = -1
    for trace in trace_manager.figData.data:  # Loop through all traces in trace_manager
        z += 1 
        if trace.y is not None and len(trace.y) > 0:
            y_data = np.array(trace.y).flatten()
            y_series[trace.name] = xr.DataArray(y_data, dims='time', coords={'time': np.arange(len(y_data))})
            #print(f"Trace {z}:{trace.name}:")
        else:
            print(f"Warning {z}:{trace.name}: has no data or invalid data: {len(trace.y)}")

    
    # Lists to store time indices and start indices for each pair
    time_indices_list = []
    start_idx_list = []

    # Loop through the pairs of trace names
    for n, (name1, name2) in enumerate(pairs):
        
        
        try:
            # Handle the case where traces with the exact names are not found
            matching_name1 = [key for key in y_series.keys() if key.startswith(name1) and key.endswith(age_band_extension)]
            matching_name2 = [key for key in y_series.keys() if key.startswith(name2) and key.endswith(age_band_extension)]
            
            if matching_name1 and matching_name2:
                df1 = y_series[matching_name1[0]]
                df2 = y_series[matching_name2[0]]
            else:
                print(f"Rolling correlation - no matching traces found for :{name1}*{age_band_extension}: and :{name2}*{age_band_extension}: - matchingname1 :{matching_name1}: and matchingname2 :{matching_name2}:")
                continue  # Skip this pair if no match found

        except KeyError:
            print(f"Rolling correlation Key - no matching traces found for :{name1}*{age_band_extension}: and :{name2}*{age_band_extension}: - matchingname1 :{matching_name1}: and matchingname2 :{matching_name2}:")
            continue  # Skip this pair if no match found

        # Find the first non-zero index for both series in their raw form
        start_idx1_raw = first_nonzero_index(df1)
        start_idx2_raw = first_nonzero_index(df2)
        start_idx = max(start_idx1_raw, start_idx2_raw)

        # Filter and align the data, excluding NaNs and zeros
        filtered_df1, filtered_df2 = filter_and_align(df1, df2)

        if len(filtered_df1) == 0 or len(filtered_df2) == 0:
            print(f"Filtered data for {name1} and {name2} is empty. Skipping this pair.")
            continue

        # Calculate rolling correlation and significance
        rolling_corr, p_values = rolling_significance_test(filtered_df1, filtered_df2, window_size_correl)

        # Synchronize the rolling correlation with the data series
        rolling_corr[:0] = np.nan  

        # Time indices for plotting
        time_indices = np.arange(0, len(filtered_df1))

        # Append the time indices and start index to the lists
        time_indices_list.append(time_indices)
        start_idx_list.append(start_idx)

        # Plot rolling correlation
        trace_manager.add_trace(
            name = f'Rolling Corr<br>{name1}<br>{name2} {age_band_extension}' if age_band_extension != "" else f'Rolling Corr<br>{name1}<br>{name2}',
            x=time_indices,
            y=rolling_corr[0:],
            line=dict(dash='solid', width=1.5, color=colors[(n_color+1) % len(colors)]),
            secondary=True,
            axis_assignment='y7'
        )

        # Plot significant correlation (p < 0.05)
        trace_manager.add_trace(
            name=f'Sig Corr<br>{name1}<br>{name2} {age_band_extension} (p<0.05)' if age_band_extension != "" else f'Sig Corr<br>{name1}<br>{name2} (p<0.05)',
            x=time_indices,
            y=(p_values[0:] < 0.05).astype(int),
            line=dict(dash='dash', width=1, color=colors[(n_color+2) % len(colors)]),
            secondary=True,
            axis_assignment='y7'
        )

        # Plot p-values
        trace_manager.add_trace(
            name=f'P-Values<br>{name1}<br>{name2} {age_band_extension}' if age_band_extension != "" else f'P-Values<br>{name1}<br>{name2}',
            x=time_indices,
            y=p_values[0:],  
            mode='lines+markers',
            marker=dict(size=3, color='gray'),
            line=dict(dash='dot', width=1, color='gray'),
            text=p_values[0:],
            hoverinfo='text',
            secondary=True,
            axis_assignment='y7'
        )

        # Calculate and plot phase shift correlation
        phase_corr = phase_shift_correlation(filtered_df2, filtered_df1, max_phase_shift)
        
        trace_manager.add_trace(
            name=f'Ph Shift Corr<br>{name1}<br>{name2} {age_band_extension}' if age_band_extension != "" else f'Ph Shift Corr<br>{name1}<br>{name2}',
            x=np.arange(-max_phase_shift, max_phase_shift + 1),
            y=phase_corr,
            line=dict(dash='solid', width=2, color=colors[(n_color+3) % len(colors)]),  # Different color
            secondary=True,
            axis_assignment='y'
        )

    # Return the lists containing time indices and start indices for each pair
    return time_indices_list, start_idx_list


def phase_shift_correlation(series1, series2, max_shift=50):
    # Calculate the correlation of two time series at different phase shifts.
    correlations = []
    for shift in range(-max_shift, max_shift + 1):
        shifted_series2 = np.roll(series2, shift)
        # Remove NaN or invalid values (if any) after shifting
        valid_idx = ~np.isnan(series1) & ~np.isnan(shifted_series2)
        correlation = np.corrcoef(series1[valid_idx], shifted_series2[valid_idx])[0, 1]
        correlations.append(correlation)
    return np.array(correlations)


def rolling_significance_test(series1, series2, window_size):
    # Mask out NaNs and zeros in series1 and series2
    valid_series1 = series1.where((series1 != 0) & (~np.isnan(series1)))
    valid_series2 = series2.where((series2 != 0) & (~np.isnan(series2)))

    rolling_corr = []
    p_values = []

    # Calculate rolling correlation only for valid windows (non-NaN and non-zero)
    for i in range(len(valid_series1) - window_size + 1):
        window1 = valid_series1[i:i + window_size]
        window2 = valid_series2[i:i + window_size]

        # Skip windows with NaN or zeros in either series
        if window1.isnull().any() or window2.isnull().any() or (window1 == 0).all() or (window2 == 0).all():
            rolling_corr.append(np.nan)
            p_values.append(np.nan)
            continue

        # Calculate correlation and p-value for the valid window
        corr, p_value = stats.pearsonr(window1.values, window2.values)
        rolling_corr.append(corr)
        p_values.append(p_value)

    return np.array(rolling_corr), np.array(p_values)


# Function for filtering and aligning series (removing NaNs and zeros)
def filter_and_align(series1, series2):
    # Mask out leading NaNs and zeros in both series
    non_zero_1 = series1.where((series1 != 0) & (~np.isnan(series1)))
    non_zero_2 = series2.where((series2 != 0) & (~np.isnan(series2)))

    # Align series by truncating them to the same length
    min_length = min(len(non_zero_1), len(non_zero_2))
    return non_zero_1[:min_length], non_zero_2[:min_length]


# Helper function for identifying the first valid (non-zero, non-NaN) index in a series
def first_nonzero_index(series, threshold=1e-6):
    # Mask out NaNs and values close to zero (below the threshold)
    valid_series = series.where((series != 0) & (~np.isnan(series)) & (np.abs(series) > threshold))
    
    # Find the first valid (non-zero, non-NaN) index
    non_zero_indices = np.nonzero(~np.isnan(valid_series.values))[0]
    
    if len(non_zero_indices) > 0:
        return non_zero_indices[0]
    else:
        return 0  # Return 0 if no valid data exists
    

# Update plot layout for dual y-axes - traces were assignd to yaxis by tracemanager       
def plot_layout(title_text_in, figData, color_palette, desired_width=2096, desired_height=800):
    
    # Update the layout of the figure
    figData.update_layout(
        colorway=color_palette,
        title=dict(
            text=title_text_in,
            y=0.98,
            font=dict(family='Arial', size=18),
            x=0.03,  # Move title to the left (0.0 = far left)
            xanchor='left',  # Anchor title to the left side
            yanchor='top'
        ),
        annotations=[  # Add annotation manually
            dict(
                text=annotation_text,
                x=0.1,  # Position annotation to the left of the plot
                y=0.99, # Adjust the y-position slightly lower than the main title
                xanchor='left',  # Anchor the annotation to the left
                yanchor='top',        
                font=dict(family='Arial',size=14, color='grey'),  # annotation style
                showarrow=False,  # No arrow
                align='left',   # This ensures the text itself is left-aligned
                xref='paper',  # This ensures 'x' is relative to the plot area (paper means the entire canvas)
                yref='paper'  # This ensures 'y' is relative to the plot area
            )
        ],
        
        #xaxis=dict(
        #    title='primaryX-axis: Days from 2020-01-01',  # Set the title for the standard x-axis
        #    title_standoff=25,  # Adjust the distance of the title from the axis (optional)
        #    showticklabels=True,  # Ensure that the tick labels are shown
        #),
        
        # For different scale set autorange=True for yaxis2-7 and for yaxis8-13
        # For same scale remove autorange=True for yaxis2-7, yaxis8-13 not used then
        # 1st Age_Band (idx)
        yaxis=dict(title='yaxis1 sum DIF VDA', type="log", side='left'), # yaxis_type log/linear from the instance of PlotConfig
        yaxis2=dict(title='yaxis1 D baseline / sum DIF D', anchor='free', position=0.05, side='left',autorange=True),
        yaxis3=dict(title="yaxis3 sum DIF VDA cumulated", overlaying="y", position=0.9, side="right"),
        yaxis4=dict(title='yaxis4_title', overlaying="y", side="right"),
        yaxis5=dict(title='yaxis5_title', overlaying="y", side="left", position=0.15 ),  # 1st and 2nd derivative y5
        yaxis6=dict(title='yaxis6_title', overlaying="y", side="left", position=0.25 ),   # 1st and 2nd derivative y6
        yaxis7=dict(title='yaxis7_title', overlaying='y', side='right', position=0.8), # Rolling Pearson y7   
        #xaxis6=dict(title='xaxis6',  overlaying='x', side='bottom', position=0.5),  # Add this line for recurrence plot x-axis         
        
        # 2nd Age_Band (idx) - used only if differnt sacle for age_bands needed
        yaxis8=dict(title='',overlaying="y", type="log", side='left', autorange=True),   # title=f'{plo["yaxis1_title"]} y1'             
        yaxis9=dict(title='',overlaying="y", anchor='free', position=0.05, side='left', autorange=True), # title='Values y2 VD'
        yaxis10=dict(title='', overlaying="y", position=0.9, side="right", autorange=True), # title='Cumulative Values y3 DVD'
        yaxis11=dict(title='', overlaying="y", side="right",autorange=True), # title='Cumulative Values y4 VD'
        yaxis12=dict(title='', overlaying="y", side="left", position=0.15, autorange=True),  # 1st and 2nd Derivative y5 DVD'
        yaxis13=dict(title='', overlaying="y", side="left", position=0.25, autorange=True),  # 1st and 2nd Derivative y6 DVD'
        # Rolling Pearson Correlation has the same scale -1 to 1 so no additional yaxis14 needed 

        legend=dict(
            orientation="v",
            xanchor="left",
            x=1.05,
            yanchor="top",
            y=1,
            font=dict(size=10)
            #itemwidth=200  # legend width (in pixels)
        ),
        margin=dict(l=40, r=50, t=40, b=40),

        # Set desired width and height
        #width=desired_width,
        #height=desired_height
    )

    
class TraceManager:
    def __init__(self):
        self.figData = make_subplots(specs=[[{"secondary_y": True}]])  # Initialize the figure with subplots
        self.axis_list = []  # Stores axis assignments for each trace
        self.trace_visibility = {}  # Dictionary to track the visibility of traces by their names

    def add_trace(self, name, x, y, line=None, mode=None, marker=None, text=None, hoverinfo=None, axis_assignment='y1', secondary=False, plot=True):
        """
        Adds a trace to the figure and assigns it to a specific axis.

        Args:
        - plot (bool): If True, the trace will be added to the plot and visible in the HTML file.
        - axis_assignment: Axis for the trace ('y1', 'y2', etc.).
        - Other args: Same as in Plotly (e.g., trace_name, x_data, y_data, line, mode, text, secondary, hoverinfo).
        """
        
        # Create the trace even if plot=False
        trace = go.Scatter(
            x=x,
            y=y,
            mode=mode,               # Directly use 'mode' (Plotly's parameter)
            marker=marker,           # Directly use 'marker' (Plotly's parameter)
            line=line,               # Directly use 'line' (Plotly's parameter)
            name=name,               # Trace name
            text=text,               # Hover text
            hoverinfo=hoverinfo      # Hover information
        )

        # Add the trace to the figure
        self.figData.add_trace(trace, secondary_y=secondary)

        # Store the axis assignment
        self.axis_list.append(axis_assignment)

        # Update the trace's axis assignment
        self._update_axis_for_trace(len(self.axis_list) - 1, axis_assignment)

        # Mark the trace visibility status
        self.trace_visibility[name] = plot  # Only visible if plot is True

    def _update_axis_for_trace(self, trace_index, axis_assignment):
        # Updates the axis for the specific trace in the figure.
        
        assigned_axis = axis_assignment
        trace = self.figData.data[trace_index]

        # Update the trace's axis based on the assignment
        if assigned_axis == 'y1':
            trace.update(yaxis='y1')
        elif assigned_axis == 'y2':
            trace.update(yaxis='y2')
        elif assigned_axis == 'y3':
            trace.update(yaxis='y3')
        elif assigned_axis == 'y4':
            trace.update(yaxis='y4')
        elif assigned_axis == 'y5':
            trace.update(yaxis='y5')
        elif assigned_axis == 'y6':
            trace.update(yaxis='y6')
        elif assigned_axis == 'y7':
            trace.update(yaxis='y7')
        elif assigned_axis == 'y8':
            trace.update(yaxis='y8')
        elif assigned_axis == 'y9':
            trace.update(yaxis='y9')
        elif assigned_axis == 'y10':
            trace.update(yaxis='y10')
        elif assigned_axis == 'y11':
            trace.update(yaxis='y11')
        elif assigned_axis == 'y12':
            trace.update(yaxis='y12')
        elif assigned_axis == 'y13':
            trace.update(yaxis='y13')

    def filter_traces(self):
        # Filters out traces that are set to be invisible        
        # Remove traces that are not visible
        self.figData.data = [trace for trace in self.figData.data if self.trace_visibility.get(trace.name, True)]

    def get_fig(self):        
        #Returns the figure object, including all traces (visible and invisible)        
        return self.figData

    def get_axis_list(self):
        # Returns the list of axis assignments        
        return self.axis_list

    def save_to_html(self, filename):
        #Save the current figure to an interactive HTML file, excluding invisible traces
        
        self.filter_traces()  # Filter traces before saving
        pio.write_html(self.figData, filename)
        print(f"Figure saved to {filename}")



# Saves the trace data (x, y) from all traces in the figure managed by a TraceManager to a csv file 
def save_traces_to_csv(trace_manager, filename='trace_data.csv'):
    """
    The first column of CSV file is 'Day' and subsequent columns are the trace names.
    
    Args:
    - trace_manager: An instance of TraceManager.
    - filename: The name of the CSV file to save the data.
    """
    # Get the filename without the extension
    file_name_without_extension, file_extension = os.path.splitext(filename)

    # Ensure the filename has a .csv extension if not already present
    if not file_extension:
        filename = f'{file_name_without_extension}.csv'

    # Initialize a list to store all rows (days and y-values)
    all_data = []
    
    # Get the x-values (assumed to be the same for all traces)
    x_data = trace_manager.get_fig().data[0].x  # Assuming x-values are the same for all traces

    # Collect the y-values for each trace
    trace_names = []  # To store the names of the traces (for headers)
    for trace in trace_manager.get_fig().data:
        trace_name = trace.name if trace.name else 'Unnamed'
        trace_names.append(trace_name)
    
    # Create the header: 'Day', followed by trace names
    header = ['Day'] + trace_names
    all_data.append(header)

    # Now, collect the y-values for each day and each trace
    for i, x in enumerate(x_data):
        row = [x]  # Start with the day value (x)
        
        for trace in trace_manager.get_fig().data:
            if len(trace.y) > i:  # Ensure trace.y has at least i+1 elements
                row.append(trace.y[i])  # Append the corresponding y-value for the trace
            else:
                # If trace.y is empty or doesn't have enough values, append None or a placeholder
                row.append(None)

        all_data.append(row)
    
    # Write the collected data to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_data)
    
    print(f"Trace data saved to {filename}")


# Function for creating shades (reusable for each color palette)
def generate_shades(base_color, num_shades=5, lightness_factor=0.1):
    shades = []
    # Convert hex color to RGB
    if base_color.startswith("#"):            
        base_color = mcolors.to_rgb(base_color)

    # Convert RGB to HSV (hue, saturation, brightness)
    hsv = colorsys.rgb_to_hsv(base_color[0], base_color[1], base_color[2])

    # Create shades by varying the brightness
    for i in range(num_shades):
        new_value = min(1.0, max(0.4, hsv[2] + lightness_factor * (i - 2)))  # Adjust brightness
        new_rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], new_value)  # Keep hue and saturation constant
        new_hex = mcolors.rgb2hex(new_rgb)  # Convert back to Hex
        shades.append(new_hex)

    return shades


# Function for creating color pairs and shades
def generate_color_shades(color_palette, n_pairs=11):
    color_shades = {}
    for i in range(n_pairs):
        # Select color pairs from the palette
        base_color_dvd_1 = color_palette[i % len(color_palette)]
        base_color_vd_1 = color_palette[(i + 1) % len(color_palette)]

        # Calculate shading for the DVD and VD
        shades_dvd_1 = generate_shades(base_color_dvd_1)
        shades_vd_1 = generate_shades(base_color_vd_1)

        # Save the shades
        color_shades[i] = (shades_dvd_1, shades_vd_1)

    return color_shades


# call main script
if __name__ == "__main__": main()
