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

# This script processes data from pivot CSV files located in the TERRA folder, 
# which were generated from a Czech Freedom of Information request (Vesely_106_202403141131.csv). 
# The pivot files were created using the DB Browser for SQLite.
# can simulate adding dAEFIs (deadly adverse event following immun.) 
# can also simulate modulated sin waves for d, duvx , dvx, (dvda -deaths all v doses)  


title_text="Add dAEFIs to real D-Curves or to generated sinus Wave D-Curves  - threshold dAEFI 1/5000 DOSE RAND_DAY_RANGE 1-250 WND_14 AG 50-54 vs 75-79" # Title of plot
annotation_text = "legend: ae -> dAEFIs added, n-> normalized per 10000 People, pe -> simulate equal death rate for dvx and duvx (proportional to population)"
plotfile_name = "AB) backcalc dAEFI simulation sinus real world basline" 
plot_name_append_text=""        # apend text - to plotfile name and directory, to save in uniqe file location
normalize=True                  # normalize dvd values
normalize_cumulate_deaths=False # normalize cumulated deaths bevore cummulation
population_minus_death=False    # deducts the deceased from the total population
custom_legend_column = 10       # A change could require adjustments to the code. 
axis_y1_logrithmic=True         # show logarithmic y1 axis
savetraces_to_csv=False         # save calcualted results of all traces into a csv file
window_size_mov_average =30     # Define the window size for the rolling average (adjust this as needed)

# Define the event threshold (1 in 10,000 VDA events triggers an event)
event_threshold = 5000          # Trigger simulates one  dAEFI (deadly adverse event following immunisation) per 5,000 VDA (vax dose all)
future_day_range = 250          # Random future day range (for example 1 to 14)
window_size = 14                # Define the window size for the rolling average dAEFI (adjust this as needed)

# simulation behavior
simulate_sinus = True           # uses modulated sin wave to simulate Death curve
simulate_proportinal_norm = False   # simulate constant death curve adjusted to (uvx, vx , total) population (use real data or sin wave for each dey 1..1534)
simulate_dAEFI = True
real_world_baseline = True

def main():

    # List of tuples with the age bands you want to compare
    age_band_compare = [
                ('50-54', '75-79'),
        ]
        
    # CSV file pairs with age_band with death and population/doses data  
    csv_files_dvd = [
            r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_D.csv",
            r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DUVX.csv",
            r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVX.csv",
            r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVDA.csv",
        ]

    csv_files_vd = [
            r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_POP.csv",
            r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_UVX.csv",
            r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VX.csv",
            r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VDA.csv",
        ]

    modulated_wave_params_ag1 = [
                    {'small_period': 3, 'large_period': 360, 'small_amplitude': 1, 'large_amplitude': 2, 'vertical_shift': 5, 'horizontal_shift': 0}, # Curve 1 DUVX - AG1
                    {'small_period': 3, 'large_period': 360, 'small_amplitude': 1, 'large_amplitude': 2, 'vertical_shift': 5, 'horizontal_shift': 0}, # Curve 2 DVX - AG1
        ]  
    
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

    # Loop through each pair of age bands in the list
    pair_nr = 0
    for age_band_pair in age_band_compare:    
        # Create an instance of TraceManager
        trace_manager = TraceManager()   
        dae = ""
        for k in range(0, 2):        # plot simulate add dAEFI
            if k==0: 
                simulate_dAEFI = False     
                aefi = ""
            else: 
                simulate_dAEFI = True
                aefi = "ae " # legend prefix if true

            for r in range(0, 2):        # plot simulate proportional dvx duvx dvda curves with similar death rates
                if r==0: 
                    simulate_proportinal_norm = False     
                    prop = ""
                else: 
                    simulate_proportinal_norm = True
                    prop = "pr " 

                for s in range(0, 2):        #  plot normalized per 100000 people ( or dvda - doses)
                    if s==0: 
                        normalize = False     
                        norm = ""
                    else: 
                        normalize = True
                        norm = "n "
                        
                    dae = f"{aefi}{prop}{norm}" # legend prefix                           
                    # Get the shades for the current index and age band pair
                    for idx, age_band in enumerate(age_band_pair):

                        # Use Plotly and Plotly_r color palette with 10 different colors 
                        # for the two age groups and plot additional decay and correlation traces
                        if idx == 0:    
                            colors = px.colors.qualitative.Plotly                    
                        elif idx == 1:  
                            colors = px.colors.qualitative.Plotly_r          

                        # Define the days range (from 0 to 1530)
                        max_days = len(dataframes_vd[0][age_band])
                        days = np.linspace(0, max_days-1, max_days) 

                        # Loop through each time point to calculate deaths for UVX and VX - check for division by zero
                        proportion_uvx = []
                        proportion_vx = []            
                        proportion_vda = []
                        # Loop through each time point to calculate deaths for UVX and VX - check for division by zero
                        for t in range(len(dataframes_vd[0][age_band])):  
                            # Calculate proportion for UVX
                            if dataframes_vd[0][age_band][t] != 0:  
                                proportion_uvx.append(dataframes_vd[1][age_band][t] / dataframes_vd[0][age_band][t])  # Proportion uvx = Population curve 2 (UVX) / Population curve 1 (total pop)
                            else:
                                proportion_uvx.append(0)  # If total population is zero, set proportion to 0

                            # Calculate proportion for VX
                            if dataframes_vd[0][age_band][t] != 0:  
                                proportion_vx.append(dataframes_vd[2][age_band][t] / dataframes_vd[0][age_band][t])  # Proportion vx = Population curve 3 (VX) / Population curve 1 (total pop)
                            else:
                                proportion_vx.append(0)  
                
                            # Calculate proportion for VDA (normalize per dose insted vxd-person)
                            if dataframes_vd[0][age_band][t] != 0:  
                                proportion_vda.append(dataframes_vd[3][age_band][t] / dataframes_vd[0][age_band][t])  # Proportion vda = Population curve 4 (VDA) / Population curve 1 (total pop)
                            else:
                                proportion_vda.append(0)  
                
                        # Convert proportions into numpy arrays for easier element-wise operations
                        proportion_uvx = np.array(proportion_uvx)
                        proportion_vx = np.array(proportion_vx)
                        proportion_vda = np.array(proportion_vda)

                        # Define the parameters for each modulated wave, depending on AG idx
                        if idx == 0 :
                                modulated_wave_params = modulated_wave_params_ag1
                        if idx == 1 :
                                modulated_wave_params = modulated_wave_params_ag2
                        
                        # Generat D, DUVX, DVX, DVDA data as modulated sin curves, by using the two parameter sets 
                        sin_curves = generate_modulated_wave(days, modulated_wave_params)

                        # calculate cumulated population
                        cum_pop_vd = []
                        for i in range(0, len(dataframes_dvd)):   
                            # Calculate cumulative VD data trace
                            if i == 0:  # csv file[0] POP
                                if population_minus_death:
                                    # POP - cum D
                                    cum_dataframes_vd = dataframes_vd[0][age_band] - dataframes_dvd[0][age_band].cumsum()                         
                                else :
                                    # POP
                                    cum_dataframes_vd = dataframes_vd[0][age_band]                
                            elif i == 1: # csv file [1] UVX
                                if population_minus_death:
                                    # POP - cum D - cum UVX
                                    cum_dataframes_vd = dataframes_vd[0][age_band] - dataframes_dvd[1][age_band].cumsum() + dataframes_vd[1][age_band].cumsum()                         
                                else :
                                    # POP - cum UVX
                                    cum_dataframes_vd =  dataframes_vd[0][age_band]  + dataframes_vd[1][age_band].cumsum() 
                            else:   # csv files [i]                    
                                if population_minus_death:
                                    # VX..VDX - cum D 
                                    cum_dataframes_vd = dataframes_vd[i][age_band].cumsum() - dataframes_dvd[i][age_band].cumsum()                        
                                else :
                                    # VX..VDX
                                    cum_dataframes_vd = dataframes_vd[i][age_band].cumsum() 
                            
                            # Append the generated sin curve to the list
                            cum_pop_vd.append(cum_dataframes_vd)                                               

                        # Loop through each day (time point)
                        proportions = []
                        trace_block = ""
                        for t in range(len(sin_curves[0])):

                            # Calculate population proportions for UVX and VX
                            C   = 1 / (1 + (cum_pop_vd[2][t]/cum_pop_vd[1][t])) if cum_pop_vd[1][t] != 0 else 0
                            C2  = 1 / (1 + (cum_pop_vd[3][t]/cum_pop_vd[1][t])) if cum_pop_vd[1][t] != 0 else 0

                            if simulate_proportinal_norm and simulate_sinus:
                                # Scale the amplitudes of UVX and VX in proportion to their population share
                                sin_curves[1][t] = C * sin_curves[0][t]                         # Scale DUVX amplitude
                                sin_curves[2][t] = sin_curves[0][t] - (sin_curves[0][t]*C)      # Scale DVX amplitude
                                sin_curves[3][t] = sin_curves[0][t] - (sin_curves[0][t]*C2)     # Scale DVDA

                            if not simulate_proportinal_norm and simulate_sinus:
                                sin_curves[0][t] = sin_curves[0][t]  # D
                                sin_curves[1][t] = sin_curves[1][t]  # DUVX 
                                sin_curves[2][t] = sin_curves[2][t]  # DVX
                                sin_curves[3][t] = sin_curves[3][t]  # DVDA
                                
                            if simulate_proportinal_norm and not simulate_sinus:
                                # print (f"proportinal {simulate_proportinal_norm}: sinus {simulate_sinus}: - {dae}" )                                
                                sin_curves[0][t] = dataframes_dvd[0][age_band][t]
                                sin_curves[1][t] = C * dataframes_dvd[0][age_band][t]
                                sin_curves[2][t] = dataframes_dvd[0][age_band][t] - (dataframes_dvd[0][age_band][t]*C)
                                sin_curves[3][t] = dataframes_dvd[0][age_band][t] - (dataframes_dvd[0][age_band][t]*C2)

                            if not simulate_proportinal_norm and not simulate_sinus:
                                trace_block = "b2"
                                # Real data simulate proportional deaths for vx uvx vda
                                # print (f"prop {simulate_proportinal_norm}: sinus {simulate_sinus}: - {dae}" )
                                sin_curves[0][t] = dataframes_dvd[0][age_band][t]
                                sin_curves[1][t] = dataframes_dvd[1][age_band][t]
                                sin_curves[2][t] = dataframes_dvd[2][age_band][t]
                                sin_curves[3][t] = dataframes_dvd[3][age_band][t]

                            # Sum of UVX and VX (checking consistency with total deaths sin[0])
                            sum_sin_1_2 = sin_curves[1][t] + sin_curves[2][t]
                            if not np.isnan(sum_sin_1_2) and not np.isnan(sin_curves[0][t]) and not np.isclose(sum_sin_1_2, sin_curves[0][t]):
                                print(f"uvx + vx != D at time {t}: sin_1 + sin_2 = {sum_sin_1_2}, sin_0 = {sin_curves[0][t]}")
                            
                            # Append proportions to the lists for plotting (as functions of t)
                            proportions.append(C)                          

                        # Simulate dAFEI (fatal deadly Adverse Event Following Immunization)
                        mov_average_vd = dataframes_vd[3][age_band].rolling(window=window_size, min_periods=1).mean()    
                        d_events_without_vda = sin_curves[0]
                        baseline_d_avg_without_vda = pd.Series(d_events_without_vda).rolling(window=window_size, min_periods=1).mean()

                        # Initialize the cumulative VDA count
                        cumulative_vda = 0
                        d_add = 0
                        rest = 0
                        # Loop through each day to check if an event should be triggered
                        
                        daefi_events = [0] * len(sin_curves[0])
                        for day_idx in range(len(mov_average_vd)):
                            # Get the current day's VDA value (moving average VDA value)
                            vda_value = mov_average_vd[day_idx]                  
                            # Check if the value is not NaN
                            if math.isnan(vda_value):
                                continue
                            
                            #print(f"AGE: {age_band} Day {day_idx}: VDA = {vda_value}")
                            vda_value += rest
                            rest =0 
                            add_d_events = 0
                            # Accumulate the VDA value
                            cumulative_vda += vda_value

                            # Check if the cumulative VDA value triggers an event
                            if (cumulative_vda >= event_threshold) :
                                # Randomly select a future day in the range [1, 25]
                                future_day = random.randint(1, future_day_range)
                                # Calculate the future index (ensure it's within bounds)
                                future_idx = day_idx + future_day
                                if future_idx < len(sin_curves[0]):  # Ensure the index is within bounds for sin_curve[2]
                                    # Add dAEFI event to sin_curves d, vx, dvda at the future day index                                                
                                    add_d_events = cumulative_vda // event_threshold 
                                    d_add += add_d_events
                                    if simulate_dAEFI:               
                                        #add_d_events = 0                         
                                        daefi_events[future_idx]  += add_d_events # save to plot later
                                        sin_curves[0][future_idx] += add_d_events
                                        sin_curves[2][future_idx] += add_d_events
                                        sin_curves[3][future_idx] += add_d_events
                                # reset cumulative VDA count - store rest for next cumulation
                                rest = cumulative_vda % event_threshold
                                cumulative_vda = 0  

                            # After VDA events are simulated, the D events (with added VDA events) are in the same dataframe:
                            d_events_with_vda = sin_curves[0]  # D events after VDA additions
                            baseline_d_avg_with_vda = pd.Series(d_events_with_vda).rolling(window=window_size, min_periods=1).mean()

                        # Step 2: Calculate a rolling average of D events as the baseline            

                        # Initialize variables for tracking events and thresholds
                        vda_events = mov_average_vd   # Raw VDA events for the age band            
                        cumulative_vda = 0
                        d_add = 0  # Number of added D events
                        additional_d_events = []  # List to track excess D events (those above baseline)
                        event_thresholds = []  # List to store estimated event thresholds

                        # Step 3: Calculate additional D events triggered by VDA
                        estimated_th_events = [0] * len(sin_curves[0])
                        for day_idx in range(len(vda_events)):
                            vda_value = vda_events[day_idx]

                            # Calculate the D values if you know the baseline without VDA
                            if real_world_baseline == False:
                                d_value_with_vda = baseline_d_avg_with_vda[day_idx]
                                baseline_d_value = baseline_d_avg_without_vda[day_idx]
                                
                            # Calculate the D values if you don't know the baseline without VDA (real world scenario)
                            if real_world_baseline == True:
                                d_value_with_vda = d_events_with_vda[day_idx]
                                baseline_d_value = baseline_d_avg_with_vda[day_idx]

                            # Calculate the additional D events (difference between D with VDA and baseline D)
                            additional_d = d_value_with_vda - baseline_d_value

                            # Debugging: Print the excess D event calculation
                            # print(f"Day {day_idx}: VDA = {vda_value}, D (with VDA) = {d_value_with_vda}, Baseline D (with VDA) = {baseline_d_value}, Additional D = {additional_d}")
                                                        
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
                                    
                        
                        for i in range(0, len(dataframes_dvd)):   
                            
                            # Get the color shades for the current dataset (ensure the shades list is long enough)
                            if idx == 0:   # First age band
                                shades_1, shades_2 = color_shades[i]                    
                            elif idx == 1:  # Second age band reversd coloros
                                shades_1, shades_2 = color_shades_r[i]                
                            
                            age_band_extension = age_band.split('-')[0]  

                            # assign these sine waves to the respective columns in dataframes_dvd only if flag simulate_sinus is true!!
                            # Initialize df_dvd if not already done (make sure df_dvd is defined somewhere before this code)
                            df_dvd = [df.copy() for df in dataframes_dvd]  # Create a new list of dataframes, identical to dataframes_dvd initially
                            leading_zeros = 0
                            if  i == 0:
                                leading_zeros = first_nonzero_index(df_dvd[i][age_band]) 
                                #print (f"idx:{idx} i:{i} AGE:{age_band} zeros:{leading_zeros}")
                                sin_curves[0] = fill_leading_days_with_zeros(sin_curves[0], leading_zeros)
                                df_dvd[i][age_band] = sin_curves[0]  # Curve 0 D
                            if  i == 1:
                                leading_zeros = first_nonzero_index(df_dvd[i][age_band])  
                                #print (f"idx:{idx} i:{i} AGE:{age_band} zeros:{leading_zeros}")  
                                sin_curves[1] = fill_leading_days_with_zeros(sin_curves[1], leading_zeros)
                                df_dvd[i][age_band] = sin_curves[1]  # Curve 1 DUVX
                            if  i == 2:
                                leading_zeros = first_nonzero_index(df_dvd[i][age_band]) 
                                #print (f"idx:{idx} i:{i} AGE:{age_band} zeros:{leading_zeros}")
                                sin_curves[2] = fill_leading_days_with_zeros(sin_curves[2], leading_zeros)
                                df_dvd[i][age_band] = sin_curves[2]  # Curve 2 DVX                    
                            if  i == 3:
                                leading_zeros = first_nonzero_index(df_dvd[i][age_band])  
                                #print (f"idx:{idx} i:{i} AGE:{age_band} zeros:{leading_zeros}")
                                sin_curves[3] = fill_leading_days_with_zeros(sin_curves[3], leading_zeros)
                                df_dvd[i][age_band] = sin_curves[3]  # Curve 3 DVDA
                                # dataframes_dvd[i][age_band] = pd.Series(proportions_vx).add(pd.Series(proportions_uvx), fill_value=0)

                                
                            # Normalize the data per 100,000                 
                            norm_dataframes_dvd = (df_dvd[i][age_band] / cum_pop_vd[i]) * 100000
                            if normalize:                    
                                plt_dataframes_dvd = norm_dataframes_dvd 
                            else :
                                plt_dataframes_dvd = df_dvd[i][age_band]                
                            
                            yaxis='y1' 
                            # Add traces for DVD on primary y-axis                            
                            trace_manager.add_trace(
                                name=f'{os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {dae}{age_band_extension}', 
                                x=df_dvd[i].iloc[:, 0],
                                y=plt_dataframes_dvd,
                                mode='lines',
                                line=dict(dash='solid', width=1, color=shades_1[0]),                    
                                secondary=False,
                                axis_assignment=yaxis)
                            
                            yaxis='y1' 
                            # Calculate add moving average trace for DVD                            
                            moving_average_dvd = plt_dataframes_dvd.rolling(window=window_size_mov_average).mean()                                
                            trace_manager.add_trace(
                                name=f'Avg {os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {dae}{age_band_extension}', 
                                x=df_dvd[i].iloc[:, 0], 
                                y=moving_average_dvd,
                                line=dict(dash='solid', width=1, color=shades_1[1]), 
                                secondary=False, 
                                axis_assignment=yaxis)

                            # First normalize deaths then cummulate 
                            if normalize_cumulate_deaths:                   
                                cum_dataframes_dvd = norm_dataframes_dvd.cumsum()
                            else:
                                cum_dataframes_dvd = df_dvd[i][age_band].cumsum()   
                            
                            yaxis='y3'                 
                            # Add cumulative DVD data trace on the secondary y-axis
                            trace_manager.add_trace(
                                name=f'cum {os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {dae}{age_band_extension}', 
                                x=df_dvd[i].iloc[:, 0],  
                                y=cum_dataframes_dvd,            
                                line=dict(dash='dot', width=1.5, color=shades_1[4]),
                                secondary=True,
                                axis_assignment=yaxis)

                            yaxis='y2'
                            # Add trace for VD
                            trace_manager.add_trace(
                                name=f'{os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {dae}{age_band_extension}',
                                x=dataframes_vd[i].iloc[:, 0],  
                                y=dataframes_vd[i][age_band],   
                                line=dict(dash='solid', width=1,  color=shades_2[0]), 
                                secondary=False, 
                                axis_assignment=yaxis)
                            
                            yaxis='y2'
                            # Calculate add moving average trace for VD
                            moving_average_vd = dataframes_vd[i][age_band].rolling(window=window_size_mov_average).mean()                
                            trace_manager.add_trace(
                                name=f'Avg {os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {dae}{age_band_extension}',                    
                                x=dataframes_vd[i].iloc[:, 0],
                                y=moving_average_vd,
                                line=dict(dash='solid', width=1,  color=shades_2[1]),
                                secondary=False, 
                                axis_assignment=yaxis)                
                            
                            yaxis='y4'
                            # Add cumulative VD data trace on the secondary y-axis
                            trace_manager.add_trace(
                                name=f'cum {os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {dae}{age_band_extension}', 
                                x=dataframes_vd[i].iloc[:, 0],  
                                y=cum_pop_vd[i],            
                                line=dict(dash='dot', width=1.5, color=shades_2[4]),
                                secondary=True,
                                axis_assignment=yaxis
                            )                                                        

                        j=0
                        if simulate_dAEFI:
                            # Add dAEFI                
                            yaxis='y7'                                
                            ave_daefi_events = pd.Series(daefi_events).rolling(window=window_size, min_periods=1).mean()
                            trace_manager.add_trace(
                                name=f'dAEFI {dae}{age_band_extension}', 
                                x=dataframes_vd[i].iloc[:, 0],  
                                y=ave_daefi_events,            
                                line=dict(dash='solid', width=1.5, color=colors[(j+1) % len(colors)]),
                                secondary=True,
                                axis_assignment=yaxis
                            )

                            # Add horizontal line at the mean of ave_daefi_events
                            yaxis='y7'
                            mean_ave_daefi_events = ave_daefi_events.mean()
                            trace_manager.add_trace(
                                name=f'Mean dAEFI {dae}{age_band_extension}', 
                                x=dataframes_vd[i].iloc[:, 0],  
                                y=[mean_ave_daefi_events] * len(dataframes_vd[i].iloc[:, 0]),  # horizontal line
                                line=dict(dash='solid', width=1.5, color=colors[(j+1) % len(colors)]), 
                                secondary=True,
                                axis_assignment=yaxis
                            )
                           
                            # Add estimated threshold (= estimated dAEFI per x doses)                 
                            yaxis='y5'
                            ave_estimated_th_events = pd.Series(estimated_th_events).rolling(window=window_size, min_periods=1).mean()
                            trace_manager.add_trace(
                                name=f'estim dAEFI {dae}{age_band_extension}', 
                                x=dataframes_vd[i].iloc[:, 0],  
                                y=ave_estimated_th_events,            
                                line=dict(dash='solid', width=1.5, color=colors[(j+2) % len(colors)]),
                                secondary=True,
                                axis_assignment=yaxis
                            )      

                            # Add horizontal line at the mean for estimated dAEFI per x doses
                            yaxis='y5'
                            mean_ave_estimated_th_events = ave_estimated_th_events.mean()
                            trace_manager.add_trace(
                                name=f'Mean estim dAEFI {dae}{age_band_extension}', 
                                x=dataframes_vd[i].iloc[:, 0],  
                                y=[mean_ave_estimated_th_events] * len(dataframes_vd[i].iloc[:, 0]),  # horizontal line
                                line=dict(dash='solid', width=1.5, color=colors[(j+2) % len(colors)]), 
                                secondary=True,
                                axis_assignment=yaxis
                            )
 
                            # Add horizontal line at the used event_threshold value
                            yaxis='y5'
                            trace_manager.add_trace(
                                name=f'Event Threshold {dae}{age_band_extension}', 
                                x=dataframes_vd[i].iloc[:, 0],  
                                y=[event_threshold] * len(dataframes_vd[i].iloc[:, 0]),  # horizontal line at event_threshold
                                line=dict(dash='solid', width=1.5, color='orange'),  # Red color for the threshold line
                                secondary=True,
                                axis_assignment=yaxis
                            )

                        # Add proportion constant c 
                        yaxis='y7'                        
                        trace_manager.add_trace(
                            name=f'C {dae}{age_band_extension}', 
                            x=dataframes_vd[i].iloc[:, 0],  
                            y=proportions,            
                            line=dict(dash='dot', width=1.5, color=colors[(j+3) % len(colors)]),
                            secondary=True,
                            axis_assignment=yaxis
                        )
                        
                        # Assign the plot traces-curves to the y-axis
                        plot_layout(trace_manager.get_fig(), px.colors.qualitative.Dark24)

        
    # Save the plot to an HTML file with a custom legend
    # If you want to automatically save the plots in a different directory to prevent them from being overwritten, 
    # you can add the dependent variables here!
    html_file_path = rf'C:\github\CzechFOI-SIM\Plot Results\dAEFI\{plotfile_name} {plot_name_append_text} AG_{age_band_compare[pair_nr][0]} vs {age_band_compare[pair_nr][1]}.html'
    trace_manager.save_to_html(html_file_path)
    pair_nr += 1 # claculate next age band of the tuple (pair)

    # Extract the base filename without the .html extension
    file_name_without_extension, _ = os.path.splitext(html_file_path)

    # Saves the traces to a .csv file
    if savetraces_to_csv:
        save_traces_to_csv(trace_manager, file_name_without_extension)


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
    

# Update plot layout for dual y-axes - traces were assignd to yaxis by tracemanager       
def plot_layout(figData, color_palette, desired_width=2096, desired_height=800):
    
    # Update the layout of the figure
    figData.update_layout(
        colorway=color_palette,
        title=dict(
            text=title_text,
            y=0.98,
            font=dict(family='Arial', size=18),
            x=0.03,  # Move title to the left (0.0 = far left)
            xanchor='left',  # Anchor title to the left side
            yanchor='top'
        ),
        annotations=[  # Add annotation manually
            dict(
                text=annotation_text,
                x=0.28,  # Position annotation to the left of the plot
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

        # For different scale set autorange=True for yaxis2-7 and for yaxis8-13
        # For same scale remove autorange=True for yaxis2-7, yaxis8-13 not used then
        # 1st Age_Band (idx)
        yaxis=dict(title="yaxis1_title", type="log", side='left'), # yaxis_type log/linear from the instance of PlotConfig
        yaxis2=dict(title='Values y2 VD', anchor='free', position=0.05, side='left', autorange=True),
        yaxis3=dict(title="yaxis3_title", overlaying="y", position=0.9, side="right", autorange=True),
        yaxis4=dict(title='Cumulative Values y4 VD', overlaying="y", side="right", autorange=True),
        yaxis5=dict(title='1st and 2nd Derivative y5 DVD', overlaying="y", side="left", position=0.15, autorange=True),  # 1st and 2nd derivative y5
        yaxis6=dict(title='1st and 2nd Derivative y6 VD', overlaying="y", side="left", position=0.25, autorange=True),   # 1st and 2nd derivative y6
        yaxis7=dict(title='Rolling Pearson Correlation y7', overlaying='y', side='right', position=0.8, autorange=True), # Rolling Pearson y7        
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

    
# Class to mange yaxis assignment for the traces
class TraceManager:
    def __init__(self):
        self.figData = make_subplots(specs=[[{"secondary_y": True}]])  # Initialize the figure with subplots
        self.axis_list = []  # Stores axis assignments for each trace

    def add_trace(self, name, x, y, line=None, mode=None, marker=None, text=None, hoverinfo=None, axis_assignment='y1', secondary=False):
        """
        Adds a trace to the figure and assigns it to a specific axis.

        Args:
        - axis_assignment: Axis for the trace ('y1', 'y2', etc.).
        - Other args: Same as in Plotly (e.g., trace_name, x_data, y_data, line, mode, text, secondary, hoverinfo).
        """

        # Add trace using Plotly's standard parameters
        self.figData.add_trace(go.Scatter(
            x=x,
            y=y,
            mode=mode,              # Directly use 'mode' (Plotly's parameter)
            marker=marker,          # Directly use 'marker' (Plotly's parameter)   
            line=line,              # Directly use 'line' (Plotly's parameter)
            name=name,              # Trace name
            text=text,              # Hover text
            hoverinfo=hoverinfo     # Hover information
        ), secondary_y=secondary)   # Use 'secondary_y' for secondary axis
        
        # Store the axis assignment
        self.axis_list.append(axis_assignment)

        # Update the trace's axis assignment
        self._update_axis_for_trace(len(self.axis_list) - 1, axis_assignment)

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

    def get_fig(self):
        # Returns the figure object
        return self.figData

    def get_axis_list(self):
        # Returns the list of axis assignments
        return self.axis_list
    
    def save_to_html(self, filename):
        #Save the current figure to an interactive HTML file.
        # filename (str): Path to the HTML file to save the figure.
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
