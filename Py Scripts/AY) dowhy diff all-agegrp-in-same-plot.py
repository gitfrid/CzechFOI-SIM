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
import dowhy
import matplotlib.pyplot as plt


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

# tutorial: https://www.pywhy.org/dowhy/v0.12/example_notebooks/tutorial-causalinference-machinelearning-using-dowhy-econml.html
# https://github.com/amit-sharma/causal-inference-tutorial/

plot_title_text="D-curves and Doses-curves one year apart, for age groups from 15 to 85" # Title of plot
annotation_text ="AY) dowhy all-agegrp-in-same-plot.py / D_Curves (RAW)) -vs- D_Curves + 1 year (RAW)) / not normalized "
plotfile_name = "AY) dowhy diff all-agegrp-in-same-plot" 
plot_name_append_text="dAEFI"   # apend text - to plotfile name and directory, to save in uniqe file location
normalize=False                 # normalize dvd values
normalize_cumulate_deaths=False # normalize cumulated deaths bevore cummulation
population_minus_death=True     # deducts the deceased from the total population
custom_legend_column = 10       # A change could require adjustments to the code. 
axis_y1_logrithmic=True         # show logarithmic y1 axis
savetraces_to_csv=False         # save calcualted results of all traces into a csv file
window_size_mov_average=30      # Define the window size for the rolling average (adjust this as needed)
calc_correlation=False          # clacualte and plot rolling and phase shift correlation
window_size_correl=280          # window size for rolling pearson correlation
max_phase_shift=300             # max phase shift for phase shift correlation
plot_trace=True                 # bool - if true (default) trace is saved into the html file 


# Define the event threshold (1 in 10,000 VDA events triggers an event)
event_threshold = 5000       # Trigger simulates one  dAEFI (deadly adverse event following immunisation) per 5,000 VDA (vax dose all)
future_day_range = 1       # Random future day range (for example 1 to 250)
window_size = 14             # Define the window size for the rolling average dAEFI (adjust this as needed)

# simulation behavior
simulate_sinus=False            # uses modulated sin wave to simulate Death curve
simulate_proportinal_norm=False # simulate constant death curve adjusted to (uvx, vx , total) population (use real data or sin wave for each dey 1..1534)
simulate_dAEFI=True
real_world_baseline=True


def main():
    global dae

    # just for debugging - is faster
    dbg_age_band_compare = [
        ('23', '24'),
        ('53', '54'),
        ('73', '74'),
        ('74', '75'),
        ('75', '76'),

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
    for k in range(2):

        # Initialize DAE based on the value of k.
        if k == 0:
            dae = "RAW"
            simulate_dAEFI = False
        elif k == 1:
            dae = "AEF"
            simulate_dAEFI = True
                       
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
                                                    
                # Loop for each CSV File (NUM_D, NUM_DUVX, NUM_DVX, NUM_DVDA)
                for i in range(0, len(dataframes_dvd)):   
                    
                    
                    # Simulate D-Data as sinus waves (optional) 
                    df_dvd = process_sine_data(
                        dataframes_dvd, 
                        cum_pop_vd, 
                        updated_sin_curves, 
                        age_band, 
                        i  # Index 0-3 for the specific dataset (NUM_D, NUM_DUVX, NUM_DVX, NUM_DVDA or NUM_POP, NUM_UVX, NUM_VX, NUM_VDA)
                    )

                    #print(f"col:{ df_dvd[i].columns}")

                    if i == 0 or i == 3:
                       D_curve, daefi_events_out = add_dAEFI_To_Trace(df_dvd, dataframes_vd[3][age_band],i,age_band, window_size, event_threshold, future_day_range, simulate_dAEFI)
                       
                       #print(f"D_curve_out if type: {D_curve[i].columns}")
                    else:
                       D_curve = df_dvd
                       #print(f"D_curve_out else type: {D_curve[i].columns}")

                    # Determine the color shades based on idx (age_band)
                    shades_1, shades_2 = (color_shades[i] if idx == 0 else color_shades_r[i])
                    # Add the traces with Deaths and all given Doses data to the plot 
                    add_traces_for_data(
                        trace_manager,
                        D_curve, 
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

        if k == 0:
            tracename_prefix = "Avg NUM_D RAW"  # Use RAW for k=0
            colors = px.colors.qualitative.Plotly 
        else:
            tracename_prefix = "Avg NUM_D AEF"  # Use AEF for k=1
            colors = px.colors.qualitative.Plotly_r

        # Loop through each pair of age bands in the age_band_compare list
        # calcualte difference of death curves for each age_band pair
        calculate_age_band_differences(
            tracename_prefix, 
            f"NUM_D {dae}", 
            age_band_compare, 
            trace_manager, 
            window_size_mov_average, 
            flip_axis=True,
            color=colors, 
            normalize=normalize,  # Add the normalize flag
            cum_pop_vd=cum_pop_vd  # Pass cum_pop_vd variable here
        )

        if k == 0:
            tracename_prefix = "Avg NUM_VDA RAW"  # Use RAW for k=0
        else:
            tracename_prefix = "Avg NUM_VDA AEF"  # Use AEF for k=1

        # Loop through each pair of age bands in the age_band_compare list
        # calculate difference of all doses curves for each age_band pair
        calculate_age_band_differences(
            tracename_prefix, 
            f"NUM_VDA {dae}", 
            age_band_compare, 
            trace_manager, 
            window_size_mov_average, 
            flip_axis=False, 
            color=colors,
            normalize=False,  # Add the normalize flag
            cum_pop_vd=cum_pop_vd  # Pass cum_pop_vd variable here
        )

    # Initialize empty list to hold both RAW and AEF traces
    plot_data_combined = []
    html_file_path = rf'C:\CzechFOI-SIM\Plot Results\dAEFI\{plotfile_name} {plot_name_append_text} causalimpact AG_15-85.html'


    dowhy_trace_pairs_raw = [
        (f'Avg NUM_VDA', f'Avg NUM_D')
    ]
    plot_data_combined = plot_causalimpact_for_traces(trace_manager, dowhy_trace_pairs_raw, age_band_compare, html_file_path, plot_data=plot_data_combined)

    
    # Create the layout for the Plotly figure with a combined title
    layout = go.Layout(
        title="Causal Impact Comparison raw Doses_Curve -> raw D_Curve\n(RAW vs. RAW + 1 year AG)",
        xaxis=dict(title="Doses_curve / Age"),
        yaxis=dict(title="D_Curve / Doses per one D"),
        showlegend=True
    )
    
    # Create the figure
    fig = go.Figure(data=plot_data_combined, layout=layout)

    # Save the plot to an HTML file
    fig.write_html(rf"{html_file_path}")
    print(f"Combined plot saved as {html_file_path}")


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



def plot_causalimpact_for_traces(trace_manager, dowhy_trace_pairs, age_band_compare, html_file_path, plot_data=None):
    # Initialize plot_data if not provided
    if plot_data is None:
        plot_data = []

    # Create y_series from the trace data
    y_series = {}
    z = -1
    for trace in trace_manager.figData.data:
        z += 1
        if trace.y is not None and len(trace.y) > 0:
            y_data = np.array(trace.y).flatten()
            y_series[trace.name] = xr.DataArray(y_data, dims='time', coords={'time': np.arange(len(y_data))})
        else:
            print(f"Warning {z}:{trace.name}: has no data or invalid data: {len(trace.y)}")

    # Extracting the two tracename prefixes 
    name1_prefix = f"{dowhy_trace_pairs[0][0]} RAW"
    name2_prefix = f"{dowhy_trace_pairs[0][1]} RAW"
    name3_prefix = f"{dowhy_trace_pairs[0][0]} RAW"
    name4_prefix = f"{dowhy_trace_pairs[0][1]} RAW"

    # Initialize lists to store mean points for RAW and AEF
    mean_raw_points = []
    mean_aef_points = []
    mean_raw_text = []
    mean_aef_text = []

    # Loop through the pairs of trace names
    for idx, (start, end) in enumerate(age_band_compare):
        print(f"Processing Age Band {start}")    
        age_band_extension = f"{start}"
        age_band_extension2 = f"{end}"

        try:
            matching_name1 = [key for key in y_series.keys() if key.startswith(name1_prefix) and key.endswith(age_band_extension)]
            matching_name2 = [key for key in y_series.keys() if key.startswith(name2_prefix) and key.endswith(age_band_extension)]
            matching_name3 = [key for key in y_series.keys() if key.startswith(name3_prefix) and key.endswith(age_band_extension2)]
            matching_name4 = [key for key in y_series.keys() if key.startswith(name4_prefix) and key.endswith(age_band_extension2)]
            
            if matching_name1 and matching_name2 and matching_name3 and matching_name4:
                Doses_curve_RAW = y_series[matching_name1[0]]
                D_Curve_RAW = y_series[matching_name2[0]]
                Doses_curve_AEF = y_series[matching_name3[0]]
                D_Curve_AEF = y_series[matching_name4[0]]
            else:
                print(f"Warning: no matching traces found for :{name1_prefix}*{age_band_extension}:{name2_prefix}*{age_band_extension}:{name3_prefix}*{age_band_extension}:{name4_prefix}*{age_band_extension}: Skipping this age band.")
                continue  # Skip this pair if no match found

        except KeyError:
            print(f"Error: no matching traces found for :{name1_prefix}*{age_band_extension}:{name2_prefix}*{age_band_extension}:{name3_prefix}*{age_band_extension}:{name4_prefix}*{age_band_extension}:. Skipping this age band.")
            continue  # Skip this pair if no match found

        # Replace NaNs with 0 in Doses_curve and D_Curve 
        for k in range(2):
            if k == 0:  # RAW
                Doses_curve = Doses_curve_RAW.fillna(0)
                D_Curve = D_Curve_RAW.fillna(0)
            if k == 1:  # AEF
                Doses_curve = Doses_curve_AEF.fillna(0)
                D_Curve = D_Curve_AEF.fillna(0)

            # --- Causal Inference with DoWhy ---
            print(f"Performing causal analysis for Age Band {start} using DoWhy...")

            # Add the AgeGroup to the data
            ts_data = pd.DataFrame({
                'Doses_curve': Doses_curve,
                'D_Curve': D_Curve,
                'AgeGroup': [start] * len(D_Curve),  # Add the AgeGroup variable
                'Time': np.arange(len(D_Curve))  # Add the Time variable (index of each time step)
            })

            #                'AgeGroup': [start] * len(Doses_curve),  # Add the AgeGroup variable

            # Define a simple causal graph including Age Group and Time as confounders
            causal_graph = """
            digraph {
                Doses_curve -> D_Curve;
                Time -> D_Curve;                
                AgeGroup -> Doses_curve;
                AgeGroup -> D_Curve;
            }
            """
#                AgeGroup -> Doses_curve;
#                AgeGroup -> D_Curve;

            # Initialize the causal model
            model = dowhy.CausalModel(
                data=ts_data,
                treatment="Doses_curve",  
                outcome="D_Curve",        
                graph=causal_graph        
            )

            #model.view_model()
            
            # Perform causal inference            
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=False)
            print(identified_estimand)

            # Estimate the causal effect using a dowhy methode
            try:
                # Use the bootstrap argument (True) to perform bootstrapping
                # causal_estimate = model.estimate_effect(identified_estimand, method_name='backdoor.linear_regression',test_significance=True)
                # Estimation
                causal_estimate = model.estimate_effect(identified_estimand,
                                                method_name="backdoor.linear_regression",
                                                control_value=0.0002,
                                                #treatment_value=1,
                                                #confidence_intervals=False,
                                                test_significance=True)
                print("Causal Estimate is " + str(causal_estimate.value))  

            except Exception as e:
                print(f"Error with Estimator Result: {e}")
                causal_estimate = None          

            # Calculate mean values for Doses_curve and D_Curve
            mean_doses = np.mean(Doses_curve.values)
            mean_d_curve = np.mean(D_Curve.values)

            # Store the causal estimate for the current age band
            causal_effect_value = causal_estimate.value if causal_estimate is not None else np.nan

            print(f"mean_doses {'RAW' if k == 0 else 'AEF'}: {mean_doses}, mean_d_curve: {mean_d_curve}, causal_estimate: {causal_effect_value}")

            # Add the age group information to the hover text
            if k == 0:
                age_group_text = f"Age Group: {start}\nMean Doses: {mean_doses:.6f}\nMean D Curve: {mean_d_curve:.6f}\nCausal Estimate: {causal_effect_value:.8f}"
            if k == 1:    
                age_group_text = f"Age Group: {end}\nMean Doses: {mean_doses:.6f}\nMean D Curve: {mean_d_curve:.6f}\nCausal Estimate: {causal_effect_value:.8f}"

            if k == 0:
                mean_raw_points.append((mean_doses, mean_d_curve, causal_effect_value))
                mean_raw_text.append(age_group_text)
            if k == 1:
                mean_aef_points.append((mean_doses, mean_d_curve, causal_effect_value))
                mean_aef_text.append(age_group_text)

        # Color settings
        colors_r = px.colors.qualitative.Plotly_r
        colors = px.colors.qualitative.Plotly

        # --- Plotting AEF and RAW traces ---
        color_r = colors_r[(idx + 2) % len(colors_r)]
        color = colors[(idx + 2) % len(colors)]

        # Plot the RAW trace
        trace = go.Scatter(
            x=Doses_curve_RAW.values,
            y=D_Curve_RAW.values,
            mode='lines+markers',
            name=f"{name1_prefix}-{name2_prefix} {start}",
            line=dict(color=color, width=1),
            marker=dict(color=color, size=3)
        )
        plot_data.append(trace)

        # Plot the AEF trace
        trace = go.Scatter(
            x=Doses_curve_AEF.values,
            y=D_Curve_AEF.values,
            mode='lines+markers',
            name=f"{name3_prefix}-{name4_prefix}+1 {start}",
            line=dict(color=color_r, width=1),
            marker=dict(color=color_r, size=3)
        )
        plot_data.append(trace)

    # Plot mean AEF and RAW points
    mean_aef_trace = go.Scatter(
        x=[point[0] for point in mean_aef_points],
        y=[point[1] for point in mean_aef_points],
        mode='markers',
        name='mean (RAW+1)',
        marker=dict(color='red', size=10, symbol='x'),
        text=mean_aef_text,
        hoverinfo='text'
    )
    plot_data.append(mean_aef_trace)

    mean_raw_trace = go.Scatter(
        x=[point[0] for point in mean_raw_points],
        y=[point[1] for point in mean_raw_points],
        mode='markers',
        name='mean (RAW)',
        marker=dict(color='blue', size=10, symbol='x'),
        text=mean_raw_text,
        hoverinfo='text'
    )
    plot_data.append(mean_raw_trace)

    # Plot the "Doses per additional death" for AEF and RAW
    doses_per_death_trace = go.Scatter(
        x=[int(start) for start, _ in age_band_compare],
        y=[point[2] for point in mean_aef_points],  # For AEF
        mode='markers',
        name='causal estimate (RAW+1)',
        marker=dict(color='orange', size=10, symbol='x'),
        text=mean_aef_text,
        hoverinfo='text'
    )
    plot_data.append(doses_per_death_trace)

    doses_per_death_raw_trace = go.Scatter(
        x=[int(start) for start, _ in age_band_compare],
        y=[point[2] for point in mean_raw_points],  # For RAW
        mode='markers',
        name='causal estimate (RAW)',
        marker=dict(color='green', size=10, symbol='x'),
        text=mean_raw_text,
        hoverinfo='text'
    )
    plot_data.append(doses_per_death_raw_trace)

    # Calculate the difference in causal estimates (AEF - RAW) for each age group
    causal_estimate_dif = [point_aef[2] - point_raw[2] for point_aef, point_raw in zip(mean_aef_points, mean_raw_points)]
    # Calculate doses per death based on the difference in causal estimates    
    doses_per_death_dif = [(1 / diff) if diff != 0 else np.nan for diff in causal_estimate_dif]
    #doses_per_death_dif =  causal_estimate_dif

    # Plot the Doses per Death Difference (AEF - RAW)
    doses_per_death_diff_trace = go.Scatter(
        x=[int(start) for start, _ in age_band_compare],
        y=doses_per_death_dif,  # Doses per Death Difference (AEF - RAW)
        mode='markers',
        name='Doses per Death difference (RAW+1 -RAW)',
        marker=dict(color='purple', size=10, symbol='x'),
        text=[
            f"Age Group: {start}\n"
            f"Doses per Death difference (RAW+1-RAW): {doses:.8f}\n"
            f"Mean Doses (RAW): {mean_doses:.6f}\n"
            f"Mean D Curve (RAW): {mean_d_curve:.6f}\n"
            f"Causal Estimate (RAW): {point_raw[2]:.8f}\n"
            f"Causal Estimate (RAW+1): {point_aef[2]:.8f}\n"            
            for start, doses, mean_doses, mean_d_curve, point_raw, point_aef in zip(
                age_band_compare, doses_per_death_dif,
                [point[0] for point in mean_raw_points], [point[1] for point in mean_raw_points],
                mean_raw_points, mean_aef_points)
        ],
        hoverinfo='text'
    )
    plot_data.append(doses_per_death_diff_trace)

    return plot_data


def calculate_age_band_differences(tracename_prefix, tracename_new, age_band_compare, trace_manager, window_size_mov_average, flip_axis, color, normalize, cum_pop_vd):    
    totalled_diff = None

    for n, age_band_pair in enumerate(age_band_compare):
        age_band1, age_band2 = age_band_pair[0], age_band_pair[1]

        # Set color palette based on index (even or odd)
        #colors = px.colors.qualitative.Plotly if n % 2 == 0 else px.colors.qualitative.Plotly_r
        colors = color

        # Get the traces for the two age bands
        trace1 = get_trace_by_name(tracename_prefix, age_band1, trace_manager)
        trace2 = get_trace_by_name(tracename_prefix, age_band2, trace_manager)

        # Ensure traces are valid
        if trace1 is None or trace2 is None:
            print(f"Warning: One or both traces for {age_band1} and {age_band2} not found.")
            continue  # Skip this iteration if either trace is missing

        # Validate that traces have matching lengths
        if len(trace1.y) != len(trace2.y):
            print(f"Warning: Trace lengths for {age_band1} and {age_band2} do not match.")

        # Calculate the raw difference in deaths (D curve) between the two age groups
        df_diff = np.array(trace2.y) - np.array(trace1.y)
        df_diff = np.nan_to_num(df_diff, nan=0)  # Replace NaNs with zeros

        # Sum the raw differences across all traces
        totalled_diff = df_diff if totalled_diff is None else totalled_diff + df_diff

        time_indices = np.arange(len(df_diff))

        # Normalize the data if required
        cum_totalled_diff = totalled_diff.cumsum() 
        if normalize:
            cum_totalled_diff_normalized = (cum_totalled_diff / cum_pop_vd) * 100000  # Normalize per 100,000
        else:
            cum_totalled_diff_normalized = cum_totalled_diff  # Keep raw sum

        yaxis = 'y6' if n % 2 == 0 else 'y5'

        # Add the raw difference trace to the plot
        trace_manager.add_trace(
            name=f'DIF-{tracename_new} {age_band1}<br>{age_band2}',
            x=time_indices,  # Use time indices for x-axis
            y=df_diff,
            line=dict(dash='solid', width=1.5, color=colors[(n + 1) % len(colors)]),
            secondary=True,
            axis_assignment=yaxis,
            plot=plot_trace
        )

       # Calculate and add moving average trace for the raw difference
        moving_average_dif = pd.Series(df_diff).rolling(window=window_size_mov_average, center=True).mean()
        trace_manager.add_trace(
            name=f'Avg DIF-{tracename_new} {age_band1}<br>{age_band2}',
            x=time_indices,
            y=moving_average_dif.fillna(0),
            line=dict(dash='solid', width=1.5, color=colors[(n + 3) % len(colors)]),
            secondary=True,
            axis_assignment=yaxis,
            plot=plot_trace
        )

        # Add cumulative difference trace (raw or normalized)
        yaxis = 'y4' if n % 2 == 0 else 'y3'
        trace_manager.add_trace(
            name=f'Cum DIF-{tracename_new} {age_band1}<br>{age_band2}',
            x=time_indices,
            y=cum_totalled_diff_normalized,
            line=dict(dash='dot', width=1.5, color=colors[(n + 2) % len(colors)]),
            secondary=True,
            axis_assignment=yaxis,
            plot=plot_trace
        )


    if normalize:
        sum_totalled_diff_normalized = (totalled_diff / cum_pop_vd) * 100000  # Normalize per 100,000
    else:
        sum_totalled_diff_normalized = totalled_diff  # Keep raw sum

    # Flatten sum_totalled_diff_normalized if it's 2D (to avoid the ValueError)
    if sum_totalled_diff_normalized.ndim > 1:
        sum_totalled_diff_normalized = sum_totalled_diff_normalized.flatten()


    # If any totalled difference was calculated, add it to the plot
    if totalled_diff is not None:
        trace_manager.add_trace(
            name=f'Sum DIF-{tracename_new}',
            x=time_indices,
            y=sum_totalled_diff_normalized,
            line=dict(dash='solid', width=1.5, color=colors[(n + 4) % len(colors)]),
            secondary=True,
            axis_assignment='y6' if flip_axis else 'y5',
        )
        
        # Add moving average for the sum of differences
        moving_average_totalled_diff = pd.Series(sum_totalled_diff_normalized).rolling(window=window_size_mov_average, center=True).mean()
        trace_manager.add_trace(
            name=f'Avg sum DIF-{tracename_new}',
            x=time_indices,
            y=moving_average_totalled_diff.fillna(0),
            line=dict(dash='solid', width=1.5, color=colors[(n + 5) % len(colors)]),
            secondary=True,
            axis_assignment='y6' if flip_axis else 'y5',
        )
       
        if normalize:
            # Apply cumulative sum and normalization per 100,000
            cum_sum_totalled_diff_normalized = (totalled_diff.cumsum() / cum_pop_vd) * 100000
        else:
            # Just keep the raw cumulative sum
            cum_sum_totalled_diff_normalized = totalled_diff.cumsum()

        # Flatten cum_sum_totalled_diff_normalized if it's 2D (to avoid the ValueError)
        if cum_sum_totalled_diff_normalized.ndim > 1:
            cum_sum_totalled_diff_normalized = cum_sum_totalled_diff_normalized.flatten()

        # Add the cumulative sum trace
        trace_manager.add_trace(
            name=f'Sum cum DIF-{tracename_new}',  # Name the sum trace appropriately
            x=time_indices,  # Same time indices for consistency
            y=cum_sum_totalled_diff_normalized,  # Cumulative sum of differences
            line=dict(dash='solid', width=1.5, color=colors[(n + 6) % len(colors)]),  # Customize line style and color for the sum
            secondary=True,  # Use secondary axis (if needed)
            axis_assignment='y4' if flip_axis else 'y3',
        )

    else:
        print("Warning: No differences were calculated, so no cumulative sum was plotted.")


def get_trace_by_name(name0, age_band_extension, trace_manager):
    # print(f"Checking for trace:{name0}: with extension:{age_band_extension}:")
    # Loop through all traces in the figure
    for trace in trace_manager.figData.data:
        # Check if the trace name starts with name0 and ends with the provided age_band_extension (if given)
        # print(f"TRN:{trace.name}")
        if trace.name.startswith(name0) and trace.name.endswith(age_band_extension):
            print(f"Found trace: {trace.name}")
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


def add_dAEFI_To_Trace(D_Curve, Doses_Curve, i, age_band, window_size, event_threshold, future_day_range, simulate_dAEFI):
    # Calculate the moving average of VDA using a rolling window
    mov_average_vd = Doses_Curve.rolling(window=window_size, min_periods=1, center=True).mean()

    # Initialize the cumulative VDA count and other variables
    cumulative_vda = 0
    d_add = 0
    rest = 0

    # Initialize output lists (this will depend on whether you expect lists or DataFrames)
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

            if i is not None and age_band is not None:
                length = len(D_Curve[i][age_band])
            else:
                length = len(D_Curve)
            
            if future_idx < length:  # Ensure the index is within bounds for D_Curve
                # Calculate the number of dAEFI events to add or subtract based on cumulative VDA
                add_d_events = cumulative_vda // event_threshold
                d_add += add_d_events

                if simulate_dAEFI:
                    # print(f"add dAEFI: {cumulative_vda} >= {event_threshold} -> {add_d_events}")
                    # Adjust the D_curve_out and daefi_events_out based on the sign of Doses_Curve value
                    if Doses_Curve[day_idx] > 0:
                        if i is not None and age_band is not None:
                            # Directly modify the list using indices (list indexing, not .loc[])
                            D_curve_out[i].loc[future_idx, str(age_band)] += add_d_events  # Add to D_curve
                            daefi_events_out[i].loc[future_idx, str(age_band)] += add_d_events  # Add to daefi_events
                        else:
                            daefi_events_out[future_idx] += add_d_events  # Add dAEFI event
                            D_curve_out[future_idx] += add_d_events      # Add to D_curve
                    else:
                        if i is not None and age_band is not None:
                            # Directly modify the list using indices (list indexing, not .loc[])
                            D_curve_out[i].loc[future_idx, str(age_band)] += add_d_events  # Add to D_curve
                            daefi_events_out[i].loc[future_idx, str(age_band)] += add_d_events  # Add to daefi_events
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
    mov_average_vd = dataframes_vd[3][age_band].rolling(window=window_size, min_periods=1, center=True).mean()
    
    # Initialize the cumulative VDA count and other variables
    cumulative_vda = 0
    rest = 0

    # Initialize output lists
    daefi_events_out = [0] * len(sin_curves[0])
    sin_curve_out = copy.deepcopy(sin_curves[0])  # Create a deep copy of sin_curves[0]

    # Loop through each day to check if an event should be triggered
    for day_idx in range(len(mov_average_vd)):
        # Get the current day's VDA value (moving average VDA value)
        vda_value = mov_average_vd[day_idx]

        # Skip if the value is NaN
        if math.isnan(vda_value):
            continue

        # Add any remaining "rest" value to the current day's VDA value
        vda_value += rest
        rest = 0  # Reset the remainder

        # Accumulate the VDA value
        cumulative_vda += vda_value

        # Check if the cumulative VDA value triggers an event
        if event_threshold != 0 and abs(cumulative_vda) >= abs(event_threshold):
            # Calculate the number of events to add or subtract
            add_d_events = cumulative_vda // event_threshold  # This will be positive or negative depending on the sign of event_threshold
            rest = cumulative_vda % event_threshold  # Remainder for next accumulation

            # Randomly select a future day in the range [1, future_day_range]
            future_day = random.randint(1, future_day_range)
            future_idx = day_idx + future_day
            if future_idx < len(sin_curve_out):  # Ensure the future index is within bounds
                # Add or subtract dAEFI events in the future (depending on the sign of add_d_events)
                if simulate_dAEFI:  
                    daefi_events_out[future_idx] += add_d_events  # Save to plot later
                    sin_curve_out[future_idx] += add_d_events  # Update sin curve for future day

            # Reset the cumulative VDA after triggering the event
            cumulative_vda = 0

    # Return the updated sin_curve_out and daefi_events_out
    return sin_curve_out, daefi_events_out


# Process sine wave data, fill leading zeros, and assign appropriate color shades.
def process_sine_data(dataframes_dvd, cum_pop_vd, sin_curves,  
                                 age_band, i):
    
    # Initialize a copy of the dataframes_dvd to avoid modifying the original
    df_dvd = [df.copy() for df in dataframes_dvd]
    
    # Process sine waves and assign them to the appropriate columns
    leading_zeros = first_nonzero_index(df_dvd[i][age_band])
    sin_curves[i] = fill_leading_days_with_zeros(sin_curves[i], leading_zeros)

    df_dvd[i][age_band] = sin_curves[i]  # Update the appropriate curve (D, DUVX, DVX, DVDA)
    
    # No normalization here (keep raw data as is for later processing)
    return df_dvd


def add_traces_for_data(trace_manager, df_dvd, df_vd, i, age_band, dae, window_size_mov_average, normalize, shades_1, shades_2, csv_files_dvd, csv_files_vd, cum_pop_vd):
    age_band_extension = age_band.split('-')[0]

    # Add traces for DVD (i == 0 or i == 3)
    if i == 0 or i == 3:
    
        # Normalize the data per 100,000 if required
        if normalize:
            norm_dataframes_dvd = (df_dvd[i][age_band] / cum_pop_vd[i]) * 100000
            plt_dataframes_dvd = norm_dataframes_dvd
            cum_dataframes_dvd = (df_dvd[i][age_band].cumsum() / cum_pop_vd[i]) * 100000
        else:
            plt_dataframes_dvd = df_dvd[i][age_band]
            cum_dataframes_dvd = df_dvd[i][age_band].cumsum()
            
        # Trace for DVD (primary y-axis)
        # print(f'adding:{os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {dae}{age_band_extension}:')
        trace_manager.add_trace(
            name=f'{os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {dae}{age_band_extension}',
            x=df_dvd[i].iloc[:, 0],
            y=plt_dataframes_dvd,
            line=dict(dash='solid', width=1, color=shades_1[0]),
            secondary=False,
            axis_assignment='y1'
        )

        # Moving average trace for DVD
        moving_average_dvd = plt_dataframes_dvd.rolling(window=window_size_mov_average, center=True).mean()
        trace_manager.add_trace(
            name=f'Avg {os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {dae}{age_band_extension}',
            x=df_dvd[i].iloc[:, 0],
            y=moving_average_dvd,
            line=dict(dash='solid', width=1, color=shades_1[1]),
            secondary=False,
            axis_assignment='y1'
        )

        # Cumulative DVD data
        trace_manager.add_trace(
            name=f'cum {os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {dae}{age_band_extension}',
            x=df_dvd[i].iloc[:, 0],
            y=cum_dataframes_dvd,
            line=dict(dash='dot', width=1.5, color=shades_1[2]),
            secondary=True,
            axis_assignment='y3'
        )

    # Add traces for VD data
    trace_manager.add_trace(
        name=f'{os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {dae}{age_band_extension}',
        x=df_vd[i].iloc[:, 0],
        y=df_vd[i][age_band],
        line=dict(dash='solid', width=1, color=shades_2[0]),
        secondary=False,
        axis_assignment='y2'
    )

    # Moving average trace for VD
    moving_average_vd = df_vd[i][age_band].rolling(window=window_size_mov_average, center=True).mean()
    trace_manager.add_trace(
        name=f'Avg {os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {dae}{age_band_extension}',
        x=df_vd[i].iloc[:, 0],
        y=moving_average_vd,
        line=dict(dash='solid', width=1, color=shades_2[1]),
        secondary=False,
        axis_assignment='y2'
    )

    # Cumulative VD data
    trace_manager.add_trace(
        name=f'cum {os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {dae}{age_band_extension}',
        x=df_vd[i].iloc[:, 0],
        y=cum_pop_vd[i],
        line=dict(dash='dot', width=1.5, color=shades_2[2]),
        secondary=True,
        axis_assignment='y4'
    )


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
