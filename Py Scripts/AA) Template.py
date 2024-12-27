import pandas as pd
import plotly.graph_objects as go
import os
import shutil

def main():
    
    plot_name = "PVT_CUM_NUM_D_NORM PVT_CUM_NUM"
    
    # List of CSV file paths
    csv_files_primary = [
        r"C:\CzechFOI-DA\TERRA\PVT_CUM_NUM_UVX_D_NORM.csv",
        r"C:\CzechFOI-DA\TERRA\PVT_CUM_NUM_VX_D_NORM.csv"
    ]

    csv_files_secondary = [
        r"C:\CzechFOI-DA\TERRA\PVT_CUM_NUM_UVX.csv",
        r"C:\CzechFOI-DA\TERRA\PVT_CUM_NUM_VX.csv"
    ]
    
    # Read all CSV files into lists of DataFrames
    dataframes_primary = [pd.read_csv(file) for file in csv_files_primary]
    dataframes_secondary = [pd.read_csv(file) for file in csv_files_secondary]
    
    # Get the list of age bands from the columns (excluding the first column which is 'days')
    age_bands = dataframes_primary[0].columns[1:]
    
    # Initialize the directory and script copy
    full_plotfile_name = init_function(plot_name)
    
    # Create a plot for each age band
    for age_band in age_bands:
        fig = go.Figure()
        
        # Add traces for the primary y-axis
        for df, file in zip(dataframes_primary, csv_files_primary):
            fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df[age_band], mode='lines', name=f'{file} (Primary)'))
        
        # Add traces for the secondary y-axis
        for df, file in zip(dataframes_secondary, csv_files_secondary):
            fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df[age_band], mode='lines', name=f'{file} (Secondary)', yaxis='y2'))              

        fig.update_layout(
            title=f'Age Group cumulated compared : {age_band}',
            xaxis_title='Days',
            yaxis_title='Primary Values',
            yaxis2=dict(
                title='Secondary Values',
                overlaying='y',
                side='right'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="right",
                x=0.8
            ),
            legend_title='CSV Files'
        )
        
        #fig.show()
        fig.write_html(f"{full_plotfile_name} {age_band}.html")
        print(f"Plot for age band {age_band} has been saved to HTML file.")


# Initialize the "Plot Results" directory and copy this PyScript into it
def init_function(plot_name):
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    first_root_dir = os.path.abspath(os.sep.join(script_dir.split(os.sep)[:2]))
    plot_result_path = os.path.join(first_root_dir, "Plot Results", f"{script_name} {plot_name}")
    os.makedirs(plot_result_path, exist_ok=True)
    
    script_file_name = os.path.join(plot_result_path, f"{script_name}.py")
    if not os.path.exists(script_file_name):
        shutil.copy(__file__, script_file_name)
    # return plot file name without extension
    return plot_result_path + os.sep + f"{script_name} {plot_name}"

# call main script
if __name__ == "__main__": main()
