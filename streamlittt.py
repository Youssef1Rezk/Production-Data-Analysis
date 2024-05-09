# import the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st
import matplotlib.animation as animation
from datetime import datetime
from scipy.optimize import curve_fit
from PIL import Image
import requests
from io import BytesIO


# Load the data
# GitHub raw content URL of the Excel file
excel_file_url = 'https://raw.githubusercontent.com/Youssef1Rezk/Production-Data-Analysis/main/Volve%20production%20data.xlsx'

# Read the Excel file
df = pd.read_excel(excel_file_url)
# GitHub raw content URL of the image
image_url = 'https://raw.githubusercontent.com/Youssef1Rezk/Production-Data-Analysis/main/spe-logo-blue.png__314x181_q85_subsampling-2.png'

# Use requests to retrieve the image from the GitHub URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
# Display logo
st.sidebar.image(image,  use_column_width=True)
# Add title and description
st.title("Production Data Analysis")
st.write("""
         ## Explore the production data from Volve field
         Welcome to the Volve Field production data analysis app! 
         Here, you can explore various insights derived from the production data.
         """)

# some functiions needed:
# Define the color list for the plot
colors_list = ['#FFBF00', '#FF7F50', '#DE3163', '#40E0D0', '#6495ED', '#CCCCFF']

# Initialize session state for selected_volume_type if it doesn't exist
if 'selected_volume_type' not in st.session_state:
    st.session_state['selected_volume_type'] = "BORE_OIL_VOL"
# Calculate GOR and create a new dataframe
df['GOR'] = df['BORE_GAS_VOL'] / df['BORE_OIL_VOL']
df7 = df[['DATEPRD', 'NPD_WELL_BORE_NAME', 'GOR']]
df7 = df7.query("NPD_WELL_BORE_NAME != '15/9-F-4'")
df['DATEPRD'] = pd.to_datetime(df['DATEPRD'])
# Filter the dataframe to remove data associated with '15/9-F-4'
filtered_df = df[df['WELL_BORE_CODE'] != '15/9-F-4']
# Define the plotting function with time range slider
def plot_gor(well_bore_name, df7):
    # Get the date range for the selected well
    well_data = df7[df7['NPD_WELL_BORE_NAME'] == well_bore_name]
    min_date = well_data['DATEPRD'].min()
    max_date = well_data['DATEPRD'].max()

    # Convert min_date and max_date to datetime.date objects
    min_date = min_date.to_pydatetime().date()
    max_date = max_date.to_pydatetime().date()

    # Create a time range slider
    start_date, end_date = st.slider(
        'Select the date range',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Filter the dataframe based on the selected date range
    df_well = well_data[(well_data['DATEPRD'] >= pd.to_datetime(start_date)) & 
                        (well_data['DATEPRD'] <= pd.to_datetime(end_date))]

    # Plotting code
    fig, ax = plt.subplots(figsize=(15, 4))
    sns.lineplot(x="DATEPRD", y="GOR", data=df_well, hue="NPD_WELL_BORE_NAME", palette="Dark2", linewidth=1)
    plt.title(f"GOR for {well_bore_name} from {start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("GOR (m³/m³)")
    plt.legend(title='Well Name')
    st.pyplot(fig)

# Define the plotting function with time range slider
def plot_production_profile(well_bore_code, df):

    # Get the date range for the selected well
    well_data = filtered_df[filtered_df['WELL_BORE_CODE'] == well_bore_code]
    min_date = well_data['DATEPRD'].min()
    max_date = well_data['DATEPRD'].max()

    # Convert min_date and max_date to datetime.date objects
    min_date = min_date.to_pydatetime().date()
    max_date = max_date.to_pydatetime().date()
    
    # Create a time range slider with a unique key using key_suffix
    start_date, end_date = st.slider(
        'Select the date range',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
        key=f'date_range_slider_{well_bore_code}'  # Unique key for the slider
    )
    
    # Convert start_date and end_date to the same type as 'DATEPRD'
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the data based on the selected date range
    filtered_data = well_data[(well_data['DATEPRD'] >= start_date) & (well_data['DATEPRD'] <= end_date)]

    # Plotting the production profile
    plt.figure(figsize=(8, 4))
    plt.plot(filtered_data['DATEPRD'], filtered_data['BORE_OIL_VOL'], label='oil', color='#FFBF00')
    plt.plot(filtered_data['DATEPRD'], filtered_data['BORE_WAT_VOL'], label='water', color='#6495ED')
    plt.xlabel("Time")
    plt.ylabel("Production")
    plt.title(f"Production Profile for Well: {well_bore_code}")
    plt.legend()
    plt.grid(which='major', linestyle='-', alpha=.5, color="#6666")
    plt.minorticks_on()
    plt.grid(which='minor', linestyle='--', alpha=.2, color="#9999")
    st.pyplot(plt.gcf())  # Display the plot in Streamlit
# Define a function to plot total production by year with a date range slider
def plot_total_production_by_year_with_slider(df):
    # Create a date range selection bar
    start_date_yearly, end_date_yearly = st.slider(
        'Select the date range',
        min_value=df['DATEPRD'].min().to_pydatetime().date(),
        max_value=df['DATEPRD'].max().to_pydatetime().date(),
        value=(df['DATEPRD'].min().to_pydatetime().date(), df['DATEPRD'].max().to_pydatetime().date()),
        format="YYYY-MM-DD",
        key='date_range_slider_total_production'  # Unique key for the slider
    )

    # Filter the dataframe based on the selected date range
    filtered_df_yearly = df[(df['DATEPRD'] >= pd.to_datetime(start_date_yearly)) & 
                        (df['DATEPRD'] <= pd.to_datetime(end_date_yearly))]

    # Group by year and sum the oil and water production
    production_by_year = filtered_df_yearly.groupby(filtered_df_yearly["DATEPRD"].dt.year)[["BORE_OIL_VOL", "BORE_WAT_VOL"]].sum()

    # Plot the total production for each year
    fig, ax = plt.subplots(figsize=(10, 6))
    production_by_year.plot(kind="line", ax=ax)
    plt.title("Total Production of Oil and Water Each Year")
    plt.xlabel("Year")
    plt.ylabel("Production")
    plt.grid(which="major", linestyle="-", alpha=.5, color="#6666")
    plt.minorticks_on()
    plt.grid(which="minor", linestyle="--", alpha=.2, color="#9999")

    # Show the plot directly
    st.pyplot(fig)

# Define a function to plot total production by year with a date range slider
def plot_total_production_by_year_with_slider2(df):
    # Create a date range selection bar
    start_date_yearly, end_date_yearly = st.slider(
        'Select the date range',
        min_value=df['DATEPRD'].min().to_pydatetime().date(),
        max_value=df['DATEPRD'].max().to_pydatetime().date(),
        value=(df['DATEPRD'].min().to_pydatetime().date(), df['DATEPRD'].max().to_pydatetime().date()),
        format="YYYY-MM-DD",
        key='date_range_slider_total_production_2'  # Unique key for the slider
    )

    # Filter the dataframe based on the selected date range
    filtered_df_yearly = df[(df['DATEPRD'] >= pd.to_datetime(start_date_yearly)) & 
                        (df['DATEPRD'] <= pd.to_datetime(end_date_yearly))]

    # Group by year and sum the oil and water production
    production_by_year = filtered_df_yearly.groupby(filtered_df_yearly["DATEPRD"].dt.year)[["BORE_OIL_VOL", "BORE_WAT_VOL","BORE_GAS_VOL"]].sum()

    # Plot the total production for each year
    fig, ax = plt.subplots(figsize=(10, 6))
    production_by_year.plot(kind="line", ax=ax)
    plt.title("Total Production of Oil, Water, and Gas Each Year")
    plt.xlabel("Year")
    plt.ylabel("Production")
    plt.grid(which="major", linestyle="-", alpha=.5, color="#6666")
    plt.minorticks_on()
    plt.grid(which="minor", linestyle="--", alpha=.2, color="#9999")

    # Show the plot directly
    st.pyplot(fig)

# Define a function to plot total production by year with a date range slider
def plot_total_production_by_year_with_slider2(df):
    # Create a date range selection bar
    start_date_yearly, end_date_yearly = st.slider(
        'Select the date range',
        min_value=df['DATEPRD'].min().to_pydatetime().date(),
        max_value=df['DATEPRD'].max().to_pydatetime().date(),
        value=(df['DATEPRD'].min().to_pydatetime().date(), df['DATEPRD'].max().to_pydatetime().date()),
        format="YYYY-MM-DD"
    )

    # Filter the dataframe based on the selected date range
    filtered_df_yearly = df[(df['DATEPRD'] >= pd.to_datetime(start_date_yearly)) & 
                        (df['DATEPRD'] <= pd.to_datetime(end_date_yearly))]

    # Group by year and sum the oil and water production
    production_by_year = filtered_df_yearly.groupby(filtered_df_yearly["DATEPRD"].dt.year)[["BORE_OIL_VOL", "BORE_WAT_VOL","BORE_GAS_VOL"]].sum()

    # Plot the total production for each year
    fig, ax = plt.subplots(figsize=(10, 6))
    production_by_year.plot(kind="line", ax=ax)
    plt.title("Total Production of Oil and Water Each Year")
    plt.xlabel("Year")
    plt.ylabel("Production")
    plt.grid(which="major", linestyle="-", alpha=.5, color="#6666")
    plt.minorticks_on()
    plt.grid(which="minor", linestyle="--", alpha=.2, color="#9999")

    # Show the plot directly
    st.pyplot(fig)


# Create a sidebar radio button for navigation
selected_page = st.sidebar.radio(
    "Navigation",
    ["Show Data", "Part 1", "Part 2","part 3"]
)

# Show Data page
if selected_page == "Show Data":
    st.write("Data Overview")
    st.dataframe(df)

# Part 1 page
elif selected_page == "Part 1":
    st.subheader('What is the contribution of each well in oil, gas, water production?')
    df2 = df[['NPD_WELL_BORE_NAME','BORE_OIL_VOL','BORE_GAS_VOL','BORE_WAT_VOL']]
    df3 = df2.groupby('NPD_WELL_BORE_NAME').sum().drop(index='15/9-F-4')
    for col in df3.columns:
        total_sum = df3[col].sum()
        df3[col] = (df3[col] / total_sum) * 100
    df3 = df3.applymap(lambda x: f'{round(x, 2)} %')
    st.dataframe(df3)

    st.subheader('For every month in the data, what is the average oil production?')
    df['monthly_date'] = df['DATEPRD'].dt.to_period('M')
    d2 = df.groupby('monthly_date')['BORE_OIL_VOL'].mean()
    st.dataframe(d2)

    st.subheader('Which wells are injectors and which are producers?')
    unique_flow_kind = df.groupby(['NPD_WELL_BORE_NAME'])['FLOW_KIND'].unique()
    st.dataframe(unique_flow_kind)

    st.subheader('Which well has worked the most?')
    most_worked_well = df.groupby(['NPD_WELL_BORE_NAME'])['ON_STREAM_HRS'].sum().sort_values(ascending=False)
    st.dataframe(most_worked_well)



# Part 2 page
elif selected_page == "Part 2":
    # Set a flag in session state to indicate the button has been pressed
    st.session_state['part2_active'] = True
# Reset part2_active flag if Part 2 is not selected
if selected_page != "Part 2":
    # Reset part2_active flag
    st.session_state['part2_active'] = False


# Use the flag to keep the dropdown and plot active even after reruns
if 'part2_active' in st.session_state and st.session_state['part2_active']:
    # Title for the plot selection
    st.subheader('The contribution of each well in oil, gas, water production.')
        
    # Group by well name and calculate the sum
    d = df.groupby("NPD_WELL_BORE_NAME")[["BORE_OIL_VOL", "BORE_WAT_VOL", "BORE_GAS_VOL"]].sum().drop(index="15/9-F-4")

    # Convert to percentage
    for col in ["BORE_OIL_VOL", "BORE_WAT_VOL", "BORE_GAS_VOL"]:
        d[col] = (d[col] / d[col].sum()) * 100

    # Format the numbers as percentages
    d = d.applymap(lambda x: f'{round(x, 2)} %')

    # Display the dataframe
    st.dataframe(d)

    # Dropdown menu for selecting the volume type with an on_change callback
    selected_volume_type = st.selectbox(
        'Choose the volume type to display',
        ["BORE_OIL_VOL", "BORE_WAT_VOL", "BORE_GAS_VOL"],
        index=["BORE_OIL_VOL", "BORE_WAT_VOL", "BORE_GAS_VOL"].index(st.session_state.get('selected_volume_type', 'BORE_OIL_VOL')),
        key='volume_type_select'  # Adding a key to ensure the widget state is preserved correctly
    )

    # Update the session state immediately
    st.session_state['selected_volume_type'] = selected_volume_type

    # Title for the plot selection
    st.subheader(f'The contribution of each well in {selected_volume_type}')

    # Group by well name and calculate the sum for the selected volume type
    d = df.groupby("NPD_WELL_BORE_NAME")[selected_volume_type].sum().drop(index="15/9-F-4")

    # Convert to percentage
    d_pct = (d / d.sum()) * 100

    # Create the bar plot with a larger figure size
    fig, ax = plt.subplots(figsize=(20, 8))
    d_pct.plot(kind='bar', color=colors_list[:len(d)], edgecolor=None, ax=ax)
    plt.title(f"Contribution of each well in {selected_volume_type}", fontsize=16)

    # Customize the ticks and spines
    plt.xticks(fontsize=16, rotation=45)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.yticks([])

    # Annotate the bars with the percentage values
    for p, color in zip(ax.patches, colors_list):
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() 
        ax.annotate(f'{height:.2f}%', (x + width/2, y + height*1.02), ha='center', color=color, fontsize=17)

    # Adjust the legend position
    plt.legend([p for p in ax.patches], [well for well in d.index], loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=len(d.index), fontsize=14)

    # Show the plot in Streamlit
    st.pyplot(fig)

    # Title for the second plot selection
    st.subheader('Total production of each well [oil, water]')

    # Calculate the total production of oil and water for each well
    df5 = df.groupby("NPD_WELL_BORE_NAME")[["BORE_OIL_VOL", "BORE_WAT_VOL"]].sum().drop(index="15/9-F-4")
    df5['total production'] = df5['BORE_OIL_VOL'] + df5['BORE_WAT_VOL']

    # Drop the individual oil and water volume columns to keep only the total production
    df6 = df5.drop(columns=['BORE_OIL_VOL', 'BORE_WAT_VOL'])

    # Create the horizontal bar plot for total production
    fig, ax = plt.subplots(figsize=(10, 8))
    df6['total production'].plot(kind='barh', color='#6495ED', edgecolor=None, ax=ax)
    plt.title('Total Production of Each Well [Oil, Water]', fontsize=16)

    # Customize the ticks and spines
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Annotate the bars with the total production values
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() 

    # Show the plot in Streamlit
    st.pyplot(fig)

    # Calculate GOR and create a new dataframe
    df['GOR'] = df['BORE_GAS_VOL'] / df['BORE_OIL_VOL']
    df7 = df[['DATEPRD', 'NPD_WELL_BORE_NAME', 'GOR']]
    df7 = df7.query("NPD_WELL_BORE_NAME != '15/9-F-4'")

    # Streamlit UI components for GOR profile
    st.subheader('Profile of the GOR for Each Well with Time')
    selected_well = st.selectbox('Select a Well', options=df7['NPD_WELL_BORE_NAME'].unique())
    plot_gor(selected_well, df7)
    
     # Streamlit UI component to select a well for production profile
    st.subheader('Production profile for each well')
    selected_well_profile = st.selectbox(
    'Select a Well for Production Profile', options=df[df['WELL_BORE_CODE'] != '15/9-F-4']['WELL_BORE_CODE'].unique())
    
    # Call the plotting function with the selected well
    plot_production_profile(selected_well_profile, df)
    # Title for the fifth plot selection
    st.subheader('Total Production of Oil and Water in Each Year')
    # Call the function to plot total production by year with the date range slider
    plot_total_production_by_year_with_slider(df)
    # Title for the sixth plot selection
    st.subheader('Production of TOATL oil and water and gas in each year')
    # Call the function to plot total production by year with the date range slider
    plot_total_production_by_year_with_slider2(df)
# Part 3 page
elif selected_page == "part 3":
    st.subheader("Oil Production of 15/9-F-14")
    
    # Filter the DataFrame for the specific well and remove zero oil production entries
    df_part3 = df[df["NPD_WELL_BORE_NAME"] == '15/9-F-14']
    df_part3 = df_part3[df_part3["BORE_OIL_VOL"] != 0]
    
    # Define the outlier_treatment function
    def outlier_treatment(datacolumn):
        sorted(datacolumn)
        Q1, Q3 = np.percentile(datacolumn, [25, 75])
        IQR = Q3 - Q1
        lower_range = Q1 - (1.5 * IQR)
        upper_range = Q3 + (1.5 * IQR)
        return lower_range, upper_range

    # Apply outlier treatment to oil volume data
    lowerbound, upperbound = outlier_treatment(df_part3["BORE_OIL_VOL"])
    df_part3.drop(df_part3[(df_part3["BORE_OIL_VOL"] > upperbound) | (df_part3["BORE_OIL_VOL"] < lowerbound)].index, inplace=True)

    # Slider for smoothing window
    smoothing_window = st.slider('Select smoothing window size', min_value=1, max_value=100, value=70, step=1)
    
    # Apply rolling mean with the selected window size
    df_part3['oil_smoothed'] = df_part3["BORE_OIL_VOL"].rolling(window=smoothing_window, center=True).mean()

    # Get the date range for the selected well
    min_date = df_part3['DATEPRD'].min()
    max_date = df_part3['DATEPRD'].max()

    # Convert min_date and max_date to datetime.date objects
    min_date = min_date.to_pydatetime().date()
    max_date = max_date.to_pydatetime().date()

    # Create a time range slider
    start_date, end_date = st.slider(
        'Select the date range',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Filter the dataframe based on the selected date range
    df_part3_filtered = df_part3[(df_part3['DATEPRD'] >= pd.to_datetime(start_date)) & 
                                  (df_part3['DATEPRD'] <= pd.to_datetime(end_date))]

    # Create scatter plot for oil production
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.scatter(df_part3_filtered['DATEPRD'], df_part3_filtered['oil_smoothed'])
    plt.xlabel("Date")
    plt.ylabel("Smoothed Oil Production")
    plt.title("Smoothed Oil Production of 15/9-F-14 within Selected Date Range")
    st.pyplot(fig)
     # Second title for the effect of smoothing
    st.subheader("Show Effect of Smoothing")

    # Create a time range slider for the period of the smoothing effect
    start_date, end_date = st.slider(
        'Select the period for smoothing effect',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Filter the dataframe based on the selected period
    df_period = df_part3[(df_part3['DATEPRD'] >= pd.to_datetime(start_date)) & 
                         (df_part3['DATEPRD'] <= pd.to_datetime(end_date))]

    # Plotting the data
    plt.figure(figsize=(18, 10))
    plt.plot(df_period["DATEPRD"], df_period["BORE_OIL_VOL"], label="Original", color="orange")
    plt.plot(df_period["DATEPRD"], df_period["oil_smoothed"], label="Smoothed", color="blue")
    plt.legend()
    plt.grid(which="major", color="#6666", linestyle="-", alpha=0.5)
    plt.grid(which="minor", color="#9999", linestyle="-", alpha=0.1)
    plt.minorticks_on()
    plt.xlabel("Date")
    plt.ylabel("Oil Production")
    plt.title("Oil Production and Smoothing Effect of 15/9-F-14")
    st.pyplot(plt.gcf())
    # Third title for curve-fitting
    st.subheader("Curve-Fitting")

            # Filter the DataFrame for the specific well '15/9-F-14' and remove zero oil production entries
    df_filtered = df[(df["NPD_WELL_BORE_NAME"] == '15/9-F-14') & (df["BORE_OIL_VOL"] != 0)]

    # Calculate the days from the date for the specific well
    df_filtered["days"] = (df_filtered["DATEPRD"] - df_filtered["DATEPRD"].min()).dt.days

    # Apply rolling mean with a window size of 70 to smooth the oil volume data
    df_filtered['oil_smoothed'] = df_filtered["BORE_OIL_VOL"].rolling(window=70, center=True).mean()
    df_filtered = df_filtered[["oil_smoothed", "days"]].dropna()

    # Get T and Q for fitting
    T = df_filtered["days"]
    Q = df_filtered["oil_smoothed"]

    # Normalize the time and rate data
    T_normalized = T / max(T)
    Q_normalized = Q / max(Q)

    # Exponential model function
    def exponential(t, qi, di):
        return qi * np.exp(-di * t)

    # Harmonic model function
    def harmonic(t, qi, di):
        return qi / (1 + di * t)

    # Hyperbolic model function
    def hyperbolic(t, qi, di, b):
        return qi / ((1 + b * di * t)**(1/b))

    # Fitting function
    def fit_model(model_func, T, Q, p0=None):
        T_normalized = T / max(T)
        Q_normalized = Q / max(Q)
        params, _ = curve_fit(model_func, T_normalized, Q_normalized, p0=p0)
        qi, di = params[:2]
        qi = qi * max(Q)
        di = di / max(T)
        b = params[2] if len(params) > 2 else 0
        return qi, di, b

    # Fit models
    qi_exp, di_exp, _ = fit_model(exponential, T, Q)
    qi_harm, di_harm, _ = fit_model(harmonic, T, Q)
    qi_hyp, di_hyp, b_hyp = fit_model(hyperbolic, T, Q, p0=[1, 1, 1])

    # Generate model curves
    Q_exp = exponential(T, qi_exp, di_exp)
    Q_harm = harmonic(T, qi_harm, di_harm)
    Q_hyp = hyperbolic(T, qi_hyp, di_hyp, b_hyp)

    # Plotting function
    def plot_model(model):
        plt.figure(figsize=(10, 6))
        plt.plot(T, Q, label='Smoothed', color="green")
        if model == 'exponential' or model == 'all':
            plt.plot(T, Q_exp, label='Exponential', color="blue")
        if model == 'harmonic' or model == 'all':
            plt.plot(T, Q_harm, label='Harmonic', color="red")
        if model == 'hyperbolic' or model == 'all':
            plt.plot(T, Q_hyp, label='Hyperbolic', color="yellow")
        plt.legend()
        plt.grid(which="major", color="#6666", linestyle='-', alpha=.5)
        plt.grid(which="minor", color="#9999", linestyle='-', alpha=.1)
        plt.minorticks_on()
        plt.xlabel("Days")
        plt.ylabel("Oil Production (Smoothed)")
        plt.title("Curve Fitting Models for Oil Production of Well 15/9-F-14")
        st.pyplot(plt.gcf())

    # Dropdown for model selection in Streamlit
    model_selected = st.selectbox('Select a model:', ['exponential', 'harmonic', 'hyperbolic', 'all'])

    # Display the interactive plot based on the selected model
    plot_model(model_selected)
