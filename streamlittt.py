# import the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import base64
import requests
import matplotlib.animation as animation
from datetime import datetime
from scipy.optimize import curve_fit
from PIL import Image
import requests
from io import BytesIO
import streamlit.components.v1 as components
image = np.random.rand(128, 128, 3)  # A random color to demonstrate loading an image is not required

st.markdown(
    f"""
    <style>
    .reportview-container .main {{
        background-image: url('https://raw.githubusercontent.com/Youssef1Rezk/Production-Data-Analysis/4d67db33134fdd724a62cbb5328163ac0b12e601/pexels-johannes-havn-835931-1716008.jpg') !important;
        background-size: cover !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Function to fetch image from a URL
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://raw.githubusercontent.com/Youssef1Rezk/Production-Data-Analysis/2b8b49a8370e73a47617460edadebfc13b5e386b/6436964_3293677.jpg");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

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
st.markdown("""
             <h1 style='color: white; text-align: center;'>Volve Field Production Data Analysis</h1>
             <h2 style='color: white; text-align: center;'>Welcome to the SPE Science Fair!</h2>
             <p style='color: white; text-align: center;'>
             Explore the production data from the Volve field in this interactive analysis app. 
             Dive into various insights and discover patterns within the production data.
             </p>
             """, unsafe_allow_html=True)


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
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 4))
    sns.lineplot(x="DATEPRD", y="GOR", data=df_well, hue="NPD_WELL_BORE_NAME", palette="muted", linewidth=2.5)

    # Customize the plot for a better look
    plt.title(f"GOR for {well_bore_name} from {start_date} to {end_date}", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("GOR (m³/m³)", fontsize=14)
    plt.legend(title='Well Name', fontsize=12)
    sns.despine(fig)  # Remove the top and right spines from plot

    # Show the interactive plot in Streamlit
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
    # Define colors for each type of production volume
    colors = {'BORE_OIL_VOL': 'orange', 'BORE_WAT_VOL': 'blue'}
    # Plotting the production profile with Plotly
    fig = px.line(filtered_data, x='DATEPRD', y=['BORE_OIL_VOL', 'BORE_WAT_VOL'],
                  labels={'DATEPRD': 'Date', 'value': 'Volume', 'variable': 'Type'},
                  title=f"Production Profile for Well: {well_bore_code}",color_discrete_map=colors)
    
    # Update traces to customize the appearance
    fig.update_traces(line=dict(width=2))
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("BORE_OIL_VOL=", "Oil Volume")))
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("BORE_WAT_VOL=", "Water Volume")))
    
    # Update layout to control the size and position
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Glassy white background
        paper_bgcolor='rgba(0, 71, 171, 0.5)',  # Semi-transparent blue background
        margin=dict(l=50, r=20, t=50, b=40),  # Adjust margins to fit the chart within the frame
        font=dict(family="Arial", size=12, color="white"),
        xaxis=dict(linecolor='black', linewidth=2, mirror=True),
        yaxis=dict(linecolor='black', linewidth=2, mirror=True),
        legend=dict(title='Production Type', bgcolor='rgba(255, 255, 255, 0.4)', bordercolor='black', borderwidth=2),
        title=dict(font=dict(size=16, color='white')),
        hoverlabel=dict(bgcolor='white', font_size=12, font_family="Arial"),
        hovermode="closest"
    )

   # Convert the Plotly figure to HTML and store it in a variable
    fig_html = fig.to_html()

    # Custom HTML and CSS to create a container with rounded corners
    rounded_corner_container = f"""
    <div style='border-radius: 15px; overflow: hidden;'>
        {fig_html}
    </div>
    """

    # Use Streamlit's HTML component to render the custom container
    components.html(rounded_corner_container, height=600,width=710)
    st.plotly_chart(fig, use_container_width=True)




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
    production_by_year = filtered_df_yearly.groupby(filtered_df_yearly["DATEPRD"].dt.year)[["BORE_OIL_VOL", "BORE_WAT_VOL"]].sum().reset_index()
    # Define colors for each type of production volume
    colors = {'BORE_OIL_VOL': 'orange', 'BORE_WAT_VOL': 'blue'}
    # Plot the total production for each year using Plotly
    fig = px.line(production_by_year, x='DATEPRD', y=['BORE_OIL_VOL', 'BORE_WAT_VOL'],
                  labels={'DATEPRD': 'Year', 'value': 'Production Volume', 'variable': 'Type'},
                  title="Total Production of Oil and Water Each Year",color_discrete_map=colors)
    
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Glassy white background
        paper_bgcolor='rgba(0, 71, 171, 0.5)',  # Semi-transparent blue background
        margin=dict(l=50, r=20, t=50, b=40),  # Adjust margins to fit the chart within the frame
        font=dict(family="Arial", size=12, color="white"),
        xaxis=dict(linecolor='black', linewidth=2, mirror=True),
        yaxis=dict(linecolor='black', linewidth=2, mirror=True),
        legend=dict(title='Production Type', bgcolor='rgba(255, 255, 255, 0.4)', bordercolor='black', borderwidth=2),
        title=dict(font=dict(size=16, color='white')),
        hoverlabel=dict(bgcolor='white', font_size=12, font_family="Arial"),
        hovermode="closest"
    )

    # Convert the Plotly figure to HTML and store it in a variable
    fig_html = fig.to_html()

    # Custom HTML and CSS to create a container with rounded corners
    rounded_corner_container = f"""
    <div style='border-radius: 15px; overflow: hidden;'>
        {fig_html}
    </div>
    """

    # Use Streamlit's HTML component to render the custom container
    components.html(rounded_corner_container, height=600,width=710)

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

    # Group by year and sum the oil, water, and gas production
    production_by_year = filtered_df_yearly.groupby(filtered_df_yearly["DATEPRD"].dt.year)[["BORE_OIL_VOL", "BORE_WAT_VOL", "BORE_GAS_VOL"]].sum().reset_index()
    # Define colors for each type of production volume
    colors = {'BORE_OIL_VOL': 'orange', 'BORE_WAT_VOL': 'blue','BORE_GAS_VOL' :'red'}
    # Plot the total production for each year using Plotly
    fig = px.line(production_by_year, x='DATEPRD', y=['BORE_OIL_VOL', 'BORE_WAT_VOL', 'BORE_GAS_VOL'],
                  labels={'DATEPRD': 'Year', 'value': 'Production Volume', 'variable': 'Type'},
                  title="Total Production of Oil, Water, and Gas Each Year",color_discrete_map=colors)
    
    # Update layout for a glassy look and rounded edges
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Glassy white background
        paper_bgcolor='rgba(0, 71, 171, 0.5)',  # Semi-transparent blue background
        margin=dict(l=50, r=20, t=50, b=40),  # Adjust margins to fit the chart within the frame,
        font=dict(family="Arial", size=12, color="white"),
        xaxis=dict(linecolor='black', linewidth=2, mirror=True),
        yaxis=dict(linecolor='black', linewidth=2, mirror=True),
        legend=dict(title='Production Type', bgcolor='rgba(255, 255, 255, 0.4)', bordercolor='black', borderwidth=2),
        title=dict(font=dict(size=16, color='white')),
        hoverlabel=dict(bgcolor='white', font_size=12, font_family="Arial"),
        hovermode="closest"
    )

    # Convert the Plotly figure to HTML and store it in a variable
    fig_html = fig.to_html()

    # Custom HTML and CSS to create a container with rounded corners
    rounded_corner_container = f"""
    <div style='border-radius: 15px; overflow: hidden;'>
        {fig_html}
    </div>
    """

    # Use Streamlit's HTML component to render the custom container
    components.html(rounded_corner_container, height=600,width=710)

# Function to get base64 of the binary file
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as file:
        data = file.read()
    return base64.b64encode(data).decode()

# Create a sidebar radio button for navigation
selected_page = st.sidebar.radio(
    "Navigation",
    ["Show Data", "Part 1", "Part 2","part 3","About the Project"]
)

# Show Data page
if selected_page == "Show Data":
    st.write("Data Overview")
    st.dataframe(df)

# Part 1 page
elif selected_page == "Part 1":
    st.markdown('<h2 style="color: white;">What is the contribution of each well in oil, gas, water production?</h2>', unsafe_allow_html=True)
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

    # Create the bar plot using Plotly
    fig = px.bar(
        d_pct.reset_index(),
        x='NPD_WELL_BORE_NAME',
        y=selected_volume_type,
        text=d_pct.apply(lambda x: f'{x:.2f}%'),
        title=f"Contribution of each well in {selected_volume_type}",
        color=selected_volume_type,
        color_continuous_scale=['blue', 'red']  # Gradient from orange to blue
    )

    # Customize the layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
    plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Glassy white background
    paper_bgcolor='rgba(0, 71, 171, 0.5)',  # Semi-transparent blue background
    margin=dict(l=60, r=20, t=50, b=40),  # Adjust margins to fit the chart within the frame
    font=dict(family="Arial", size=12, color="white"),
    xaxis_title="Well Name",
    yaxis_title="Percentage Contribution",
    yaxis=dict(tickformat=".2f"),
    coloraxis_colorbar=dict(title='Percentage Contribution'),  # Add a color bar title
    legend=dict(title='Production Type', bgcolor='rgba(255, 255, 255, 0.4)', bordercolor='black', borderwidth=2),
    title=dict(font=dict(size=16, color='white')),
    hoverlabel=dict(bgcolor='white', font_size=12, font_family="Arial"),
    hovermode="closest"
 )

    # Convert the Plotly figure to HTML and store it in a variable
    fig_html = fig.to_html()

    # Custom HTML and CSS to create a container with rounded corners
    rounded_corner_container = f"""
    <div style='border-radius: 15px; overflow: hidden;'>
        {fig_html}
    </div>
    """

    # Use Streamlit's HTML component to render the custom container
    components.html(rounded_corner_container, height=600,width=710)

    # Title for the second plot selection
    st.subheader('Total production of each well [oil, water]')

    # Calculate the total production of oil and water for each well
    df5 = df.groupby("NPD_WELL_BORE_NAME")[["BORE_OIL_VOL", "BORE_WAT_VOL"]].sum().drop(index="15/9-F-4")
    df5['total production'] = df5['BORE_OIL_VOL'] + df5['BORE_WAT_VOL']

    # Drop the individual oil and water volume columns to keep only the total production
    df6 = df5.drop(columns=['BORE_OIL_VOL', 'BORE_WAT_VOL']).reset_index()

    # Create the horizontal bar plot for total production using Plotly
    fig = px.bar(df6, y='NPD_WELL_BORE_NAME', x='total production', orientation='h',
                title='Total Production of Each Well [Oil, Water]',
                labels={'NPD_WELL_BORE_NAME': 'Well Name', 'total production': 'Total Production'},
                color='total production',
                color_continuous_scale=['blue', 'red'])  # Gradient from orange to blue

    # Customize the layout for a glassy look and rounded edges
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Glassy white background
        paper_bgcolor='rgba(0, 71, 171, 0.5)',  # Semi-transparent blue background
        margin=dict(l=90, r=20, t=50, b=40),  # Adjust margins to fit the chart within the frame,
        font=dict(family="Arial", size=12, color="white"),
        xaxis_title="Total Production",
        yaxis_title="Well Name",
        coloraxis_colorbar=dict(title='Total Production'),  # Add a color bar title
        legend=dict(title='Production Type', bgcolor='rgba(255, 255, 255, 0.4)', bordercolor='black', borderwidth=2),
        title=dict(font=dict(size=16, color='white')),
        hoverlabel=dict(bgcolor='white', font_size=12, font_family="Arial"),
        hovermode="closest"
    )

    # Convert the Plotly figure to HTML and store it in a variable
    fig_html = fig.to_html()

    # Custom HTML and CSS to create a container with rounded corners
    rounded_corner_container = f"""
    <div style='border-radius: 15px; overflow: hidden;'>
        {fig_html}
    </div>
    """

    # Use Streamlit's HTML component to render the custom container
    components.html(rounded_corner_container, height=600,width=710)
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
        # Define the outlier_treatment function
    def outlier_treatment(datacolumn):
        sorted(datacolumn)
        Q1, Q3 = np.percentile(datacolumn, [25, 75])
        IQR = Q3 - Q1
        lower_range = Q1 - (1.5 * IQR)
        upper_range = Q3 + (1.5 * IQR)
        return lower_range, upper_range

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

    # Plotting function with Plotly for curve fitting models
    def plot_model(T, Q, Q_exp, Q_harm, Q_hyp, model):
        fig = px.line()
        fig.add_scatter(x=T, y=Q, mode='lines', name='Smoothed', line=dict(color='green'))
        if model == 'exponential' or model == 'all':
            fig.add_scatter(x=T, y=Q_exp, mode='lines', name='Exponential', line=dict(color='blue'))
        if model == 'harmonic' or model == 'all':
            fig.add_scatter(x=T, y=Q_harm, mode='lines', name='Harmonic', line=dict(color='red'))
        if model == 'hyperbolic' or model == 'all':
            fig.add_scatter(x=T, y=Q_hyp, mode='lines', name='Hyperbolic', line=dict(color='yellow'))
        fig.update_layout(title="Curve Fitting Models for Oil Production of Well 15/9-F-14",
                        xaxis_title='Days',
                        yaxis_title='Oil Production (Smoothed)',
                        legend_title='Model')
                # Apply custom styles for a glassy appearance and rounded edges
        fig.update_layout(
            plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Glassy white background
            paper_bgcolor='rgba(0, 71, 171, 0.5)',  # Semi-transparent blue background
            margin=dict(l=50, r=20, t=50, b=40),  # Adjust margins to fit the chart within the frame
            font=dict(family="Arial", size=12, color="white"),
            xaxis=dict(linecolor='black', linewidth=2, mirror=True),
            yaxis=dict(linecolor='black', linewidth=2, mirror=True),
            legend=dict(title='Model', bgcolor='rgba(255, 255, 255, 0.4)', bordercolor='black', borderwidth=2),
            title=dict(font=dict(size=16, color='white')),
            hoverlabel=dict(bgcolor='white', font_size=12, font_family="Arial"),
            hovermode="closest"
        )

        # Custom HTML and CSS to create a container with rounded corners
        rounded_corner_container = f"""
        <div style='border-radius: 15px; overflow: hidden;'>
            {fig.to_html()}
        </div>
        """

        # Use Streamlit's HTML component to render the custom container
        components.html(rounded_corner_container, height=600,width=710)
        

    # Streamlit page configuration

    if selected_page == "part 3":
        st.subheader("Oil Production of 15/9-F-14")
        
        # Filter the DataFrame for the specific well and remove zero oil production entries
        df_part3 = df[df["NPD_WELL_BORE_NAME"] == '15/9-F-14']
        df_part3 = df_part3[df_part3["BORE_OIL_VOL"] != 0]
        
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

        # Create scatter plot for oil production with Plotly
        fig = px.scatter(df_part3_filtered, x='DATEPRD', y='oil_smoothed',
                        labels={'DATEPRD': 'Date', 'oil_smoothed': 'Smoothed Oil Production'},
                        title="Smoothed Oil Production of 15/9-F-14 within Selected Date Range")
        # Update traces to customize the marker color and appearance
        fig.update_traces(
            marker=dict(color='blue'),  # Customize marker properties
            selector=dict(mode='markers')  # Apply updates to the scatter markers
        )
        # Apply custom styles for a glassy appearance and rounded edges
        fig.update_layout(
            plot_bgcolor='rgba(255, 255, 255, .9)',  # Glassy white background
            paper_bgcolor='rgba(0, 71, 171, .5)',  # Semi-transparent white background
            margin=dict(l=50, r=20, t=50, b=40),  # Adjust margins to fit the chart within the frame
            font=dict(family="Arial", size=12, color="white"),
            xaxis=dict(linecolor='black', linewidth=2, mirror=True),
            yaxis=dict(linecolor='black', linewidth=2, mirror=True),
            legend=dict(title='Model', bgcolor='rgba(255, 255, 255, 0.4)', bordercolor='black', borderwidth=2),
            title=dict(font=dict(size=16, color='white')),
            hoverlabel=dict(bgcolor='white', font_size=12, font_family="Arial"),
            hovermode="closest"
        )
                # Custom HTML and CSS to create a container with rounded corners
        rounded_corner_container = f"""
        <div style='border-radius: 15px; overflow: hidden;'>
            {fig.to_html()}
        </div>
        """
        # Use Streamlit's HTML component to render the custom container
        components.html(rounded_corner_container, height=600,width=710)
        # Show Effect of Smoothing
        st.subheader("Show Effect of Smoothing")

                # Plotting the data with Plotly for smoothing effect
        fig = px.line(df_part3_filtered, x="DATEPRD", y=["BORE_OIL_VOL", "oil_smoothed"],
                    labels={'DATEPRD': 'Date', 'BORE_OIL_VOL': 'Original Oil Production', 'oil_smoothed': 'Smoothed Oil Production'},
                    title="Oil Production and Smoothing Effect of 15/9-F-14")
                # Update traces to customize the line colors and appearance
        fig.update_traces(
            mode='lines+markers',
            line=dict(color='blue', width=2),  # Set the color for the original oil production line
            selector=dict(name='BORE_OIL_VOL')  # Select the trace by name
        )
        fig.update_traces(
                    mode='lines+markers',
                    line=dict(color='green', width=2),  # Set the color for the original oil production line
                    selector=dict(name='oil_smoothed')  # Select the trace by name
                )
        



        # Apply custom styles for a glassy appearance and rounded edges
        fig.update_layout(
            plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Glassy white background
            paper_bgcolor='rgba(0, 71, 171, 0.5)',  # Semi-transparent blue background
            margin=dict(l=50, r=20, t=50, b=40),  # Adjust margins to fit the chart within the frame
            font=dict(family="Arial", size=12, color="white"),
            xaxis=dict(linecolor='black', linewidth=2, mirror=True),
            yaxis=dict(linecolor='black', linewidth=2, mirror=True),
            legend=dict(title='Model', bgcolor='rgba(255, 255, 255, 0.4)', bordercolor='black', borderwidth=2),
            title=dict(font=dict(size=16, color='white')),
            hoverlabel=dict(bgcolor='white', font_size=12, font_family="Arial"),
            hovermode="closest"
        )

        # Custom HTML and CSS to create a container with rounded corners
        rounded_corner_container = f"""
        <div style='border-radius: 15px; overflow: hidden;'>
            {fig.to_html()}
        </div>
        """

        # Use Streamlit's HTML component to render the custom container
        components.html(rounded_corner_container, height=600,width=710)

        # Curve-Fitting
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

        # Fit models
        qi_exp, di_exp, _ = fit_model(exponential, T, Q)
        qi_harm, di_harm, _ = fit_model(harmonic, T, Q)
        qi_hyp, di_hyp, b_hyp = fit_model(hyperbolic, T, Q, p0=[1, 1, 1])

        # Generate model curves
        Q_exp = exponential(T, qi_exp, di_exp)
        Q_harm = harmonic(T, qi_harm, di_harm)
        Q_hyp = hyperbolic(T, qi_hyp, di_hyp, b_hyp)

        # Dropdown for model selection in Streamlit
        model_selected = st.selectbox('Select a model:', ['exponential', 'harmonic', 'hyperbolic', 'all'])

        # Display the interactive plot based on the selected model
        plot_model(T, Q, Q_exp, Q_harm, Q_hyp, model_selected)

# Check the selected page from the radio buttons
elif selected_page == "About the Project":
    # Use markdown with unsafe_allow_html to allow HTML and CSS
    st.markdown("""
        <style>
        .main * {
            color: white;
        }
        </style>
        <h2>About This Project</h2>
        <p>
        This project is showcased at the SPE Science Fair and focuses on the analysis of production data from the Volve field. 
        The goal is to provide interactive visualizations and insights that can help understand the field's performance.
        </p>
        <h2>Data Overview</h2>
        <p>
        The dataset contains daily and monthly production data for seven (07) wellbores from the Volve field in Norway. 
        Available information includes the wellbore name, the operating time, the hydrocarbon (oil & gas) production, and the fluid injection.
        </p>
        <p>
        This comprehensive dataset is a valuable resource for analysis and can be found on Kaggle.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("https://www.kaggle.com/datasets/lamyalbert/volve-production-data")

    # Acknowledgment for Eng. Ahmed Abd Elgawad with LinkedIn and YouTube channel information
        
    st.markdown("""
        <h2 style='color: white; text-align: center;'>Acknowledgments</h2>
        <p style='color: white; text-align: center;'>
        I would like to express my deepest appreciation to Eng. Ahmed Abd Elgawad for his invaluable guidance and support throughout this project.
        </p>
        <p style='color: white; text-align: center;'>
        For more professional insights, you can connect with Eng. Abd Elgawad on 
        <a href='https://www.linkedin.com/in/ahmed-abd-elgawad-9a6889212/' style='color: white;'>LinkedIn</a> 
        and explore his analytical work on his 
        <a href='https://www.youtube.com/@AhmedAbdElgawad-petroAnalyst' style='color: white;'>YouTube channel</a>.
        </p>
        <div style='text-align: center;'>
            <img src='https://raw.githubusercontent.com/Youssef1Rezk/Production-Data-Analysis/c66eff7657ed8cd1d513321001c378cf809769aa/photo_2024-05-10_08-56-10.jpg
' 
            style='border-radius: 50%; width: 300px; height: 300px;'>
        </div>
        """, unsafe_allow_html=True)


 
