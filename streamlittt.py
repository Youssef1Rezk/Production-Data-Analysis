# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from welly import Project
import folium
from IPython.core.display import HTML
import time
import welly.quality as wq
from welly import Well
from IPython.display import HTML
from welly import Location
import pkg_resources

# Main function to run the app
# Main function to run the app
def main():
    st.title("Horizontal Drilling Trajectory Visualization")

    # Sidebar for category selection
    category = st.sidebar.radio("Select Analysis Category", ("Single Well Location", "Multi Well Analysis"))

    # Set local file paths for LAS and CSV files for single well analysis
    las_file_url = 'https://raw.githubusercontent.com/Youssef1Rezk/well-trajectory/main/L05-15-Spliced.las'
    csv_file_url = 'https://raw.githubusercontent.com/Youssef1Rezk/well-trajectory/main/L05-15-Survey.csv'

    # Create an empty placeholder for category content
    content_placeholder = st.empty()

    if category == "Single Well Location":
        with content_placeholder.container():
            st.header("Single Well Location Analysis")
            time.sleep(0.5)  # Wait for half a second to simulate animation

            try:
                # Load LAS data using Welly
                well = Well.from_las(las_file_url)
                
                # Button to display LAS file data
                if st.button("Show LAS File Data"):
                    st.write("Loaded LAS file successfully.")
                    st.pyplot(well.plot(extents='curves'))

                # Load survey data from CSV file
                survey = pd.read_csv(csv_file_url)

                # Button to display survey data
                if st.button("Show Survey Data"):
                    st.write("Survey data:")
                    st.write(survey.head())

                # Extract and display relevant columns
                survey_subset = survey[['MD', 'INC', 'AZI']]
                well.location.add_deviation(survey_subset.values)

                # Retrieve position data
                x_loc = well.location.position[:, 0]
                y_loc = well.location.position[:, 1]
                z_loc = well.location.position[:, 2]

                # Button to plot 2D trajectory views
                if st.button("Show 2D Trajectory Plots"):
                    st.subheader("2D Trajectory Plots")
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                    ax1.plot(x_loc, y_loc, lw=3)
                    ax1.set_title("X Location vs Y Location")

                    ax2.plot(x_loc, z_loc, lw=3)
                    ax2.set_title("X Location vs TVD")
                    ax2.invert_yaxis()

                    ax3.plot(y_loc, z_loc, lw=3)
                    ax3.set_title("Y Location vs TVD")
                    ax3.invert_yaxis()

                    st.pyplot(fig)

                # Button to plot 2D trajectory with start/end markers
                if st.button("Show 2D Trajectory with Start/End Markers"):
                    st.subheader("2D Trajectory with Start/End Markers")
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                    ax1.plot(x_loc, y_loc, lw=3)
                    ax1.plot(x_loc[0], y_loc[0], marker='s', color='black', ms=8, label="Start")
                    ax1.plot(x_loc[-1], y_loc[-1], marker='*', color='red', ms=8, label="End")
                    ax1.set_title("X Location vs Y Location")
                    ax1.legend()

                    ax2.plot(x_loc, z_loc, lw=3)
                    ax2.plot(x_loc[0], z_loc[0], marker='s', color='black', ms=8)
                    ax2.plot(x_loc[-1], z_loc[-1], marker='*', color='red', ms=8)
                    ax2.set_title("X Location vs TVD")
                    ax2.invert_yaxis()

                    ax3.plot(y_loc, z_loc, lw=3)
                    ax3.plot(y_loc[0], z_loc[0], marker='s', color='black', ms=8)
                    ax3.plot(y_loc[-1], z_loc[-1], marker='*', color='red', ms=8)
                    ax3.set_title("Y Location vs TVD")
                    ax3.invert_yaxis()

                    st.pyplot(fig)

                # Button to compare calculated trajectory against original survey data
                if st.button("Compare Against Original Survey"):
                    st.subheader("Comparison Against Original Survey Data")
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                    ax1.plot(x_loc, y_loc, lw=7, label="Calculated")
                    ax1.plot(x_loc[0], y_loc[0], marker='s', color='black', ms=8)
                    ax1.plot(survey['X-offset'], survey['Y-offset'], label="Original Survey")
                    ax1.plot(x_loc[-1], y_loc[-1], marker='*', color='red', ms=8)
                    ax1.set_title("X Location vs Y Location")
                    ax1.legend()

                    ax2.plot(x_loc, z_loc, lw=7, label="Calculated")
                    ax2.plot(x_loc[0], z_loc[0], marker='s', color='black', ms=8)
                    ax2.plot(survey['X-offset'], survey['TVD'], label="Original Survey")
                    ax2.plot(x_loc[-1], z_loc[-1], marker='*', color='red', ms=8)
                    ax2.invert_yaxis()
                    ax2.set_title("X Location vs TVD")
                    ax2.legend()

                    ax3.plot(y_loc, z_loc, lw=7, label="Calculated")
                    ax3.plot(y_loc[0], z_loc[0], marker='s', color='black', ms=8)
                    ax3.plot(survey['Y-offset'], survey['TVD'], label="Original Survey")
                    ax3.plot(y_loc[-1], z_loc[-1], marker='*', color='red', ms=8)
                    ax3.invert_yaxis()
                    ax3.set_title("Y Location vs TVD")
                    ax3.legend()

                    st.pyplot(fig)

                # Button to show 3D plot for the trajectory
                if st.button("Show 3D Trajectory Plot"):
                    st.subheader("3D Trajectory Plot")
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot3D(x_loc, y_loc, z_loc, lw=2, color="blue")
                    ax.set_xlabel("X Location")
                    ax.set_ylabel("Y Location")
                    ax.set_zlabel("TVD")
                    ax.set_zlim(3000, 0)
                    st.pyplot(fig)

            except FileNotFoundError:
                st.error("One or both of the required files were not found. Please ensure the paths are correct.")

    elif category == "Multi Well Analysis":
        with content_placeholder.container():
            st.header("Multi Well Analysis")
            time.sleep(0.5)  # Wait for half a second to simulate animation
            
            # Load wells using Project
            wells = Project.from_las('all wells/*.las')
            st.write(f"Loaded {len(wells)} wells.")

            well_dict = {}
            for well in wells:
                if isinstance(well, Well):  # Check if it's a Well instance
                    well_dict[well.uwi] = {
                        'well name': well.name, 
                        'Latitude': well.location.latitude,  # Accessing latitude directly
                        'Longitude': well.location.longitude  # Accessing longitude directly
                    }

            wells_df = pd.DataFrame.from_dict(well_dict, orient='index')
            wells_df.reset_index(inplace=True)
            wells_df.rename(columns={'index': 'UWI'}, inplace=True)

            # Plot all well locations on a single map
            mean_lat = wells_df['Latitude'].mean()
            mean_long = wells_df['Longitude'].mean()
            m = folium.Map(location=[mean_lat, mean_long], zoom_start=7)

            for index, well_location in wells_df.iterrows():
                folium.Marker(
                    [well_location['Latitude'], well_location['Longitude']],
                    popup=well_location['well name']
                ).add_to(m)

            st.subheader("Well Locations Map")
            st.components.v1.html(m._repr_html_(), height=500, scrolling=True)

            # Plot GR from all wells
            fig, axs = plt.subplots(figsize=(14, 10), ncols=len(wells))
            for i, (ax, well) in enumerate(zip(axs, wells)):
                gr = well.get_curve('GR')
                if gr is not None:
                    ax = gr.plot(ax=ax, c='green')
                ax.set_title(f"GR for\n{well.name}")

            st.pyplot(fig)

            # Plot RHOB from all wells
            fig, axs = plt.subplots(figsize=(14, 10), ncols=len(wells))
            curve_name = 'RHOB'
            for i, (ax, well) in enumerate(zip(axs, wells)):
                rhob = well.get_curve(curve_name)
                if rhob is not None:
                    ax = rhob.plot(ax=ax, c='blue')
                ax.set_title(f"RHOB for\n{well.name}")

            st.pyplot(fig)

            # Plot NPHI from all wells
            fig, axs = plt.subplots(figsize=(14, 10), ncols=len(wells))
            curve_name = 'NPHI'
            for i, (ax, well) in enumerate(zip(axs, wells)):
                nphi = well.get_curve(curve_name)
                if nphi is not None:
                    ax = nphi.plot(ax=ax, c='orange')
                ax.set_title(f"NPHI for\n{well.name}")

            st.pyplot(fig)

            # Quality control tests
            tests = {
                'Each': [wq.no_flat, wq.no_gaps, wq.not_empty],
                'GR': [
                    wq.all_positive,
                    wq.all_between(0, 250),
                    wq.check_units(['API', 'GAPI']),
                ],'RHOB': [
                    wq.all_positive,
                    wq.all_between(1.5, 3),
                    wq.check_units(['G/CC', 'g/cm3']),
                ]
            }

            data_qc_table = wells.curve_table_html(keys=['GR', 'RHOB'], tests=tests)
            st.subheader("Data Quality Control")
            st.write(HTML(data_qc_table), unsafe_allow_html=True)

            qc_dict = {}
            for well in wells:
                qc_dict[well.name] = well.qc_table_html(tests)

            st.write("Quality Control for specific wells:")
            for well_name, qc_html in qc_dict.items():
                st.write(f"### {well_name}")
                st.write(HTML(qc_html), unsafe_allow_html=True)

            

if __name__ == "__main__":
    main()
