"""
ISS Pass Predictor & Logger - Streamlit Web App
A simple web interface for predicting ISS passes and logging observations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time # Added for retry mechanism
from requests.exceptions import RequestException # Specific exception for robustness
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Skyfield imports
from skyfield.api import load, EarthSatellite, Topos

# Set page config
st.set_page_config(
    page_title="ISS Pass Predictor",
    page_icon="🛰️",
    layout="wide"
)

# Title
st.title("🛰️ ISS Pass Predictor & Logger")
st.markdown("Predict when the International Space Station will be visible from your location!")

# --- Initial Setup and Caching ---

# FIX: Initialize non-serializable Skyfield objects directly in the script's global scope.
try:
    # Load Skyfield timescale
    ts = load.timescale()
    # Load ephemeris (downloads de421.bsp if not present)
    eph = load('de421.bsp')
except Exception as e:
    st.error(f"Failed to load Skyfield data: {e}. Please check your Skyfield installation.")
    ts = None
    eph = None


# Initialize session state for persistent data
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None
if 'observations_df' not in st.session_state:
    # Initialize with the correct columns, ready for concatenation
    st.session_state.observations_df = pd.DataFrame(columns=[
        'pass_id', 'observation_time', 'weather', 'successful', 'notes', 'actual_altitude'
    ])

@st.cache_data(ttl=3600)  # Cache TLE data for 1 hour
def download_tle_data(max_retries=5, initial_delay=3, timeout=20):
    """
    Download current ISS TLE data from Celestrak with a retry mechanism.
    Handles temporary network issues.
    
    INCREASED: max_retries (to 5), initial_delay (to 3s), and timeout (to 20s) 
    for higher resilience against slow servers.
    """
    tle_url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544"
    last_exception = None

    for attempt in range(max_retries):
        try:
            # Increased timeout for better connection resilience
            response = requests.get(tle_url, timeout=timeout)
            response.raise_for_status()
            
            tle_lines = response.text.strip().split('\n')
            
            # Find ISS data
            for i, line in enumerate(tle_lines):
                if 'ISS' in line or '25544' in line:
                    if i + 2 < len(tle_lines):
                        return tle_lines[i:i+3]
            
            # If we reach here, data was downloaded but not found
            raise ValueError("ISS data not found or file format changed.")

        except RequestException as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                st.warning(f"TLE download failed (Attempt {attempt + 1}/{max_retries}). Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                # Last attempt failed
                st.error(f"Error downloading TLE data after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            # Handle non-request exceptions (like ValueError from data parsing)
            st.error(f"Error processing TLE data: {e}")
            return None

    # If the loop finishes without returning successfully, return None
    return None

def calculate_visible_passes(satellite, observer_location, start_time, days=7, min_altitude=10.0):
    """Calculate all visible ISS passes for a specified time period, including rise/set azimuth."""
    # Ensure ts is available before proceeding
    if ts is None:
        return []
        
    end_time = start_time + days
    t, events = satellite.find_events(observer_location, start_time, end_time, altitude_degrees=min_altitude)
    
    passes = []
    i = 0
    while i < len(events):
        # A pass is defined by: 0=Rise, 1=Max Alt, 2=Set
        # Check if the sequence is (Rise, Max, Set)
        if i + 2 < len(events) and events[i] == 0 and events[i+1] == 1 and events[i+2] == 2:
            rise_time = t[i]
            max_alt_time = t[i+1]
            set_time = t[i+2]
            
            duration_minutes = (set_time - rise_time) * 24 * 60
            
            difference = satellite - observer_location
            
            # Max Altitude and Brightness
            topocentric_max = difference.at(max_alt_time)
            alt_max, az_max, distance = topocentric_max.altaz()
            max_altitude = alt_max.degrees
            
            # Rise Azimuth (Horizon to Max Alt)
            topocentric_rise = difference.at(rise_time)
            _, rise_az, _ = topocentric_rise.altaz()
            
            # Set Azimuth (Max Alt to Horizon)
            topocentric_set = difference.at(set_time)
            _, set_az, _ = topocentric_set.altaz()
            
            distance_km = distance.km
            # Simple approximation of visual brightness (magnitude). Lower is brighter.
            # Using 400km as a reference for roughly 0 magnitude
            brightness = -2.0 - (400 / distance_km) 
            
            pass_info = {
                'rise_time': rise_time.utc_datetime().replace(tzinfo=None), # Make timezone naive
                'max_alt_time': max_alt_time.utc_datetime().replace(tzinfo=None), # Make timezone naive
                'set_time': set_time.utc_datetime().replace(tzinfo=None), # Make timezone naive
                'max_altitude': round(max_altitude, 2),
                'rise_azimuth': round(rise_az.degrees, 1),
                'set_azimuth': round(set_az.degrees, 1),
                'duration_minutes': round(duration_minutes, 1),
                'brightness': round(brightness, 2)
            }
            
            passes.append(pass_info)
            i += 3 # Skip past Rise, Max, Set
        else:
            i += 1 # Move to the next event
    
    return passes

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Location")
    latitude = st.number_input("Latitude (°N)", value=39.9612, min_value=-90.0, max_value=90.0, step=0.0001)
    longitude = st.number_input("Longitude (°W)", value=-82.9988, min_value=-180.0, max_value=180.0, step=0.0001)
    elevation = st.number_input("Elevation (m)", value=275, min_value=0, max_value=10000)
    
    st.subheader("Prediction Settings")
    days_ahead = st.slider("Days to predict", 1, 14, 7)
    min_altitude = st.slider("Minimum altitude (°)", 0.0, 30.0, 10.0, step=0.5)
    altitude_threshold = st.slider("Filter passes above (°)", 0.0, 90.0, 30.0, step=5.0)
    
    if st.button("🔄 Calculate Passes", type="primary"):
        if ts is None or eph is None:
            st.error("Cannot calculate passes because Skyfield data failed to load.")
        else:
            with st.spinner("Downloading TLE data and calculating passes..."):
                
                # Download TLE
                tle_data = download_tle_data()
                if tle_data:
                    # EarthSatellite initialization
                    satellite = EarthSatellite(tle_data[1], tle_data[2], tle_data[0], ts)
                    observer = Topos(latitude, longitude, elevation_m=elevation)
                    
                    # Calculate passes
                    now = ts.now()
                    all_passes = calculate_visible_passes(satellite, observer, now, days=days_ahead, min_altitude=min_altitude)
                    
                    if all_passes:
                        predictions_df = pd.DataFrame(all_passes)
                        predictions_df['pass_id'] = range(1, len(predictions_df) + 1)
                        # The calculated times in the function are already timezone-naive, 
                        # but we ensure dtypes are consistent across Streamlit runs.
                        predictions_df['rise_time'] = pd.to_datetime(predictions_df['rise_time'])
                        predictions_df['max_alt_time'] = pd.to_datetime(predictions_df['max_alt_time'])
                        predictions_df['set_time'] = pd.to_datetime(predictions_df['set_time'])
                        
                        st.session_state.predictions_df = predictions_df
                        st.success(f"✅ Found {len(predictions_df)} passes!")
                    else:
                        st.warning("No passes found for the specified criteria.")

# --- Main Content ---
if st.session_state.predictions_df is not None:
    predictions_df = st.session_state.predictions_df
    
    # Location Map Enhancement
    st.subheader("🌍 Observer Location")
    location_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
    st.map(location_data, zoom=9)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "⭐ Best Passes", "📝 Log Observation", "📈 Analytics"])
    
    # --- Tab 1: Predictions ---
    with tab1:
        st.header("All Predictions")
        
        display_cols = [
            'pass_id', 'rise_time', 'max_alt_time', 'max_altitude', 
            'rise_azimuth', 'set_azimuth', 'duration_minutes', 'brightness'
        ]
        
        st.dataframe(
            predictions_df[display_cols].style.format({
                'max_altitude': '{:.1f}°',
                'rise_azimuth': '{:.1f}°',
                'set_azimuth': '{:.1f}°',
                'duration_minutes': '{:.1f} min',
                'brightness': '{:.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Daily pass frequency
        st.subheader("Passes per Day")
        daily_passes = predictions_df['rise_time'].dt.date.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        daily_passes.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title('ISS Pass Frequency')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Passes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    # --- Tab 2: Best Passes ---
    with tab2:
        st.header(f"Best Passes (Above {altitude_threshold}°)")
        good_passes = predictions_df[predictions_df['max_altitude'] >= altitude_threshold]
        
        if len(good_passes) > 0:
            st.metric("Good Passes Found", len(good_passes))
            st.dataframe(
                good_passes[display_cols].style.format({
                    'max_altitude': '{:.1f}°',
                    'rise_azimuth': '{:.1f}°',
                    'set_azimuth': '{:.1f}°',
                    'duration_minutes': '{:.1f} min',
                    'brightness': '{:.2f}'
                }),
                use_container_width=True
            )
        else:
            st.info(f"No passes found above {altitude_threshold}°")
    
    # --- Tab 3: Log Observation ---
    with tab3:
        st.header("Log an Observation")
        
        if len(predictions_df) > 0:
            
            # Create a user-friendly format for the select box
            pass_options = predictions_df.apply(
                lambda row: f"Pass {row['pass_id']} - {row['rise_time'].strftime('%Y-%m-%d %H:%M')} (Max Alt: {row['max_altitude']:.1f}°)",
                axis=1
            ).tolist()
            
            selected_option = st.selectbox(
                "Select Pass to Log",
                options=pass_options
            )
            
            # Extract the pass_id from the selected string
            selected_pass_id = int(selected_option.split(' - ')[0].replace('Pass ', ''))
            
            col1, col2 = st.columns(2)
            with col1:
                weather = st.selectbox("Weather", ["Clear", "Partly Cloudy", "Cloudy", "Overcast", "Rainy"])
                successful = st.checkbox("Successfully Observed", value=True)
            with col2:
                # Use key to reset the input when a different pass is selected
                actual_altitude = st.number_input("Actual Altitude (°)", min_value=0.0, max_value=90.0, value=None, step=0.1, key=f"altitude_input_{selected_pass_id}", help="Estimate the highest point the ISS reached.")
                notes = st.text_area("Notes", height=100, key=f"notes_input_{selected_pass_id}")
            
            if st.button("💾 Save Observation"):
                observation = {
                    'pass_id': selected_pass_id, # Use the extracted ID
                    'observation_time': datetime.now(),
                    'weather': weather,
                    'successful': successful,
                    'notes': notes if notes else "",
                    'actual_altitude': actual_altitude if actual_altitude else np.nan
                }
                
                # Use pd.concat to append the new observation
                new_obs_df = pd.DataFrame([observation])
                st.session_state.observations_df = pd.concat([st.session_state.observations_df, new_obs_df], ignore_index=True)
                    
                st.success("✅ Observation saved!")
                # Force a re-run to clear the form visually
                st.rerun() 
        
        # Show existing observations
        if not st.session_state.observations_df.empty:
            st.subheader("Your Observations")
            # Display only relevant columns, excluding potentially empty ones from initialization
            display_obs_cols = [c for c in ['pass_id', 'observation_time', 'weather', 'successful', 'actual_altitude', 'notes'] if c in st.session_state.observations_df.columns]
            st.dataframe(st.session_state.observations_df[display_obs_cols], use_container_width=True)
    
    # --- Tab 4: Analytics ---
    with tab4:
        st.header("Analytics")
        
        if not st.session_state.observations_df.empty:
            
            # Data cleaning and merge
            clean_observations_df = st.session_state.observations_df.copy()
            
            # Ensure 'pass_id' is integer for reliable merge
            clean_observations_df['pass_id'] = pd.to_numeric(clean_observations_df['pass_id'], errors='coerce').astype('Int64')
            predictions_df['pass_id'] = pd.to_numeric(predictions_df['pass_id'], errors='coerce').astype('Int64')
            
            # Merge and filter only observed rows
            # We use a left merge to combine predictions with corresponding observations
            merged_df = pd.merge(predictions_df, clean_observations_df, on='pass_id', how='left', suffixes=('_pred', '_obs'))
            observed_rows = merged_df[merged_df['successful'].notna()]
            
            total_observed = len(observed_rows)
            
            if total_observed > 0:
                success_count = observed_rows['successful'].sum()
                success_rate = success_count / total_observed
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Success Rate", f"{success_rate:.1%}")
                    st.metric("Total Observations", total_observed)
                
                with col2:
                    # Pie chart
                    fig, ax = plt.subplots(figsize=(4, 4))
                    labels = ['Successful', 'Failed']
                    sizes = [success_count, total_observed - success_count]
                    colors = ['#4CAF50', '#FF5722'] 
                    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
                    ax.set_title('Observation Success Rate')
                    st.pyplot(fig)
                
                # Weather analysis
                if 'weather' in observed_rows.columns:
                    st.subheader("Success by Weather")
                    weather_analysis = observed_rows.groupby('weather')['successful'].agg(['mean', 'count'])
                    weather_analysis.columns = ['Success Rate', 'Count']
                    weather_analysis['Success Rate'] = (weather_analysis['Success Rate'] * 100).map('{:.1f}%'.format)
                    st.dataframe(weather_analysis)
            else:
                st.info("No completed observations yet. Log some to see analytics!")
        else:
            st.info("No observations yet. Log some to see analytics!")
        
        # Export data
        st.subheader("Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Ensure predictions_df is clean before export
            predictions_df['rise_time'] = predictions_df['rise_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            predictions_df['max_alt_time'] = predictions_df['max_alt_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            predictions_df['set_time'] = predictions_df['set_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            csv_predictions = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions",
                data=csv_predictions,
                file_name="iss_predictions.csv",
                mime="text/csv"
            )
        
        with col2:
            if not st.session_state.observations_df.empty:
                # Format datetime for CSV export
                obs_df_export = st.session_state.observations_df.copy()
                # Check if 'observation_time' is a datetime type before applying dt.strftime
                if pd.api.types.is_datetime64_any_dtype(obs_df_export['observation_time']):
                     obs_df_export['observation_time'] = obs_df_export['observation_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                csv_observations = obs_df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Observations",
                    data=csv_observations,
                    file_name="iss_observations.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Re-calculate merged_df if needed for export robustness
            if not st.session_state.observations_df.empty:
                merged_df_export = pd.merge(predictions_df, clean_observations_df, on='pass_id', how='left', suffixes=('_pred', '_obs'))
                
                # Format dates to strings before export
                date_cols = ['rise_time_pred', 'max_alt_time_pred', 'set_time_pred', 'observation_time']
                for col in date_cols:
                    if col in merged_df_export.columns:
                        # Attempt to convert datetime columns to string; ignore if already string
                        try:
                            merged_df_export[col] = merged_df_export[col].astype(str)
                        except:
                            pass # Column is likely already str or non-datetime
                
                csv_merged = merged_df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Data",
                    data=csv_merged,
                    file_name="iss_full_data.csv",
                    mime="text/csv"
                )

else:
    st.info("<-- Configure your location and click 'Calculate Passes' to get started!")
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Set your location** in the sidebar (latitude, longitude, elevation)
        2. **Adjust prediction settings** (days ahead, minimum altitude)
        3. **Click "Calculate Passes"** to get ISS predictions
        4. **View predictions** in the Predictions tab (note the **Rise/Set Azimuths** to know where to look!)
        5. **Check "Best Passes"** for high-altitude passes (easiest to see)
        6. **Log observations** after watching a pass
        7. **View analytics** to see your success rate
        """)
    
    st.markdown("""
    ### About
    This app predicts when the International Space Station will be visible from your location.
    The ISS orbits Earth every ~90 minutes, but you can only see it during **twilight** when it's
    illuminated by the sun while your location is in darkness.
    
    **Altitude & Direction Guide:**
    - **0° Altitude** = Horizon
    - **30° Altitude** = Good viewing (above most buildings)
    - **90° Altitude** = Directly overhead (the best view!)
    - **0° Azimuth** = North
    - **90° Azimuth** = East
    - **180° Azimuth** = South
    - **270° Azimuth** = West
    """)