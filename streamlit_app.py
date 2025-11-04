"""
ISS Pass Predictor & Logger - Streamlit Web App
A simple web interface for predicting ISS passes and logging observations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
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

# Initialize session state
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None
if 'observations_df' not in st.session_state:
    st.session_state.observations_df = pd.DataFrame(columns=[
        'pass_id', 'observation_time', 'weather', 'successful', 'notes', 'actual_altitude'
    ])

@st.cache_data
def load_skyfield_data():
    """Load Skyfield timescale and ephemeris (cached for performance)"""
    ts = load.timescale()
    # This will download de421.bsp automatically if not present
    eph = load('de421.bsp')
    return ts, eph

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_tle_data():
    """Download current ISS TLE data from Celestrak"""
    tle_url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544"
    try:
        response = requests.get(tle_url)
        response.raise_for_status()
        tle_lines = response.text.strip().split('\n')
        
        # Find ISS data
        for i, line in enumerate(tle_lines):
            if 'ISS' in line or '25544' in line:
                if i + 2 < len(tle_lines):
                    return tle_lines[i:i+3]
        raise ValueError("ISS data not found")
    except Exception as e:
        st.error(f"Error downloading TLE data: {e}")
        return None

def calculate_visible_passes(satellite, observer_location, start_time, days=7, min_altitude=10.0):
    """Calculate all visible ISS passes for a specified time period."""
    end_time = start_time + days
    t, events = satellite.find_events(observer_location, start_time, end_time, altitude_degrees=min_altitude)
    
    passes = []
    i = 0
    while i < len(events):
        if i + 2 < len(events) and events[i] == 0 and events[i+1] == 1 and events[i+2] == 2:
            rise_time = t[i]
            max_alt_time = t[i+1]
            set_time = t[i+2]
            
            duration_minutes = (set_time - rise_time) * 24 * 60
            
            difference = satellite - observer_location
            topocentric = difference.at(max_alt_time)
            alt, az, distance = topocentric.altaz()
            max_altitude = alt.degrees
            
            distance_km = distance.km
            brightness = 0.0 - (400 / distance_km)
            
            pass_info = {
                'rise_time': rise_time.utc_datetime(),
                'max_alt_time': max_alt_time.utc_datetime(),
                'set_time': set_time.utc_datetime(),
                'max_altitude': round(max_altitude, 2),
                'duration_minutes': round(duration_minutes, 1),
                'brightness': round(brightness, 2)
            }
            
            passes.append(pass_info)
            i += 3
        else:
            i += 1
    
    return passes

# Sidebar for configuration
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
        with st.spinner("Downloading TLE data and calculating passes..."):
            # Load Skyfield
            ts, eph = load_skyfield_data()
            
            # Download TLE
            tle_data = download_tle_data()
            if tle_data:
                satellite = EarthSatellite(tle_data[1], tle_data[2], tle_data[0], ts)
                observer = Topos(latitude, longitude, elevation_m=elevation)
                
                # Calculate passes
                now = ts.now()
                all_passes = calculate_visible_passes(satellite, observer, now, days=days_ahead, min_altitude=min_altitude)
                
                if all_passes:
                    predictions_df = pd.DataFrame(all_passes)
                    predictions_df['pass_id'] = range(1, len(predictions_df) + 1)
                    st.session_state.predictions_df = predictions_df
                    st.success(f"✅ Found {len(predictions_df)} passes!")
                else:
                    st.warning("No passes found for the specified criteria.")

# Main content
if st.session_state.predictions_df is not None:
    predictions_df = st.session_state.predictions_df
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "⭐ Best Passes", "📝 Log Observation", "📈 Analytics"])
    
    with tab1:
        st.header("All Predictions")
        st.dataframe(
            predictions_df[['pass_id', 'rise_time', 'max_altitude', 'duration_minutes', 'brightness']].style.format({
                'max_altitude': '{:.1f}°',
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
    
    with tab2:
        st.header(f"Best Passes (Above {altitude_threshold}°)")
        good_passes = predictions_df[predictions_df['max_altitude'] >= altitude_threshold]
        
        if len(good_passes) > 0:
            st.metric("Good Passes", len(good_passes))
            st.dataframe(
                good_passes[['pass_id', 'rise_time', 'max_altitude', 'duration_minutes', 'brightness']].style.format({
                    'max_altitude': '{:.1f}°',
                    'duration_minutes': '{:.1f} min',
                    'brightness': '{:.2f}'
                }),
                use_container_width=True
            )
        else:
            st.info(f"No passes found above {altitude_threshold}°")
    
    with tab3:
        st.header("Log an Observation")
        
        if len(predictions_df) > 0:
            pass_id = st.selectbox(
                "Select Pass ID",
                options=sorted(predictions_df['pass_id'].tolist()),
                format_func=lambda x: f"Pass {x} - {predictions_df[predictions_df['pass_id']==x]['rise_time'].iloc[0].strftime('%Y-%m-%d %H:%M')}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                weather = st.selectbox("Weather", ["Clear", "Partly Cloudy", "Cloudy", "Overcast", "Rainy"])
                successful = st.checkbox("Successfully Observed", value=True)
            with col2:
                notes = st.text_area("Notes", height=100)
                actual_altitude = st.number_input("Actual Altitude (°)", min_value=0.0, max_value=90.0, value=None, step=0.1)
            
            if st.button("💾 Save Observation"):
                observation = {
                    'pass_id': pass_id,
                    'observation_time': datetime.now(),
                    'weather': weather,
                    'successful': successful,
                    'notes': notes if notes else "",
                    'actual_altitude': actual_altitude if actual_altitude else None
                }
                
                new_obs = pd.DataFrame([observation])
                st.session_state.observations_df = pd.concat([st.session_state.observations_df, new_obs], ignore_index=True)
                st.success("✅ Observation saved!")
                
                # Save to CSV
                st.session_state.observations_df.to_csv('iss_observations.csv', index=False)
        
        # Show existing observations
        if len(st.session_state.observations_df) > 0:
            st.subheader("Your Observations")
            st.dataframe(st.session_state.observations_df, use_container_width=True)
    
    with tab4:
        st.header("Analytics")
        
        if len(st.session_state.observations_df) > 0:
            merged_df = pd.merge(
                predictions_df,
                st.session_state.observations_df,
                on='pass_id',
                how='left'
            )
            
            # Success rate
            total_observed = merged_df['successful'].notna().sum()
            if total_observed > 0:
                success_count = merged_df['successful'].sum()
                success_rate = success_count / total_observed
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Success Rate", f"{success_rate:.1%}")
                    st.metric("Total Observations", total_observed)
                
                with col2:
                    # Pie chart
                    fig, ax = plt.subplots(figsize=(6, 6))
                    labels = ['Successful', 'Failed']
                    sizes = [success_count, total_observed - success_count]
                    colors = ['green', 'red']
                    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Observation Success Rate')
                    st.pyplot(fig)
                
                # Weather analysis
                if 'weather' in merged_df.columns:
                    st.subheader("Success by Weather")
                    weather_analysis = merged_df.groupby('weather')['successful'].agg(['mean', 'count']).round(2)
                    weather_analysis.columns = ['Success Rate', 'Count']
                    st.dataframe(weather_analysis)
            else:
                st.info("No observations yet. Log some observations to see analytics!")
        else:
            st.info("No observations yet. Log some observations to see analytics!")
        
        # Export data
        st.subheader("Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_predictions = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv_predictions,
                file_name="iss_predictions.csv",
                mime="text/csv"
            )
        
        with col2:
            if len(st.session_state.observations_df) > 0:
                csv_observations = st.session_state.observations_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Observations",
                    data=csv_observations,
                    file_name="iss_observations.csv",
                    mime="text/csv"
                )
        
        with col3:
            if len(st.session_state.observations_df) > 0:
                merged_df = pd.merge(predictions_df, st.session_state.observations_df, on='pass_id', how='left')
                csv_merged = merged_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Data",
                    data=csv_merged,
                    file_name="iss_full_data.csv",
                    mime="text/csv"
                )

else:
    st.info("👈 Configure your location and click 'Calculate Passes' to get started!")
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Set your location** in the sidebar (latitude, longitude, elevation)
        2. **Adjust prediction settings** (days ahead, minimum altitude)
        3. **Click "Calculate Passes"** to get ISS predictions
        4. **View predictions** in the Predictions tab
        5. **Check "Best Passes"** for high-altitude passes (easiest to see)
        6. **Log observations** after watching a pass
        7. **View analytics** to see your success rate
        """)
    
    st.markdown("""
    ### About
    This app predicts when the International Space Station will be visible from your location.
    The ISS orbits Earth every ~90 minutes, but you can only see it during twilight when it's
    illuminated by the sun while your location is in darkness.
    
    **Altitude Guide:**
    - 0° = Horizon
    - 30° = Good viewing (above most buildings)
    - 90° = Directly overhead
    """)

