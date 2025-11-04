# Streamlit Web App - Quick Start

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser automatically!

## Features

- 🌍 **Set your location** - Enter latitude, longitude, and elevation
- 🔮 **Predict ISS passes** - Get predictions for the next 1-14 days
- ⭐ **Filter best passes** - View only high-altitude passes (easiest to see)
- 📝 **Log observations** - Record your actual viewing attempts
- 📊 **View analytics** - See success rates and weather analysis
- 📥 **Export data** - Download predictions and observations as CSV

## Usage Tips

- **Altitude threshold**: Passes above 30° are usually easiest to see
- **Minimum altitude**: Set to 10° to filter out very low passes
- **Best viewing**: Look for passes with high max altitude (>50°)
- **Timing**: The ISS is visible during twilight (dawn/dusk)

## Notes

- First run will download ~10MB of ephemeris data (one-time download)
- TLE data is cached for 1 hour to avoid repeated downloads
- Observations are saved to `iss_observations.csv`
