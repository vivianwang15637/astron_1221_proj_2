# ISS Pass Predictor & Logger

Predict when the International Space Station will be visible from your location and track your observation history. Uses Skyfield for satellite calculations and Pandas for data analysis.

## Features

- Downloads current ISS TLE data from Celestrak
- Predicts visible passes for the next 7 days
- Logs observations with weather conditions
- Analyzes prediction accuracy
- Generates visualizations and exports to CSV

## Quick Start

### Installation

```bash
git clone https://github.com/vivianwang15637/astron_1221_proj_2.git
cd astron_1221_proj_2
pip install -r requirements.txt
```

### Setup

1. Open `main.ipynb` in Jupyter Notebook
2. Edit location coordinates in Section 1.3:
   ```python
   latitude = 39.9612    # Your latitude
   longitude = -82.9988  # Your longitude (negative for west)
   elevation = 275       # Elevation in meters
   ```
3. Run cells sequentially

## Usage

### Getting Predictions

Run cells 1-7 to calculate passes. The `predictions_df` DataFrame shows all upcoming passes. Use `good_passes_df` to filter passes above 30° altitude.

### Logging Observations

Edit the observations DataFrame in Section 2.3 to record your viewing attempts. Match observations to predictions using `pass_id` and include weather conditions and success status.

### Viewing Results

- Bar chart: pass frequency per day
- Pie chart: observation success rate
- CSV files: all data exported for analysis

## Data Sources

- **Celestrak**: TLE data for ISS (NORAD ID 25544), updated daily
- **Skyfield**: JPL DE421 ephemeris (~10MB, auto-downloaded on first run)

## Output Files

- `iss_predictions.csv` - All predicted passes
- `iss_observations.csv` - Logged observations
- `iss_full_data.csv` - Merged predictions and observations

## Project Structure

```
astron_1221_proj_2/
├── main.ipynb                 # Main notebook
├── requirements.txt           # Dependencies
├── iss_predictions.csv        # Generated predictions
├── iss_observations.csv       # Generated observations
└── iss_full_data.csv          # Merged dataset
```

## Key Concepts

- **TLE Data**: Orbital parameters used to predict satellite positions (updated daily)
- **Pass**: ISS visibility event from rise to set
- **Altitude**: Height above horizon in degrees (0° = horizon, 90° = overhead)
- **Apparent Magnitude**: Brightness measure (lower = brighter, ISS typically -2 to +3)

## Methodology

Uses **Pathway B: Pandas + Astropy + Skyfield**:
- Skyfield for orbit calculations
- Pandas for data manipulation
- Astropy for coordinate transformations
- Matplotlib for visualizations

## StreamLit
https://astron1221proj2-lkncoo4gflbke3bmrrbbvm.streamlit.app/

## References

- [Celestrak](https://celestrak.org/) - TLE data source
- [Skyfield](https://rhodesmill.org/skyfield/) - Satellite calculations
- [Astropy](https://docs.astropy.org/) - Astronomical tools
