import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Load artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("ensemble_solar_model.joblib")
        feature_meta = joblib.load("feature_metadata.joblib")
        config = joblib.load("model_config.joblib")
        return model, feature_meta, config
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, feature_meta, config = load_artifacts()

if model:
    FEATURE_COLUMNS = feature_meta["feature_columns"]
    REGION_MAPPING = feature_meta["region_encoding"]["mapping"]
    REGION_NAME_TO_CODE = {v: k for k, v in REGION_MAPPING.items()}
    RMSE = config["performance"]["test_rmse"]

    # Tier 2: Region-wise Default Presets
    REGION_DEFAULTS = {
        "North": {"t2m": 25.0, "wind": 3.0, "lat": 28.7},
        "South": {"t2m": 30.0, "wind": 2.5, "lat": 13.0},
        "East": {"t2m": 28.0, "wind": 3.2, "lat": 22.5},
        "West": {"t2m": 32.0, "wind": 2.0, "lat": 23.0}
    }

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def generate_sample_csv():
    """Creates a sample CSV with the exact 17 features expected by the model"""
    sample_data = {
        "ssrd_hourly": [500.0, 600.0, 450.0],
        "solar_elevation": [45.0, 50.0, 40.0],
        "clear_sky_index": [0.8, 0.85, 0.75],
        "hour": [12, 13, 11],
        "day_of_year": [180, 180, 180],
        "month": [6, 6, 6],
        "day_of_week": [2, 2, 2],
        "t2m_celsius": [28.0, 29.0, 27.0],
        "temp_dewpoint_diff": [8.0, 7.5, 8.5],
        "relative_humidity": [65.0, 67.0, 63.0],
        "wind_speed": [3.5, 3.0, 4.0],
        "sp": [101325.0, 101320.0, 101330.0],
        "solar_declination": [23.5, 23.5, 23.5],
        "solar_azimuth": [180.0, 185.0, 175.0],
        "theoretical_max_radiation": [1000.0, 1050.0, 950.0],
        "cloud_indicator": [0.2, 0.15, 0.25],
        "region_encoded": [0, 1, 2]
    }
    return pd.DataFrame(sample_data)

def calculate_solar_geometry(hour, day_of_year, latitude=20.0):
    """Calculate solar elevation and related parameters for a given hour"""
    # Equation of Time
    gamma = 2 * np.pi * (day_of_year - 1) / 365
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 
                    0.014615 * np.cos(2*gamma) - 0.040849 * np.sin(2*gamma))
    
    # Declination (Spencer formula)
    dec_rad = 0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - \
              0.006758 * np.cos(2*gamma) + 0.000907 * np.sin(2*gamma)
    solar_declination = np.degrees(dec_rad)
    
    # Hour Angle
    sha_rad = np.radians(15 * (hour - 12) + (eot/4))
    
    # Solar Elevation
    lat_rad = np.radians(latitude)
    sin_elev = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(sha_rad)
    solar_elevation = np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))
    
    # Solar Azimuth
    cos_azi = (np.sin(dec_rad) * np.sin(lat_rad) - np.cos(dec_rad) * np.cos(lat_rad) * np.cos(sha_rad)) / np.cos(np.arcsin(np.clip(sin_elev, -1, 1)))
    solar_azimuth = np.degrees(np.arccos(np.clip(cos_azi, -1, 1)))
    
    # Theoretical max radiation
    theoretical_max_radiation = max(0, 952.7 * np.sin(np.radians(solar_elevation)))
    
    return solar_elevation, solar_declination, solar_azimuth, theoretical_max_radiation

def generate_24hour_curve(base_input, selected_date, region):
    """Generate 24-hour diurnal power curve"""
    hours = np.arange(0, 24)
    day_of_year = selected_date.timetuple().tm_yday
    latitude = REGION_DEFAULTS.get(region, {}).get("lat", 20.0)
    
    predictions = []
    elevations = []
    
    for h in hours:
        row = base_input.copy()
        
        # Calculate solar geometry for this hour
        solar_elev, solar_decl, solar_azi, theo_max_rad = calculate_solar_geometry(h, day_of_year, latitude)
        
        # Update time-dependent features
        row["hour"] = h
        row["solar_elevation"] = solar_elev
        row["solar_declination"] = solar_decl
        row["solar_azimuth"] = solar_azi
        row["theoretical_max_radiation"] = theo_max_rad
        
        # Adjust radiation and sky indices based on solar elevation
        if solar_elev > 0:
            row["ssrd_hourly"] = base_input["ssrd_hourly"] * (solar_elev / base_input["solar_elevation"])
            row["clear_sky_index"] = min(1.0, row["ssrd_hourly"] / theo_max_rad if theo_max_rad > 0 else 0)
            row["cloud_indicator"] = 1 - row["clear_sky_index"]
        else:
            row["ssrd_hourly"] = 0
            row["clear_sky_index"] = 0
            row["cloud_indicator"] = 1
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([row])[FEATURE_COLUMNS]
        pred = max(0.0, model.predict(input_df)[0])
        
        predictions.append(pred)
        elevations.append(solar_elev)
    
    return hours, predictions, elevations

def calculate_impact_metrics(power_kw):
    """Calculate environmental and real-world impact metrics"""
    # Indian context metrics
    metrics = {
        "homes": power_kw / 5.0,  # Average Indian home ~5kW
        "led_bulbs": power_kw * 1000 / 10,  # 10W LED bulb
        "fans": power_kw * 1000 / 75,  # 75W ceiling fan
        "ac_units": power_kw / 1.5,  # 1.5 ton AC
        "co2_saved_kg_hour": power_kw * 0.82,  # India grid emission factor ~0.82 kg CO2/kWh
        "co2_saved_kg_day": power_kw * 0.82 * 8,  # Assuming 8 hours of generation
        "trees_equivalent": power_kw * 0.82 * 8 / 21,  # Tree absorbs ~21kg CO2/year
    }
    return metrics

def generate_pdf_report(prediction_data, power_kw, impact_metrics, diurnal_data=None):
    """Generate a comprehensive PDF report"""
    buffer = BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 1: Summary Report
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Solar Power Prediction Report', fontsize=16, fontweight='bold', y=0.98)
        
        # Header info
        plt.text(0.5, 0.92, f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='center', fontsize=10, style='italic')
        plt.text(0.5, 0.89, f'Region: {prediction_data.get("region", "N/A")}', 
                ha='center', fontsize=10)
        
        # Main Prediction
        plt.text(0.5, 0.80, '☀️ Predicted Solar Power', ha='center', fontsize=14, fontweight='bold')
        plt.text(0.5, 0.75, f'{power_kw:.2f} kW', ha='center', fontsize=24, color='#FF8C00')
        plt.text(0.5, 0.70, f'Confidence Range: [{power_kw-RMSE:.2f}, {power_kw+RMSE:.2f}] kW', 
                ha='center', fontsize=10, style='italic')
        
        # Environmental Impact
        plt.text(0.5, 0.62, '🌍 Environmental Impact (per hour)', ha='center', fontsize=12, fontweight='bold')
        impact_text = f"""
        CO₂ Emissions Saved: {impact_metrics['co2_saved_kg_hour']:.2f} kg
        Equivalent to {impact_metrics['trees_equivalent']:.1f} trees per day
        
        ⚡ Real-World Equivalents
        Powers {impact_metrics['homes']:.1f} average homes
        Powers {impact_metrics['led_bulbs']:.0f} LED bulbs
        Powers {impact_metrics['fans']:.0f} ceiling fans
        Powers {impact_metrics['ac_units']:.1f} air conditioners (1.5 ton)
        """
        plt.text(0.5, 0.50, impact_text, ha='center', fontsize=9, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Model Performance
        plt.text(0.5, 0.30, '📊 Model Performance Metrics', ha='center', fontsize=12, fontweight='bold')
        perf_text = f"""
        R² Score: {config['performance']['test_r2']:.4f}
        RMSE: {RMSE:.2f} kW
        MAE: {config['performance']['test_mae']:.2f} kW
        """
        plt.text(0.5, 0.22, perf_text, ha='center', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Footer
        plt.text(0.5, 0.05, 'Solar Power Prediction Platform • ML + Physics Engine', 
                ha='center', fontsize=8, style='italic', color='gray')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: 24-Hour Diurnal Curve (if available)
        if diurnal_data:
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
            fig2.suptitle('24-Hour Power Generation Analysis', fontsize=14, fontweight='bold')
            
            hours, predictions, elevations = diurnal_data
            
            # Power curve
            ax1.plot(hours, predictions, color='#FF8C00', linewidth=2.5, marker='o', markersize=4)
            ax1.fill_between(hours, predictions, alpha=0.3, color='#FFD700')
            ax1.set_xlabel('Hour of Day', fontsize=11)
            ax1.set_ylabel('Predicted Power (kW)', fontsize=11)
            ax1.set_title('Diurnal Power Generation Curve', fontsize=12, pad=20)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 23)
            ax1.set_xticks(range(0, 24, 2))
            
            # Add sunrise/sunset markers
            sunrise_idx = next((i for i, e in enumerate(elevations) if e > 0), None)
            sunset_idx = next((i for i in range(len(elevations)-1, -1, -1) if elevations[i] > 0), None)
            if sunrise_idx:
                ax1.axvline(sunrise_idx, color='orange', linestyle='--', alpha=0.5, label='Sunrise')
            if sunset_idx:
                ax1.axvline(sunset_idx, color='red', linestyle='--', alpha=0.5, label='Sunset')
            ax1.legend()
            
            # Solar elevation curve
            ax2.plot(hours, elevations, color='#4169E1', linewidth=2, marker='s', markersize=4)
            ax2.fill_between(hours, elevations, alpha=0.3, color='#87CEEB')
            ax2.set_xlabel('Hour of Day', fontsize=11)
            ax2.set_ylabel('Solar Elevation (°)', fontsize=11)
            ax2.set_title('Solar Elevation Throughout Day', fontsize=12, pad=10)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlim(0, 23)
            ax2.set_xticks(range(0, 24, 2))
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close()
    
    buffer.seek(0)
    return buffer

def intermediate_physics_engine(df, selected_region_name):
    """
    Transforms raw ERA5 data (t2m, d2m, u10, v10, ssrd, sp, lat, lon, time)
    into the 17 engineered features required by the ML model.
    """
    # Define required ERA5 columns
    REQUIRED_ERA5_COLS = ['valid_time', 't2m', 'd2m', 'u10', 'v10', 'ssrd', 'sp']
    
    # Check for missing columns
    missing_cols = [col for col in REQUIRED_ERA5_COLS if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required ERA5 columns: {', '.join(missing_cols)}. Your CSV must contain: {', '.join(REQUIRED_ERA5_COLS)}")
    
    df = df.copy()
    
    # 1. Temporal Extraction
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    df['hour'] = df['valid_time'].dt.hour
    df['month'] = df['valid_time'].dt.month
    df['day_of_year'] = df['valid_time'].dt.dayofyear
    df['day_of_week'] = df['valid_time'].dt.dayofweek
    
    # 2. Atmospheric & Weather (Kelvin to Celsius, RH, Wind Speed)
    if 't2m_celsius' not in df.columns and 't2m' in df.columns:
        df['t2m_celsius'] = df['t2m'] - 273.15
    if 'd2m_celsius' not in df.columns and 'd2m' in df.columns:
        df['d2m_celsius'] = df['d2m'] - 273.15
        
    df['temp_dewpoint_diff'] = df['t2m_celsius'] - df['d2m_celsius']
    
    # Magnus-Tetens Relative Humidity formula
    es = 6.112 * np.exp((17.67 * df["t2m_celsius"]) / (df["t2m_celsius"] + 243.5))
    e = 6.112 * np.exp((17.67 * df["d2m_celsius"]) / (df["d2m_celsius"] + 243.5))
    df["relative_humidity"] = (e / es) * 100

    if 'wind_speed' not in df.columns and 'u10' in df.columns:
        df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)

    # 3. Radiation Unit Conversion (Joules to average Watts)
    if 'ssrd_hourly' not in df.columns and 'ssrd' in df.columns:
        df['ssrd_hourly'] = df['ssrd'] / 3600.0

    # 4. Solar Geometry (Intermediate Physics: Elevation, Azimuth, Declination)
    lat_rad = np.radians(df['latitude'] if 'latitude' in df.columns else 20.0)
    doy = df['day_of_year']
    
    # Equation of Time (EoT) correction
    gamma = 2 * np.pi * (doy - 1) / 365
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 
                    0.014615 * np.cos(2*gamma) - 0.040849 * np.sin(2*gamma))
    
    # Declination (Spencer formula)
    dec_rad = 0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - \
              0.006758 * np.cos(2*gamma) + 0.000907 * np.sin(2*gamma)
    df['solar_declination'] = np.degrees(dec_rad)
    
    # Hour Angle (SHA) corrected with EoT
    sha_rad = np.radians(15 * (df['hour'] - 12) + (eot/4))
    
    # Solar Elevation calculation
    sin_elev = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(sha_rad)
    df['solar_elevation'] = np.degrees(np.arcsin(sin_elev))
    
    # Solar Azimuth calculation
    cos_azi = (np.sin(dec_rad) * np.sin(lat_rad) - np.cos(dec_rad) * np.cos(lat_rad) * np.cos(sha_rad)) / np.cos(np.arcsin(sin_elev))
    df['solar_azimuth'] = np.degrees(np.arccos(np.clip(cos_azi, -1, 1)))

    # 5. Sky Condition Indices
    # Max radiation scaling matched to your master dataset
    df['theoretical_max_radiation'] = (952.7 * np.sin(np.radians(df['solar_elevation']))).clip(0)
    df['clear_sky_index'] = (df['ssrd_hourly'] / df['theoretical_max_radiation']).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 1)
    df['cloud_indicator'] = 1 - df['clear_sky_index']

    # 6. Region Mapping
    df['region_encoded'] = REGION_NAME_TO_CODE.get(selected_region_name, 0)
    
    return df[FEATURE_COLUMNS], df # Return both model features and full data

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("☀️ Solar Power Prediction Platform")
st.caption("ML-powered solar generation forecasting • Azure-ready • College Project")

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("R²", f"{config['performance']['test_r2']:.4f}")
st.sidebar.metric("RMSE (kW)", f"{RMSE:.2f}")
st.sidebar.metric("MAE (kW)", f"{config['performance']['test_mae']:.2f}")

with st.sidebar.expander("🗺 Region Mapping"):
    for k, v in REGION_MAPPING.items():
        st.write(f"{v} → {k}")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Single Prediction",
    "📈 Visualization",
    "🧪 Batch Prediction",
    "🧠 ERA5 Auto-Features",
    "ℹ️ Model Info"
])

# =============================================================================
# TAB 1 — SINGLE PREDICTION (Enhanced with Impact Calculator & PDF Export)
# =============================================================================
with tab1:
    st.subheader("Single Prediction")

    with st.form("single_prediction"):
        col1, col2, col3 = st.columns(3)

        with col1:
            # Tier 1: Units & Tooltips
            ssrd_hourly = st.number_input("SSRD Hourly", value=500.0, help="Surface Solar Radiation Downwards (W/m² per hour)")
            solar_elevation = st.number_input("Solar Elevation (°)", value=45.0, help="Angle of sun above horizon")
            clear_sky_index = st.slider("Clear Sky Index", 0.0, 1.0, 0.8, help="Ratio of actual to clear-sky radiation")
            cloud_indicator = st.slider("Cloud Indicator", 0.0, 1.0, 0.2, help="Higher = more cloud cover")
            region = st.selectbox("Region", list(REGION_NAME_TO_CODE.keys()))

        # Tier 2: Region presets logic
        defaults = REGION_DEFAULTS.get(region, {})

        with col2:
            selected_date = st.date_input("Select Date", datetime.date.today())
            hour = st.number_input("Hour", 0, 23, 12)
            month = selected_date.month
            day_of_year = selected_date.timetuple().tm_yday
            day_of_week = selected_date.weekday()
            solar_declination = st.number_input("Solar Declination", value=23.5, help="Earth tilt-based solar declination angle")

        with col3:
            t2m_celsius = st.number_input("Temperature (°C)", value=defaults.get("t2m", 28.0))
            temp_dewpoint_diff = st.number_input("Temp–Dewpoint Diff", value=8.0)
            relative_humidity = st.number_input("Relative Humidity (%)", value=65.0)
            wind_speed = st.number_input("Wind Speed (m/s)", value=defaults.get("wind", 3.5), help="Derived from u10 & v10")
            sp = st.number_input("Surface Pressure (Pa)", value=101325.0)

        col_left, col_right = st.columns(2)
        with col_left:
            solar_azimuth = st.number_input("Solar Azimuth", value=180.0)
        with col_right:
            theoretical_max_radiation = st.number_input("Theoretical Max Radiation", value=1000.0)

        # Tier 1: Input Validation
        input_error = False
        if clear_sky_index < 0 or clear_sky_index > 1:
            st.warning("⚠️ Clear sky index must be between 0 and 1.")
            input_error = True
        if solar_elevation < 0:
            st.warning("⚠️ Solar elevation cannot be negative.")
            input_error = True

        submit = st.form_submit_button("Predict", disabled=input_error)

    if submit:
        with st.spinner("🔄 Computing prediction..."):
            input_dict = {
                "ssrd_hourly": ssrd_hourly, "solar_elevation": solar_elevation, "clear_sky_index": clear_sky_index,
                "hour": hour, "day_of_year": day_of_year, "month": month, "day_of_week": day_of_week,
                "t2m_celsius": t2m_celsius, "temp_dewpoint_diff": temp_dewpoint_diff, "relative_humidity": relative_humidity,
                "wind_speed": wind_speed, "sp": sp, "solar_declination": solar_declination, "solar_azimuth": solar_azimuth,
                "theoretical_max_radiation": theoretical_max_radiation, "cloud_indicator": cloud_indicator,
                "region_encoded": REGION_NAME_TO_CODE[region]
            }

            # Save for visualization
            st.session_state["last_input"] = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
            st.session_state["last_input_dict"] = input_dict
            st.session_state["last_region"] = region
            st.session_state["last_date"] = selected_date
            
            # Fixed: Forces prediction to be 0 or higher
            raw_pred = model.predict(st.session_state["last_input"])[0]
            pred = max(0.0, raw_pred)
            st.session_state["last_prediction"] = pred

            # Tier 2: Prediction History with timestamps
            if "history" not in st.session_state: 
                st.session_state["history"] = []
            if "history_timestamps" not in st.session_state:
                st.session_state["history_timestamps"] = []
            
            st.session_state["history"].append(pred)
            st.session_state["history_timestamps"].append(datetime.datetime.now())

        st.success(f"☀️ Predicted Power: **{pred:.2f} kW**")
        st.info(f"📊 Confidence Band: **[{max(0, pred - RMSE):.2f}, {pred + RMSE:.2f}] kW**")

        # NEW: Impact Calculator
        st.markdown("---")
        st.markdown("### 💡 Real-World Impact Analysis")
        
        impact = calculate_impact_metrics(pred)
        
        col_imp1, col_imp2, col_imp3 = st.columns(3)
        
        with col_imp1:
            st.metric("🏠 Homes Powered", f"{impact['homes']:.1f}")
            st.metric("💡 LED Bulbs", f"{int(impact['led_bulbs'])}")
            
        with col_imp2:
            st.metric("🌪️ Fans Powered", f"{int(impact['fans'])}")
            st.metric("❄️ AC Units", f"{impact['ac_units']:.1f}")
            
        with col_imp3:
            st.metric("🌍 CO₂ Saved (hour)", f"{impact['co2_saved_kg_hour']:.2f} kg")
            st.metric("🌳 Trees/day Equivalent", f"{impact['trees_equivalent']:.1f}")
        
        st.info(f"💚 **Daily Impact:** Generating at this rate for 8 hours would save **{impact['co2_saved_kg_day']:.2f} kg** of CO₂ emissions")

        # Tier 1: Feature Summary Panel
        with st.expander("🔍 Model Input Summary"):
            st.dataframe(st.session_state["last_input"])

        # Download Options
        col_csv, col_pdf = st.columns(2)
        
        with col_csv:
            report = st.session_state["last_input"].copy()
            report["predicted_power_kW"] = pred
            st.download_button("⬇️ Download CSV Report", report.to_csv(index=False), "single_prediction_report.csv")
        
        with col_pdf:
            # NEW: PDF Report Generation
            if st.button("📄 Generate PDF Report"):
                with st.spinner("Generating comprehensive PDF report..."):
                    # Generate 24-hour curve for PDF
                    diurnal_data = generate_24hour_curve(input_dict, selected_date, region)
                    
                    prediction_info = {
                        "region": region,
                        "date": selected_date.strftime("%Y-%m-%d"),
                        "hour": hour
                    }
                    
                    pdf_buffer = generate_pdf_report(prediction_info, pred, impact, diurnal_data)
                    
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"solar_prediction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF report generated successfully!")

# =============================================================================
# TAB 2 — ENHANCED VISUALIZATION (Added 24-Hour Diurnal Curve)
# =============================================================================
with tab2:
    st.subheader("Model Insights")

    if "last_input" not in st.session_state:
        st.warning("Run a single prediction first.")
    else:
        # Reset Session History Button
        if st.button("🔄 Reset Session History"):
            st.session_state["history"] = []
            st.session_state["history_timestamps"] = []
            st.rerun()
        
        # NEW: 24-Hour Diurnal Curve - THE "WOW" FACTOR
        st.markdown("### 🌅 24-Hour Power Generation Forecast")
        st.info("🌟 **Golden Hour Analysis:** See how your solar power generation changes throughout the day")
        
        if "last_input_dict" in st.session_state and "last_date" in st.session_state and "last_region" in st.session_state:
            with st.spinner("Calculating full-day solar trajectory..."):
                hours, predictions, elevations = generate_24hour_curve(
                    st.session_state["last_input_dict"],
                    st.session_state["last_date"],
                    st.session_state["last_region"]
                )
                
                # Create beautiful dual-axis plot
                fig_diurnal, ax1 = plt.subplots(figsize=(12, 6))
                
                # Power generation curve
                color_power = '#FF8C00'
                ax1.plot(hours, predictions, color=color_power, linewidth=3, marker='o', markersize=5, label='Predicted Power')
                ax1.fill_between(hours, predictions, alpha=0.3, color='#FFD700')
                ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Predicted Power (kW)', fontsize=12, fontweight='bold', color=color_power)
                ax1.tick_params(axis='y', labelcolor=color_power)
                ax1.grid(True, alpha=0.3, linestyle='--')
                ax1.set_xlim(-0.5, 23.5)
                
                # Solar elevation on secondary axis
                ax2 = ax1.twinx()
                color_elev = '#4169E1'
                ax2.plot(hours, elevations, color=color_elev, linewidth=2, linestyle='--', marker='s', markersize=4, alpha=0.7, label='Solar Elevation')
                ax2.set_ylabel('Solar Elevation (°)', fontsize=12, fontweight='bold', color=color_elev)
                ax2.tick_params(axis='y', labelcolor=color_elev)
                ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
                
                # Mark sunrise and sunset
                sunrise_idx = next((i for i, e in enumerate(elevations) if e > 0), None)
                sunset_idx = next((i for i in range(len(elevations)-1, -1, -1) if elevations[i] > 0), None)
                
                if sunrise_idx is not None:
                    ax1.axvline(sunrise_idx, color='orange', linestyle=':', linewidth=2, alpha=0.6, label=f'Sunrise (~{sunrise_idx}:00)')
                if sunset_idx is not None:
                    ax1.axvline(sunset_idx, color='red', linestyle=':', linewidth=2, alpha=0.6, label=f'Sunset (~{sunset_idx}:00)')
                
                # Peak generation marker
                peak_idx = np.argmax(predictions)
                peak_power = predictions[peak_idx]
                ax1.scatter([peak_idx], [peak_power], color='gold', s=200, zorder=5, edgecolors='red', linewidths=2)
                ax1.annotate(f'Peak: {peak_power:.1f} kW', xy=(peak_idx, peak_power), 
                            xytext=(peak_idx, peak_power + max(predictions)*0.1),
                            ha='center', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2))
                
                ax1.legend(loc='upper left', fontsize=10)
                plt.title('Diurnal Solar Power Generation Curve', fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig_diurnal)
                
                # Daily statistics
                total_daily = sum(predictions)
                avg_daily = np.mean([p for p in predictions if p > 0])
                generation_hours = sum(1 for p in predictions if p > 0)
                
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("🌞 Peak Power", f"{peak_power:.2f} kW", f"at {peak_idx}:00")
                with col_stat2:
                    st.metric("⚡ Total Daily", f"{total_daily:.1f} kWh")
                with col_stat3:
                    st.metric("📊 Avg (daylight)", f"{avg_daily:.2f} kW")
                with col_stat4:
                    st.metric("⏰ Generation Hours", f"{generation_hours} hrs")
                
                # Daily CO2 savings
                daily_co2 = total_daily * 0.82
                st.success(f"🌍 **Daily Environmental Impact:** This generation pattern would save **{daily_co2:.2f} kg** of CO₂ emissions (equivalent to **{daily_co2/21:.1f}** trees)")
        
        st.markdown("---")
        
        # Solar Elevation Impact
        st.markdown("### 🌞 Solar Elevation Sensitivity")
        elevations = np.linspace(5, 85, 30)
        base = st.session_state["last_input"].iloc[0].to_dict()
        rows = []
        for e in elevations:
            r = base.copy()
            r["solar_elevation"] = e
            rows.append(r)
        df_plot = pd.DataFrame(rows)[FEATURE_COLUMNS]
        preds_elev = model.predict(df_plot)

        fig1, ax1 = plt.subplots()
        ax1.plot(elevations, preds_elev, color='#FFA500', linewidth=2)
        ax1.set_xlabel("Solar Elevation (°)")
        ax1.set_ylabel("Predicted Power (kW)")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

        # Temperature Sensitivity Analysis
        st.markdown("### 🌡️ Temperature Sensitivity")
        temps = np.linspace(15, 45, 30)
        rows_temp = []
        for t in temps:
            r = base.copy()
            r["t2m_celsius"] = t
            rows_temp.append(r)
        df_temp = pd.DataFrame(rows_temp)[FEATURE_COLUMNS]
        preds_temp = model.predict(df_temp)

        fig2, ax2 = plt.subplots()
        ax2.plot(temps, preds_temp, color='#FF6347', linewidth=2)
        ax2.set_xlabel("Temperature (°C)")
        ax2.set_ylabel("Predicted Power (kW)")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

        # Cloud Cover Impact
        st.markdown("### ☁️ Cloud Cover Impact")
        clouds = np.linspace(0, 1, 30)
        rows_cloud = []
        for c in clouds:
            r = base.copy()
            r["cloud_indicator"] = c
            r["clear_sky_index"] = 1 - c
            rows_cloud.append(r)
        df_cloud = pd.DataFrame(rows_cloud)[FEATURE_COLUMNS]
        preds_cloud = model.predict(df_cloud)

        fig3, ax3 = plt.subplots()
        ax3.plot(clouds, preds_cloud, color='#4682B4', linewidth=2)
        ax3.set_xlabel("Cloud Indicator (0=clear, 1=overcast)")
        ax3.set_ylabel("Predicted Power (kW)")
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)

        # Tier 2: Feature Importance
        if hasattr(model, "feature_importances_"):
            st.markdown("### 📊 Feature Importance")
            fi = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
            st.bar_chart(fi.set_index("Feature"))

        # Enhanced Prediction History Chart with Timestamps
        if len(st.session_state.get("history", [])) > 1:
            st.markdown("### 📈 Session Prediction History")
            history_df = pd.DataFrame({
                "Timestamp": st.session_state["history_timestamps"],
                "Predicted Power (kW)": st.session_state["history"]
            })
            st.line_chart(history_df.set_index("Timestamp"))

# =============================================================================
# TAB 3 — BATCH PREDICTION (Enhanced with Sample CSV Download)
# =============================================================================
with tab3:
    st.subheader("🧪 Production Batch Simulator")
    st.markdown("""
    **Purpose:** Upload a dataset that is already pre-processed (17 features). 
    This simulates how the model would perform in a real-time production pipeline.
    """)
    
    # Sample CSV Download Button
    col_download, col_upload = st.columns([1, 2])
    with col_download:
        sample_csv = generate_sample_csv()
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_csv.to_csv(index=False),
            file_name="sample_17_features.csv",
            mime="text/csv",
            help="Download this template to see the exact format required"
        )
    
    with col_upload:
        file = st.file_uploader("Upload Model-Ready CSV", type=["csv"], key="batch_uploader")

    if file:
        df = pd.read_csv(file)
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        # Auto-convert 'region' to 'region_encoded' if present
        if 'region' in df.columns and 'region_encoded' not in df.columns:
            df['region_encoded'] = df['region'].map(REGION_NAME_TO_CODE)
            if df['region_encoded'].isna().any():
                st.warning(f"⚠️ Some region values couldn't be mapped. Valid values: {list(REGION_NAME_TO_CODE.keys())}")
                st.write("Unmapped regions:", df[df['region_encoded'].isna()]['region'].unique().tolist())

        # Check if all 17 features exist
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        
        if missing:
            st.error(f"❌ Feature Mismatch! Missing columns: {', '.join(missing)}")
            st.info("💡 Hint: Download the sample CSV template above or use the 'ERA5 Auto-Features' tab for raw data.")
            
            # Show what columns were found vs expected
            with st.expander("🔍 Debug: Column Comparison"):
                st.write("**Expected columns:**", FEATURE_COLUMNS)
                st.write("**Your columns:**", df.columns.tolist())
        else:
            with st.spinner("Processing batch..."):
                # REORDER columns to match model training exactly
                model_ready_df = df[FEATURE_COLUMNS]
                preds = model.predict(model_ready_df)
                # Fixed: Clips the entire array so no value is below 0
                df["predicted_power_kW"] = np.clip(preds, 0, None)
                df["lower_bound"] = (df["predicted_power_kW"] - RMSE).clip(0)
                df["upper_bound"] = df["predicted_power_kW"] + RMSE

            st.success(f"✅ Successfully processed {len(df)} records.")
            
            # Show summary statistics
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Mean Power", f"{df['predicted_power_kW'].mean():.2f} kW")
            with col_stats2:
                st.metric("Max Power", f"{df['predicted_power_kW'].max():.2f} kW")
            with col_stats3:
                st.metric("Min Power", f"{df['predicted_power_kW'].min():.2f} kW")
            
            st.dataframe(df.head(10))
            
            # Time series visualization if timestamp column exists
            if any(col in df.columns for col in ['valid_time', 'timestamp', 'time']):
                st.markdown("### 📊 Batch Prediction Time Series")
                time_col = next((col for col in ['valid_time', 'timestamp', 'time'] if col in df.columns), None)
                if time_col:
                    df_plot = df.copy()
                    df_plot[time_col] = pd.to_datetime(df_plot[time_col])
                    df_plot = df_plot.sort_values(time_col)
                    
                    fig_batch, ax_batch = plt.subplots(figsize=(10, 4))
                    ax_batch.plot(df_plot[time_col], df_plot['predicted_power_kW'], color='#32CD32', linewidth=1.5)
                    ax_batch.fill_between(df_plot[time_col], df_plot['lower_bound'], df_plot['upper_bound'], alpha=0.2, color='#32CD32')
                    ax_batch.set_xlabel("Time")
                    ax_batch.set_ylabel("Predicted Power (kW)")
                    ax_batch.grid(alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_batch)
            
            # Correlation heatmap for batch data
            if len(df) > 1:
                st.markdown("### 🔥 Feature Correlation Heatmap")
                corr_cols = [col for col in FEATURE_COLUMNS if col in df.columns][:8]  # Limit to avoid clutter
                if len(corr_cols) > 3:
                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                    corr_matrix = df[corr_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax_corr)
                    st.pyplot(fig_corr)
            
            st.download_button(
                "⬇️ Download Batch Predictions",
                df.to_csv(index=False),
                "batch_solar_predictions.csv"
            )
            
# =============================================================================
# TAB 4 — ERA5 AUTO FEATURE ENGINE (Enhanced Error Handling)
# =============================================================================
with tab4:
    st.subheader("ERA5 Physics-Informed Engine")
    st.info("Upload RAW ERA5 Data. This system will automatically calculate sun angles, wind magnitude, and sky indices.")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        era5_file = st.file_uploader("Upload RAW ERA5 CSV", type=["csv"], key="era5_batch")
    with col_b:
        chosen_region = st.selectbox("Select Target Region", list(REGION_NAME_TO_CODE.keys()), key="region_batch")

    if era5_file:
        df_raw = pd.read_csv(era5_file)
        
        try:
            with st.status("Performing Physics Engineering...", expanded=True) as status:
                st.write("Extracting timestamps...")
                st.write("Calculating Earth orbital parameters (EoT)...")
                st.write("Deriving weather and sky indices...")
                model_features, full_data = intermediate_physics_engine(df_raw, chosen_region)
                status.update(label="Feature Engineering Complete!", state="complete", expanded=False)

            # Run prediction on the engineered features
            preds = model.predict(model_features)
            
            # Convert to a Series to ensure index alignment
            full_data["predicted_power_kW"] = np.clip(preds, 0, None)
            full_data["lower_bound"] = np.maximum(0, full_data["predicted_power_kW"].values - RMSE)
            full_data["upper_bound"] = full_data["predicted_power_kW"].values + RMSE
            
            st.success(f"☀️ Power prediction completed successfully for {len(full_data)} rows!")
            
            # Display logic...
            st.dataframe(full_data[['valid_time', 'solar_elevation', 'predicted_power_kW']].head(10))
            st.download_button("⬇️ Download Full Results", full_data.to_csv(index=False), "era5_predictions.csv")
            
        except KeyError as e:
            st.error(f"❌ Missing Required ERA5 Column: {str(e)}")
            st.info("""
            **Required ERA5 columns:**
            - `valid_time` (timestamp)
            - `t2m` (2m temperature in Kelvin)
            - `d2m` (2m dewpoint temperature in Kelvin)
            - `u10` (10m U wind component)
            - `v10` (10m V wind component)
            - `ssrd` (surface solar radiation downwards)
            - `sp` (surface pressure)
            
            Optional: `latitude`, `longitude`
            """)

# =============================================================================
# TAB 5 — MODEL INFO (Enhanced with Architecture Diagram)
# =============================================================================
with tab5:
    st.markdown("""
    ## 🔍 Model Overview
    **Model Type:** Stacked Ensemble Regressor (XGBoost + Random Forest)  
    **Dataset:** ERA5 Weather Reanalysis + Solar Physics derivations (~70,000 samples)  
    **Purpose:** Academic demonstration of physics-informed machine learning for renewable energy.
    
    ### 📈 Model Architecture Pipeline
    ```
    ERA5 Raw Data → Physics Engine → 17 Engineered Features → Ensemble Model → Power Prediction
                         ↓                                            ↓
                 (Solar Geometry,                           (XGBoost + RandomForest
                  Weather Indices)                           with cross-validation)
    ```
    
    ### Key Features Calculated:
    - **Solar Geometry:** Azimuth and Elevation corrected with Equation of Time (EoT)
    - **Atmospheric Physics:** Dewpoint deficit and Magnus-Tetens humidity formula
    - **Sky Indices:** Clear Sky Index and Cloud Indicator
    - **Temporal Features:** Hour, day of year, month, day of week
    
    ### Model Performance:
    - **R² Score:** {:.4f} - Explains {:.1f}% of variance in solar power generation
    - **RMSE:** {:.2f} kW - Average prediction error
    - **MAE:** {:.2f} kW - Mean absolute error
    
    ### Physics Formulas Used:
    
    **1. Equation of Time (EoT):**
    ```
    γ = 2π(n-1)/365
    EoT = 229.18 × (0.000075 + 0.001868cos(γ) - 0.032077sin(γ) - ...)
    ```
    
    **2. Solar Declination (Spencer):**
    ```
    δ = 0.006918 - 0.399912cos(γ) + 0.070257sin(γ) - ...
    ```
    
    **3. Magnus-Tetens Relative Humidity:**
    ```
    eₛ = 6.112 × exp(17.67T/(T+243.5))
    RH = (e/eₛ) × 100
    ```
    
    ### 🎯 Unique Features of This System:
    1. **Physics-Informed ML:** Combines domain knowledge with machine learning
    2. **24-Hour Forecasting:** Can simulate full diurnal power curves
    3. **Impact Analysis:** Translates technical predictions into real-world metrics
    4. **Multi-Path Input:** Supports both pre-processed and raw ERA5 data
    5. **Professional Reporting:** Generates comprehensive PDF reports
    
    ### 📚 References:
    - ERA5 Reanalysis Data: European Centre for Medium-Range Weather Forecasts
    - Solar Position Algorithm: Spencer (1971) & Michalsky (1988)
    - Atmospheric Models: Magnus-Tetens Formula for RH calculation
    """.format(
        config['performance']['test_r2'],
        config['performance']['test_r2'] * 100,
        RMSE,
        config['performance']['test_mae']
    ))
    
    st.markdown("---")
    st.info("💡 **Pro Tip:** Use the 24-hour diurnal curve in Tab 2 to understand how solar angle affects power throughout the day!")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Solar Power Prediction System • Final Project Ready • ML + Physics Engine • Enhanced with Impact Analysis")