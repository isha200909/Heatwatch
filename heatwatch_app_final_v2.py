import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import warnings

st.set_page_config(page_title="HeatWatch — Climate's Human Cost", layout="wide")

st.markdown(
    """
    <style>
    button[role="tab"] {
        font-size: 1.05rem !important;
    }
    button[role="tab"] p {
        font-size: 1.05rem !important;
    }

    /* ── HeatWatch result card ── */
    .hw-card {
        border: 0.5px solid #d1d5db;
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
        margin-top: 1.25rem;
    }
    .hw-top {
        padding: 20px 24px 16px;
        display: flex;
        gap: 24px;
        align-items: center;
        border-bottom: 0.5px solid #e5e7eb;
    }
    .hw-hero { flex: 0 0 190px; }
    .hw-hero-number {
        font-size: 62px; font-weight: 600; line-height: 1;
        letter-spacing: -1px; color: #111827;
    }
    .hw-hero-label {
        font-size: 15px; color: #374151;
        margin-top: 4px; line-height: 1.4;
    }
    .hw-divider-v {
        width: 1px; background: #e5e7eb;
        align-self: stretch; flex-shrink: 0;
    }
    .hw-bar-zone { flex: 1; display: flex; flex-direction: column; gap: 6px; }
    .hw-bar-title { font-size: 15px; color: #374151; }
    .hw-bar-wrap {
        position: relative; height: 10px;
        background: #f3f4f6; border-radius: 99px;
    }
    .hw-bar-fill { height: 100%; border-radius: 99px 0 0 99px; }
    .hw-bar-marker {
        position: absolute; top: -5px; width: 2px; height: 20px;
        border-radius: 1px; background: #4b5563; opacity: 0.8;
    }
    .hw-bar-dot {
        position: absolute; top: 50%; width: 16px; height: 16px;
        border-radius: 50%; border: 2.5px solid #ffffff;
        transform: translate(-50%, -50%);
    }
    .hw-bar-labels {
        display: flex; justify-content: space-between;
        font-size: 13px; color: #4b5563;
    }
    .hw-bar-status { font-size: 14px; font-weight: 600; }
    .hw-zero-callout {
        flex: 1; 
        display: flex; flex-direction: column; gap: 4px; justify-content: center;
    }
    .hw-zero-title { font-size: 16px; font-weight: 600; color: #111827; }
    .hw-zero-sub { font-size: 15px; color: #374151; line-height: 1.5; }
    .hw-vuln {
        padding: 12px 24px;
        display: flex; align-items: flex-start; gap: 24px;
        border-bottom: 0.5px solid #e5e7eb;
    }
    .hw-vuln-left {
        display: flex; align-items: center; gap: 8px;
        flex: 0 0 190px;
    }
    .hw-vuln-dot {
        width: 10px; height: 10px;
        border-radius: 50%; flex-shrink: 0;
    }
    .hw-vuln-tier { font-size: 16px; font-weight: 600; }
    .hw-vuln-sep {
        width: 1px; background: #e5e7eb;
        align-self: stretch; flex-shrink: 0; margin: 0;
    }
    .hw-vuln-text { font-size: 15px; color: #374151; line-height: 1.5; }
    .hw-meta {
        padding: 10px 24px;
        display: flex; align-items: center; gap: 24px;
    }
    .hw-meta-item { display: flex; flex-direction: column; gap: 2px; }
    .hw-meta-item:first-child { flex: 0 0 190px; }
    .hw-meta-label {
        font-size: 12px; color: #4b5563;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .hw-meta-value { font-size: 16px; font-weight: 600; color: #111827; }
    .hw-meta-div { width: 1px; height: 28px; background: #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True
)

# === App Header ===
st.title("🌍 HeatWatch — Climate's Human Cost")
st.markdown("""
<div style='font-size: 1.05rem; line-height: 1.6; margin-bottom: 14px;'>
Heat-related hospital admissions are the human fingerprint of climate change. This application 
visualizes historical country vulnerabilities on an interactive dashboard and uses a machine learning model to predict future health system capacity stresses.
</div>
""", unsafe_allow_html=True)

# === Global Configurations ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_MODEL_FILE = os.path.join(BASE_DIR, 'streamlit_model_stage1_classifier.pkl')
STAGE2_MODEL_FILE = os.path.join(BASE_DIR, 'streamlit_model_stage2_lightgbm.pkl')
CLUSTERS_FILE = os.path.join(BASE_DIR, 'country_clusters.csv')
MONTHLY_BASELINES_FILE = os.path.join(BASE_DIR, 'country_monthly_baselines_2025.csv')
DATASET_2025_FILE = os.path.join(BASE_DIR, 'dataset_2025.csv')

# Mappings for automatics
COUNTRY_DATA = {
    'United States': {'region': 'North America', 'lat': 37.09, 'lon': -95.71},
    'India': {'region': 'South Asia', 'lat': 20.59, 'lon': 78.96},
    'China': {'region': 'East Asia', 'lat': 35.86, 'lon': 104.19},
    'Brazil': {'region': 'South America', 'lat': -14.23, 'lon': -51.92},
    'Nigeria': {'region': 'Africa', 'lat': 9.08, 'lon': 8.67},
    'Germany': {'region': 'Europe', 'lat': 51.16, 'lon': 10.45},
    'Japan': {'region': 'East Asia', 'lat': 36.2, 'lon': 138.25},
    'United Kingdom': {'region': 'Europe', 'lat': 55.37, 'lon': -3.43},
    'France': {'region': 'Europe', 'lat': 46.22, 'lon': 2.21},
    'Australia': {'region': 'Oceania', 'lat': -25.27, 'lon': 133.77},
    'Kenya': {'region': 'Africa', 'lat': -0.02, 'lon': 37.9},
    'Mexico': {'region': 'North America', 'lat': 23.63, 'lon': -102.55},
    'Indonesia': {'region': 'Southeast Asia', 'lat': -0.78, 'lon': 113.92},
    'Pakistan': {'region': 'South Asia', 'lat': 30.37, 'lon': 69.34},
    'Bangladesh': {'region': 'South Asia', 'lat': 23.68, 'lon': 90.35},
    'Egypt': {'region': 'Africa', 'lat': 26.82, 'lon': 30.8},
    'South Africa': {'region': 'Africa', 'lat': -30.55, 'lon': 22.93},
    'Canada': {'region': 'North America', 'lat': 56.13, 'lon': -106.34},
    'Spain': {'region': 'Europe', 'lat': 40.46, 'lon': -3.74},
    'Italy': {'region': 'Europe', 'lat': 41.87, 'lon': 12.56},
    'Thailand': {'region': 'Southeast Asia', 'lat': 15.87, 'lon': 100.99},
    'Philippines': {'region': 'Southeast Asia', 'lat': 12.87, 'lon': 121.77},
    'Vietnam': {'region': 'Southeast Asia', 'lat': 14.05, 'lon': 108.27},
    'Argentina': {'region': 'South America', 'lat': -38.41, 'lon': -63.61},
    'Colombia': {'region': 'South America', 'lat': 4.57, 'lon': -74.29}
}

COUNTRY_TO_ISO3 = {
    'United States': 'USA',
    'India': 'IND',
    'China': 'CHN',
    'Brazil': 'BRA',
    'Nigeria': 'NGA',
    'Germany': 'DEU',
    'Japan': 'JPN',
    'United Kingdom': 'GBR',
    'France': 'FRA',
    'Australia': 'AUS',
    'Kenya': 'KEN',
    'Mexico': 'MEX',
    'Indonesia': 'IDN',
    'Pakistan': 'PAK',
    'Bangladesh': 'BGD',
    'Egypt': 'EGY',
    'South Africa': 'ZAF',
    'Canada': 'CAN',
    'Spain': 'ESP',
    'Italy': 'ITA',
    'Thailand': 'THA',
    'Philippines': 'PHL',
    'Vietnam': 'VNM',
    'Argentina': 'ARG',
    'Colombia': 'COL'
}

@st.cache_data
def load_clusters():
    df = pd.read_csv(CLUSTERS_FILE)
    df['risk_tier'] = df['risk_tier'].replace({
        'High Exposure - Plateauing': 'Sustained Pressure',
        'Stable & Buffered': 'Manageable Demand'
    })
    return df

@st.cache_data
def load_monthly_baselines():
    return pd.read_csv(MONTHLY_BASELINES_FILE)

@st.cache_data
def load_dataset_2025():
    return pd.read_csv(DATASET_2025_FILE)

@st.cache_resource
def load_models():
    stage1 = joblib.load(STAGE1_MODEL_FILE)
    stage2 = joblib.load(STAGE2_MODEL_FILE)
    return stage1, stage2

try:
    clusters_df = load_clusters()
    monthly_df = load_monthly_baselines()
    dataset_2025_df = load_dataset_2025()
except Exception as e:
    st.error(f"Error loading historical/baseline data. Ensure {CLUSTERS_FILE}, {MONTHLY_BASELINES_FILE}, and {DATASET_2025_FILE} are present.")
    st.stop()

# Set up tabs for separation of concerns
tab1, tab2 = st.tabs(["📊 Historical Dashboard", "🔮 Predictor (Weekly)"])

with tab1:
    st.header("Global Vulnerability & Risk Tiers")
    st.markdown("""
    <div style='font-size: 1.05rem; line-height: 1.6;'>
    Rising temperatures are an unambiguous phenomenon — but their health impact is unevenly distributed and vastly shaped by local vulnerability. For example, the data shows Pakistan's escalating admission trends occurring against a backdrop of rising heat anomalies, illustrating what happens when climate exposure meets low adaptive capacity.
    <br><br>
    By clustering countries based on historical extreme heat events, local admission trajectories, and systemic healthcare capacity, this analysis isolates three distinct risk profiles:
    <ul>
        <li><strong>🔴 Rising Burden</strong>: The burden of extreme heat events is accelerating while the local healthcare buffer remains comparatively thin.</li>
        <li><strong>🟡 Sustained Pressure</strong>: Countries with high climate exposure but whose relative admissions trends are stable against their own historical baseline.</li>
        <li><strong>🔵 Manageable Demand</strong>: Healthcare systems possess the resources and sheer capacity to effectively manage and absorb extreme events when they occur.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualizing Countries on a Map By Risk Tier
    map_df = clusters_df.copy()
    map_df['iso_alpha'] = map_df['country'].map(COUNTRY_TO_ISO3)

    fig = px.choropleth(
        map_df,
        locations='iso_alpha',
        locationmode='ISO-3',
        color='risk_tier',
        hover_name='country',
        hover_data={
            'iso_alpha': False,
            'mean_admissions': ':.1f',
            'local_p90_admissions': ':.1f',
            'healthcare_access': True,
            'high_event_slope_local': False
        },
        projection="natural earth",
        title="Heat-related Admissions: Country Risk Tiers",
        color_discrete_map={
            'Rising Burden': '#ef4444', 
            'Sustained Pressure': '#f59e0b', 
            'Manageable Demand': '#3b82f6'
        }
    )
    
    # Enhance map styling to be professional and similar to Carthage Positron (light minimal map)
    fig.update_geos(
        showcountries=True, countrycolor="#d1d5db", countrywidth=0.5,
        showcoastlines=True, coastlinecolor="#d1d5db", coastlinewidth=0.5,
        showland=True, landcolor="#f3f4f6",
        showocean=True, oceancolor="#ffffff",
        showlakes=False,
        showframe=False,
        resolution=50,
        bgcolor='rgba(0,0,0,0)'
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="#d1d5db")
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=22),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5,
            title=None,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=14)
        )
    )
    st.plotly_chart(fig, width="stretch")
    
    st.markdown("### Country Risk Profiles Data")
    
    # Rename columns for the display dataframe
    display_df = clusters_df[['country', 'risk_tier', 'admission_slope', 'high_event_slope_local', 'healthcare_access', 'income_level']].sort_values('risk_tier').rename(columns={
        'country': 'Country',
        'risk_tier': 'Risk Tier',
        'admission_slope': 'Heat Hospitalisation Trend',
        'high_event_slope_local': 'High Burden Trend',
        'healthcare_access': 'Healthcare Access',
        'income_level': 'Income Level'
    })
    
    # Format specified columns to 2 decimal places and cast to string to enforce left alignment
    cols_to_format = ['Heat Hospitalisation Trend', 'High Burden Trend', 'Healthcare Access']
    for col in cols_to_format:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

    # Dependency-free table with sticky header + internal scroll and explicit high-contrast headers.
    st.markdown(
        """
        <style>
        .risk-table-wrap {
            max-height: 460px;
            overflow-y: auto;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            background: #ffffff;
            margin-bottom: 25px;
        }
        .risk-table {
            width: 100%;
            border-collapse: collapse;
            font-family: var(--font, "Source Sans Pro", sans-serif);
            font-size: 0.85rem;
            line-height: 1.5;
        }
        .risk-table thead th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #e5e7eb;
            color: #111827;
            font-weight: 500;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid #9ca3af;
        }
        .risk-table tbody td {
            color: #1f2937;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid #e5e7eb;
            background: #ffffff;
        }
        .risk-table thead th,
        .risk-table tbody td {
            border-right: 1px solid #e5e7eb;
        }
        .risk-table thead th:last-child,
        .risk-table tbody td:last-child {
            border-right: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='risk-table-wrap'>{display_df.to_html(index=False, classes='risk-table')}</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("""
    <div style='background-color: #e1f5fe; padding: 15px; border-radius: 5px; font-size: 0.9em; color: #0c4a6e; margin-bottom: 15px;'>
    <strong>💡 Understanding the Trends:</strong>
    <ul style='margin-top: 5px; margin-bottom: 0px;'>
        <li><strong>Heat Hospitalisation Trend</strong> (Admission Slope): Measures if the overall number of hospital visits for heat-related illnesses is trending up (positive) or down (negative) over time.</li>
        <li><strong>High Burden Trend</strong> (Local High Event Slope): Measures whether historically high burden events—defined as weeks where heat-related admissions exceed the 90th percentile for <em>that specific country</em>—are becoming more frequent (positive) or stabilizing (negative).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("""
    **Note on Underlying Data:**  
    1. **Core Dataset:** Built on the 2015–2025 Kaggle Global Climate-Health Impact Tracker (25 Countries), spatially merged with high-resolution environmental metrics from the NASA POWER database.  
    2. **Risk Profiling:** Historical risk tiers are specifically generated using data from the 2015–2023 analytical period.
    """)
with tab2:
    try:
        stage1, stage2 = load_models()
    except Exception as e:
        st.error(f"Error loading Hurdle Model files: {e}")
        st.stop()
        
    st.header("🌡️ Heat Event & Health System Capacity Forecast")
    st.info("""
    **Methodology**: To properly anticipate risk, this predictor leverages a combined model to process complex meteorological scenarios, while considering country-specific epidemiological and socioeconomic factors:
    - First, the model determines the biological & environmental probability of a heat event taking place at all.
    - Then, if an event does transpire, it assesses systemic capabilities to gauge the precise weekly metric of heat-related hospital admissions.
        """)

    st.markdown("---")
    st.subheader("📝 Dynamic Simulation Scenarios")
    
    # === Core Selectors ===
    st.markdown("##### 1. Select the target location and timeframe:")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        country_name = st.selectbox("Country", list(COUNTRY_DATA.keys()))
    with col_b:
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
                       7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        month = st.selectbox("Month", list(month_names.keys()), index=6, format_func=lambda x: month_names[x])
    with col_c:
        year = st.number_input("Year", value=2025)
    
    # Filter the monthly baseline DataFrame dynamically
    try:
        baseline_row = monthly_df[(monthly_df['country_name'] == country_name) & (monthly_df['month'] == month)].iloc[0]
    except IndexError:
        st.warning("Could not find a baseline for this country. Defaulting to averages.")
        baseline_row = monthly_df.mean(numeric_only=True) # Fallback

    # Display dynamic context
    region = COUNTRY_DATA[country_name]['region']
    latitude = COUNTRY_DATA[country_name]['lat']
    longitude = COUNTRY_DATA[country_name]['lon']
    # The models require income_level as a category, but it's not strictly numeric. 
    # Use the clusters lookup or a fallback for string values like 'income_level'.
    cluster_info = clusters_df[clusters_df['country'] == country_name]
    income_level = cluster_info['income_level'].values[0] if not cluster_info.empty else "Upper-Middle"

    st.info(f"📍 **Auto-Mapped Context for {country_name} in {month_names[month]}:** \n\nWe have automatically filled all sliders and hidden variables with their **2025 historical monthly averages** for this specific region. You can now effortlessly simulate climate shocks (such as bumping the temperature anomaly up).")
    
    st.markdown("##### 2. Tweak key climate variables to simulate environmental shocks:")
    # === Form Inputs Layout (Primary Dynamic Modifiers) ===
    col1, col2 = st.columns(2)
    with col1:
        temperature_celsius = st.slider("Average Temp (°C)", min_value=-30.0, max_value=50.0, value=float(baseline_row['temperature_celsius']))
        max_hourly_temp = st.slider("Max Hourly Temp (°C)", min_value=-30.0, max_value=50.0, value=float(baseline_row['max_hourly_temp']))
        avg_humidity = st.slider("Average Humidity (%)", min_value=0.0, max_value=100.0, value=float(baseline_row['avg_humidity']))
    with col2:
        temp_anomaly_celsius = st.slider("Temp Anomaly (+/- °C Shift from Norm)", min_value=-10.0, max_value=10.0, value=float(baseline_row['temp_anomaly_celsius']))
        avg_aerosol_depth = st.slider("Average Aerosol Depth", min_value=0.0, max_value=5.0, value=float(baseline_row.get('avg_aerosol_depth', 0.1)))
        air_quality_index = st.slider("Air Quality Index (AQI)", min_value=0.0, max_value=500.0, value=float(baseline_row.get('air_quality_index', 40.0)))

    # === Hidden Constants Layout ===
    st.markdown("---")
    with st.expander("📝 Advanced Predictive Variables (Auto-filled from 2025)", expanded=False):
        st.markdown("These constants have been mathematically pre-filled specifically for your chosen country and month. You can override them if evaluating infrastructure or development policies.")

        p1, p2 = st.columns(2)
        with p1:
            population_millions = st.number_input("Population (Millions)", value=float(baseline_row.get('population_millions', 50.0)))
        with p2:
            gdp_per_capita_usd = st.number_input("GDP Per Capita ($)", value=float(baseline_row.get('gdp_per_capita_usd', 15000.0)))
        
        st.markdown("##### Climate & Environmental Drivers")
        c1, c2 = st.columns(2)
        with c1:
            precipitation_mm = st.number_input("Precipitation (mm)", value=float(baseline_row.get('precipitation_mm', 0.0)))
            extreme_heat_hours = st.number_input("Extreme Heat Hours Recorded", value=float(baseline_row.get('extreme_heat_hours', 10.0)))
            extreme_weather_events = st.number_input("Count of Extreme Weather Events", value=int(round(baseline_row.get('extreme_weather_events', 0))))
            flood_indicator = st.selectbox("Actively Flooding?", [0, 1], index=int(round(baseline_row.get('flood_indicator', 0))), format_func=lambda x: "Yes" if x == 1 else "No")
        with c2:
            pm25_ugm3 = st.number_input("PM2.5 Pollution Index (ug/m3)", value=float(baseline_row['pm25_ugm3']))
            heat_wave_days = st.number_input("Heat Wave Days (Past Week)", value=int(baseline_row['heat_wave_days']))
            max_aerosol_depth = st.number_input("Max Aerosol Depth", value=float(baseline_row.get('max_aerosol_depth', 0.2)))
            drought_indicator = st.selectbox("Actively in Drought?", [0, 1], index=int(round(baseline_row.get('drought_indicator', 0))), format_func=lambda x: "Yes" if x == 1 else "No")

        # Retain model tracking variable in backend, removed from user view
        avg_weekly_temp = float(baseline_row.get('avg_weekly_temp', 28.0))

        st.markdown("##### Socio-Economic Factors")
        s1, s2 = st.columns(2)
        with s1:
            healthcare_access_index = st.number_input("Healthcare Access Index (0-100)", value=float(baseline_row.get('healthcare_access_index', 80.0)))
        with s2:
            food_security_index = st.number_input("Food Security Index (0-100)", value=float(baseline_row.get('food_security_index', 70.0)))

        st.markdown("##### Population Health Metrics")
        e1, e2 = st.columns(2)
        with e1:
            mental_health_index = st.number_input("Mental Health Index (0-100)", value=float(baseline_row.get('mental_health_index', 75.0)))
            respiratory_disease_rate = st.number_input("Local Respiratory Disease Rate", value=float(baseline_row.get('respiratory_disease_rate', 50.0)))
            cardio_mortality_rate = st.number_input("Local Cardio Mortality Rate", value=float(baseline_row.get('cardio_mortality_rate', 30.0)))
        with e2:
            vector_disease_risk_score = st.number_input("Vector Disease Risk Score (0-100)", value=float(baseline_row.get('vector_disease_risk_score', 10.0)))
            waterborne_disease_incidents = st.number_input("Waterborne Disease Incidents Count", value=int(round(baseline_row.get('waterborne_disease_incidents', 5))))

    st.markdown("---")
    
    # === Inference Engine ===
    if st.button("🚀 Predict Weekly Heat-Related Admissions", width="stretch"):
        # Condense all the visual forms into mathematically strict dataset arrays
        input_dict = {
            'country_name': country_name,
            'region': region,
            'income_level': income_level,
            'latitude': latitude,
            'longitude': longitude,
            'population_millions': population_millions,
            'gdp_per_capita_usd': gdp_per_capita_usd,
            'temperature_celsius': temperature_celsius,
            'precipitation_mm': precipitation_mm,
            'avg_humidity': avg_humidity,
            'heat_wave_days': heat_wave_days,
            'extreme_heat_hours': extreme_heat_hours,
            'temp_anomaly_celsius': temp_anomaly_celsius,
            'avg_weekly_temp': avg_weekly_temp,
            'max_hourly_temp': max_hourly_temp,
            'drought_indicator': drought_indicator,
            'flood_indicator': flood_indicator,
            'extreme_weather_events': extreme_weather_events,
            'pm25_ugm3': pm25_ugm3,
            'air_quality_index': air_quality_index,
            'avg_aerosol_depth': avg_aerosol_depth,
            'max_aerosol_depth': max_aerosol_depth,
            'healthcare_access_index': healthcare_access_index,
            'food_security_index': food_security_index,
            'year': year,
            'month': month,
            'respiratory_disease_rate': respiratory_disease_rate,
            'cardio_mortality_rate': cardio_mortality_rate,
            'vector_disease_risk_score': vector_disease_risk_score,
            'waterborne_disease_incidents': waterborne_disease_incidents,
            'mental_health_index': mental_health_index
        }
        
        input_df = pd.DataFrame([input_dict])

        # Align columns to model expectations to avoid feature-name mismatch warnings.
        stage1_input_df = input_df
        if hasattr(stage1, 'feature_name_') and stage1.feature_name_:
            stage1_input_df = input_df.reindex(columns=list(stage1.feature_name_), fill_value=0)
        elif hasattr(stage1, 'feature_names_in_'):
            stage1_input_df = input_df.reindex(columns=list(stage1.feature_names_in_), fill_value=0)

        stage2_input_df = input_df
        if hasattr(stage2, 'feature_name_') and stage2.feature_name_:
            stage2_input_df = input_df.reindex(columns=list(stage2.feature_name_), fill_value=0)
        elif hasattr(stage2, 'feature_names_in_'):
            stage2_input_df = input_df.reindex(columns=list(stage2.feature_names_in_), fill_value=0)
        
        with st.spinner('Validating environments against global patterns...'):
            # Hurdle Stage 1
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names",
                    category=UserWarning
                )
                stage1_pred = stage1.predict(stage1_input_df)[0]
            
            try:
                # Some classifiers support probability for better contextual output
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="X does not have valid feature names",
                        category=UserWarning
                    )
                    stage1_proba = stage1.predict_proba(stage1_input_df)[0][1] * 100
                proba_str = f"({stage1_proba:.1f}% Likelihood of Heat Admission Event this Week)"
            except Exception:
                proba_str = ""
        
        # ── Vulnerability tier lookup ──────────────────────────────────────
        tier_str = cluster_info['risk_tier'].values[0] if not cluster_info.empty else "Unknown Risk Tier"

        # Per-tier visual config: dot colour, band background, text colour, fixed context sentence
        TIER_CONFIG = {
            'Rising Burden': {
                'dot':        '#ef4444',
                'band_bg':    '#fef2f2',
                'text_color': '#991b1b',
                'context': (
                    "Heat burden is accelerating against a thin healthcare buffer — "
                    "similar admission volumes carry greater systemic risk here than in more resilient systems."
                ),
            },
            'Sustained Pressure': {
                'dot':        '#f59e0b',
                'band_bg':    '#fffbeb',
                'text_color': '#92400e',
                'context': (
                    "Climate exposure is persistently high, though admission trends remain stable "
                    "relative to historical baseline — continued monitoring is warranted even at lower volumes."
                ),
            },
            'Manageable Demand': {
                'dot':        '#3b82f6',
                'band_bg':    '#eff6ff',
                'text_color': '#1e40af',
                'context': (
                    "This healthcare system has the resources and capacity to effectively absorb extreme heat events "
                    "— risk to systemic function remains low even when events do occur."
                ),
            },
        }
        tc = TIER_CONFIG.get(tier_str, {
            'dot': '#9ca3af', 'band_bg': '#f9fafb',
            'text_color': '#374151', 'context': ''
        })

        # ── Historical reference points ────────────────────────────────────
        local_mean = float(cluster_info['mean_admissions'].values[0]) if not cluster_info.empty else 0.0
        local_p90  = float(cluster_info['local_p90_admissions'].values[0]) if not cluster_info.empty else 0.0

        # ── Stage 2 (regression) only if event predicted ───────────────────
        if stage1_pred == 1:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names",
                    category=UserWarning
                )
                stage2_pred = stage2.predict(stage2_input_df)[0]
            predicted_admissions = max(0, stage2_pred)
            heat_stress_classification = "Local capacity stress detected"

            # Bar colour determined independently by admission position — decoupled from tier colour
            if predicted_admissions >= local_p90:
                bar_fill         = '#ef4444'
                bar_dot          = '#991b1b'
                bar_status_color = '#991b1b'
                bar_status_text  = "At or above historical 90th percentile — critical range"
            elif predicted_admissions >= local_mean:
                bar_fill         = '#f59e0b'
                bar_dot          = '#92400e'
                bar_status_color = '#92400e'
                bar_status_text  = "Above historical average — approaching elevated range"
            else:
                bar_fill         = '#16a34a'
                bar_dot          = '#14532d'
                bar_status_color = '#14532d'
                bar_status_text  = "Below historical average — within normal range"

            pct_fill = min((predicted_admissions / local_p90) * 100, 100) if local_p90 > 0 else 0
            pct_mean = min((local_mean / local_p90) * 100, 100) if local_p90 > 0 else 50

            right_zone_html = (
                '<div class="hw-bar-zone">'
                '<div class="hw-bar-title">Projected admissions relative to historical range</div>'
                '<div class="hw-bar-wrap">'
                f'<div class="hw-bar-fill" style="width:{pct_fill:.1f}%; background:{bar_fill};"></div>'
                f'<div class="hw-bar-marker" style="left:{pct_mean:.1f}%;"></div>'
                f'<div class="hw-bar-dot" style="left:{pct_fill:.1f}%; background:{bar_dot};"></div>'
                '</div>'
                '<div class="hw-bar-labels">'
                '<span>0</span>'
                f'<span>Average ({local_mean:.1f})</span>'
                f'<span>90th pct ({local_p90:.1f})</span>'
                '</div>'
                f'<div class="hw-bar-status" style="color:{bar_status_color};">{bar_status_text}</div>'
                '</div>'
            )

        else:
            predicted_admissions       = 0
            heat_stress_classification = "Baseline conditions expected"
            right_zone_html = (
                '<div class="hw-zero-callout">'
                '<div class="hw-zero-title">No heat stress event anticipated this week</div>'
                '<div class="hw-zero-sub">Environmental conditions are not consistent with a '
                'heat-related admission event. No projected impact on hospital capacity is expected.</div>'
                '</div>'
            )

        # ── Render unified result card ─────────────────────────────────────
        # Build the full HTML string in Python first to avoid Streamlit
        # escaping HTML tags that are interpolated inside an f-string.
        card_html = (
            '<div class="hw-card">'

            # Top row: hero number + bar or callout
            '<div class="hw-top">'
            '<div class="hw-hero">'
            f'<div class="hw-hero-number">{predicted_admissions:.1f}</div>'
            '<div class="hw-hero-label">estimated weekly<br>heat-related admissions</div>'
            '</div>'
            '<div class="hw-divider-v"></div>'
            + right_zone_html +
            '</div>'

            # Middle row: vulnerability tier band
            f'<div class="hw-vuln" style="background:{tc["band_bg"]};">'
            '<div class="hw-vuln-left">'
            f'<div class="hw-vuln-dot" style="background:{tc["dot"]};"></div>'
            f'<span class="hw-vuln-tier" style="color:{tc["text_color"]};">{tier_str}</span>'
            '</div>'
            '<div class="hw-vuln-sep"></div>'
            f'<div class="hw-vuln-text">{tc["context"]}</div>'
            '</div>'

            # Bottom row: metadata
            '<div class="hw-meta">'
            '<div class="hw-meta-item">'
            '<span class="hw-meta-label">Heat stress likelihood</span>'
            f'<span class="hw-meta-value">{stage1_proba:.1f}%</span>'
            '</div>'
            '<div class="hw-meta-div"></div>'
            '<div class="hw-meta-item">'
            '<span class="hw-meta-label">Heat stress classification</span>'
            f'<span class="hw-meta-value">{heat_stress_classification}</span>'
            '</div>'
            '<div class="hw-meta-div"></div>'
            '<div class="hw-meta-item">'
            '<span class="hw-meta-label">Country &amp; period</span>'
            f'<span class="hw-meta-value">{country_name} \u00b7 {month_names[month]} {int(year)}</span>'
            '</div>'
            '</div>'

            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)
            
        if year == 2025:
            st.markdown("---")
            if month in [11, 12]:
                st.info("📊 **Validation Data Note**: No test validation available, no monthly data.")
            else:
                try:
                    # Convert date to extract month if needed, but assuming dataset_2025.csv has 'date' or we can parse it
                    if 'month' not in dataset_2025_df.columns:
                        temp_df = dataset_2025_df.copy()
                        temp_df['date'] = pd.to_datetime(temp_df['date'])
                        temp_df['month'] = temp_df['date'].dt.month
                    else:
                        temp_df = dataset_2025_df
                    
                    actual_data = temp_df[(temp_df['country_name'] == country_name) & (temp_df['month'] == month)]
                    
                    if not actual_data.empty:
                        actual_avg = actual_data['heat_related_admissions'].mean()
                        st.info(f"📊 **Actual 2025 Average**: **{actual_avg:.1f}** Weekly Admissions for this region in {month_names[month]}.")
                    else:
                        st.info(f"📊 **Actual 2025 Average**: Data not found for this selection in the validation dataset.")
                except Exception as e:
                    st.warning("Could not calculate actual averages for 2025.")

    st.markdown("---")
    st.caption("""
    **Note on Modelling:**  
    1. **Prediction Engine:** First Step (LightGBM Classifier) handles event likelihood; Second Step (LightGBM Regressor) estimates admission volume.  
    2. **Vulnerability Tiers:** Profiled in the historical dashboard based on KMeans Clustering.
    """)
