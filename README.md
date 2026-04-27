# HeatWatch: Climate Health Vulnerabilities

## Problem Statement
Climate change is increasing the intensity and frequency of extreme heat events, but the health burden is not distributed equally across countries. Public health teams need practical tools to answer two questions early:
1. Which countries are structurally more vulnerable to heat-related hospital demand?
2. Given current climate conditions, what level of weekly heat-related admissions should health systems expect?

This project builds a climate-health decision support workflow that combines historical vulnerability profiling with a predictive proof of concept app (HeatWatch).
It is designed to communicate clearly to non-technical audiences while remaining method-transparent for technical reviewers.

## Project Objective
Build an end-to-end pipeline that:
- Profiles country-level vulnerability using historical climate and health trends.
- Predicts weekly heat-related admissions using a two-stage machine learning approach.
- Provides an interactive Streamlit app for scenario simulation and early planning.

## Why Climate Variables Alone Are Not Sufficient
Early experiments and slide-level analysis showed that climate exposure signals (for example temperature and anomaly) are necessary, but not sufficient, to explain heat-related admissions across countries.

Key reason:
- The same climate shock can produce very different admission burden depending on healthcare access, baseline population health, air quality burden, and socioeconomic resilience.
- This creates a vulnerability paradox: some countries with lower adaptive capacity experience similarly high (or higher) admissions than countries with stronger systems.

Project implication:
- The final modeling strategy uses climate variables together with health-system and socioeconomic context, rather than a climate-only specification.
- Risk interpretation is country-relative (historical baselines and local high-burden thresholds), not only global absolute thresholds.

## Data Sources
### 1. Kaggle Dataset (Core Health + Climate Panel)
Source file: `Kaggle Files/global_climate_health_impact_tracker_2015_2025.csv`

Brief overview:
- 14,100 records, weekly frequency, Jan 2015 to Oct 2025.
- 25 countries across multiple regions and income levels.
- Includes climate indicators, air quality, health outcomes, and socioeconomic context.
- Key target used in this app: `heat_related_admissions` (rate per 100,000 population).

The Kaggle README with full field descriptions is available in:
- `Kaggle Files/README_climate_health.md`

### 2. NASA POWER API (High-Resolution Environmental Enrichment)
Source workflow: `Phase 2/Phase 2 Heat Related_Kaggle_Nasa Merge.ipynb`

Brief overview:
- Hourly NASA POWER variables were retrieved for each location and year.
- Key parameters used: `T2M` (temperature), `RH2M` (humidity), `AOD_55` (aerosol optical depth).
- Hourly data was engineered into weekly features (for example average weekly temperature, max hourly temperature, extreme heat hours, aerosol summaries) and joined to Kaggle records by date and coordinates.

## Approach
### 0. EDA Highlights (From Slide Charts)
Exploratory analysis was used to validate the problem framing before final model design. The chart workflow highlighted four patterns:
- Rising anomalies over 2015-2024, indicating a shifting heat baseline.
- More frequent high-burden admission weeks after 2020 versus earlier years.
- A vulnerability paradox: weaker healthcare access is associated with elevated admissions for several countries, despite lower system capacity.
- Multi-systemic coupling: heat-related admissions co-move with air quality and broader health stress indicators, supporting a multi-factor modeling approach.

These EDA findings directly motivated a two-stage hurdle model and the inclusion of non-climate predictors.

### 1. Historical Risk Profiling
Using historical patterns, countries were grouped into risk tiers with KMeans clustering, based on:
- Admission trend slope
- High-event slope relative to local 90th percentile burden
- Climate anomaly trajectory
- Health system capacity proxies (for example healthcare access and GDP per capita)

This output supports the app's historical dashboard and risk tier map.

### 2. Two-Stage Hurdle Prediction Pipeline
Implemented in `climate_models.ipynb`:
- Stage 1 (Classifier): predicts whether a heat-admission event occurs (zero vs non-zero week).
- Stage 2 (Regressor): predicts admission magnitude when an event is expected.

Model candidates were compared, and serialized models were exported for app inference.

### 2a. Model Selection Snapshot (Why LightGBM for Deployment)
Multiple Stage 2 regressors were benchmarked on a 2015-2023 train and 2024 test split. While CatBoost achieved the highest combined test R2, LightGBM was selected for the deployed app because it provided a stronger train-test balance with lower overfit signals.

Key comparison highlights from `New Phase Apr'26/models_comparison_v2.csv`:
- CatBoost: Combined Train R2 = 0.9323, Combined Test R2 = 0.8263, Discrepancy = 0.1061
- LightGBM: Combined Train R2 = 0.9094, Combined Test R2 = 0.8247, Discrepancy = 0.0847

Why this mattered:
- Test performance was nearly equivalent (difference in Combined Test R2 is small).
- LightGBM had lower train-test discrepancy on the combined hurdle output.
- LightGBM also showed lower Stage 2 train-test discrepancy than CatBoost (0.1600 vs 0.2018), supporting more stable generalization.

### 3. Streamlit Proof of Concept App
Final app implementation:
- `heatwatch_app.py`

App structure:
- Historical Dashboard tab: global risk tiers and country profile table.
- Predictor tab: weekly admission forecasting with scenario controls.

A monthly baseline table (`country_monthly_baselines_2025.csv`) pre-fills country-month inputs, then users can stress-test selected variables.

## One-Line App Usage
Select a country and month, let the app auto-load 2025 monthly baseline values, manually adjust key climate sliders, then click predict to simulate weekly heat-related admissions.

## Key Outputs in This Repository
- Trained model artifacts used by Streamlit:
  - `streamlit_model_stage1_classifier.pkl`
  - `streamlit_model_stage2_lightgbm.pkl`
- Historical vulnerability outputs:
  - `country_clusters.csv`
- Baseline inputs for app auto-fill behavior:
  - `country_monthly_baselines_2025.csv`

## Future Work
1. Upgrade to fully out-of-time validation with rolling/temporal backtesting across multiple years.
2. Expand geographic coverage and test transferability for data-scarce regions.
3. Add uncertainty intervals and calibrated risk bands to prediction outputs.
4. Integrate near real-time weather feeds so simulation defaults are not limited to 2025 baselines.
5. Extend outcomes beyond heat-related admissions to respiratory and cardiovascular burden forecasts.
6. Add policy simulation modules (for example healthcare capacity improvements) to quantify adaptation impact.

## Notes
This project is currently a proof of concept intended for exploratory climate-health risk analysis and operational scenario testing, not clinical diagnosis or medical triage.
