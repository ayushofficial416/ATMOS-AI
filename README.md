# Atmos AI — Storm & Heavy Rainfall Prediction

A Python-only Streamlit web application with live Open-Meteo forecast data, interactive charts, storm/heavy-rain Random Forest risk estimates, a custom prediction lab, and CSV-based model training.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install packages:

   `pip install -r requirements.txt`

4. Start the website:

   `streamlit run app.py`

5. Open the local URL shown in the terminal (usually http://localhost:8501) in Google Chrome.

## Dataset format for training your own model

CSV must contain these columns:

- precipitation_probability
- precipitation
- wind_speed_10m
- wind_gusts_10m
- pressure_msl
- relative_humidity_2m
- cloud_cover
- storm_label (0 or 1)
- rain_label (0 or 1)

`storm_label` and `rain_label` are historical observed outcomes, not predictions. Use trusted observations and careful time-based validation for serious work.

## Important model limitation

Without an uploaded historical labelled dataset, the app uses a synthetic educational dataset to demonstrate the ML pipeline. Its probabilities and validation scores are not scientifically calibrated forecasts. Live weather inputs come from Open-Meteo. Do not use this prototype as an official warning or emergency system.

## Hosting online

For a public URL, deploy this folder to Streamlit Community Cloud (requires a GitHub repository), or another Python-compatible hosting service. The app itself is Python; the Streamlit framework renders the browser interface.


## Sarvam AI assistant setup

1. Create an API key in the [Sarvam AI dashboard](https://dashboard.sarvam.ai/).
2. In the project folder, copy `.env.example` to `.env`.
3. Open `.env` and replace the placeholder with your key: `SARVAM_API_KEY=your_real_key`.
4. Install updated dependencies with `pip install -r requirements.txt`.
5. Restart the app using `streamlit run app.py`.

The key is read server-side from the environment and is not displayed in the app. Do not commit `.env` or share it publicly. For Streamlit Community Cloud, add `SARVAM_API_KEY` in the app's Secrets settings instead of uploading a `.env` file. The assistant uses Sarvam's `sarvam-105b-conversations` chat model and receives the current location's forecast and the app's model risk estimates as context.
