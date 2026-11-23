
import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =========================================================
# ATMOS AI V2 | WEATHER INTELLIGENCE
# =========================================================

st.set_page_config(
    page_title="ATMOS AI | Weather Intelligence",
    page_icon="🌩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURES = [
    "precipitation_probability",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "pressure_msl",
    "relative_humidity_2m",
    "cloud_cover"
]

# =========================================================
# PREMIUM UI DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
    linear-gradient(rgba(3,10,23,.91),rgba(3,12,27,.96)),
    url('https://images.unsplash.com/photo-1534088568595-a066f410bcda');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: rgba(5,15,31,.97);
    border-right: 1px solid rgba(100,180,255,.15);
}

.hero {
    min-height: 300px;
    padding: 38px;
    border-radius: 25px;
    background:
    linear-gradient(90deg,rgba(3,13,30,.95),rgba(3,13,30,.3)),
    url('https://images.unsplash.com/photo-1605727216801-e27ce1d0cc28');
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(110,180,255,.25);
    box-shadow: 0 20px 60px rgba(0,0,0,.35);
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.hero-tag {
    color: #65c7ff;
    font-size: 12px;
    letter-spacing: 4px;
    font-weight: 800;
}

.hero h1 {
    color: white;
    font-size: clamp(38px,5vw,65px);
    font-weight: 900;
    letter-spacing: -3px;
    margin: 10px 0;
}

.hero p {
    color: #c3d7eb;
    font-size: 15px;
}

.panel {
    background: linear-gradient(145deg,
    rgba(20,43,70,.91),
    rgba(8,22,41,.94));
    border: 1px solid rgba(120,190,255,.17);
    border-radius: 19px;
    padding: 22px;
    box-shadow: 0 12px 35px rgba(0,0,0,.2);
}

.panel-title {
    color: #8ecfff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.big-value {
    font-size: 34px;
    font-weight: 900;
    color: white;
    margin-top: 10px;
}

.muted {
    color: #9cb3ca;
    font-size: 12px;
}

.section-title {
    color: #f2f8ff;
    font-size: 21px;
    font-weight: 800;
    margin: 25px 0 15px;
}

.stButton button {
    border-radius: 10px;
    border: 1px solid #278fd1;
    background: linear-gradient(110deg,#0878bd,#1656a5);
    color: white;
    font-weight: 700;
    min-height: 42px;
}

.stButton button:hover {
    border-color: #69d4ff;
    color: white;
    box-shadow: 0 0 20px rgba(40,160,255,.3);
}

div[data-testid="stMetric"] {
    background: rgba(13,37,63,.7);
    border: 1px solid rgba(130,190,255,.14);
    padding: 15px;
    border-radius: 14px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #b9d4ec;
}

hr {
    border-color: rgba(130,190,255,.15);
}

div[data-testid="stChatMessage"] {
    background: rgba(15,39,66,.85);
    border: 1px solid rgba(100,180,255,.12);
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# API FUNCTIONS
# =========================================================

@st.cache_data(ttl=1800)
def geocode(city):

    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        },
        timeout=20
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        return None

    result = results[0]

    return {
        "name": result["name"],
        "country": result.get("country", ""),
        "latitude": result["latitude"],
        "longitude": result["longitude"]
    }


@st.cache_data(ttl=600)
def get_weather(lat, lon):

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "precipitation,rain,showers,weather_code,"
                "wind_speed_10m,wind_gusts_10m,pressure_msl"
            ),
            "hourly": (
                "temperature_2m,relative_humidity_2m,"
                "precipitation_probability,precipitation,"
                "rain,showers,wind_speed_10m,"
                "wind_gusts_10m,pressure_msl,cloud_cover"
            ),
            "daily": (
                "weather_code,temperature_2m_max,"
                "temperature_2m_min,precipitation_sum,"
                "precipitation_probability_max"
            ),
            "forecast_days": 7,
            "timezone": "auto"
        },
        timeout=25
    )

    response.raise_for_status()
    return response.json()


# =========================================================
# AI / ML MODEL
# =========================================================

@st.cache_resource
def train_demo_models():

    rng = np.random.default_rng(42)
    n = 12000

    data = pd.DataFrame({
        "precipitation_probability": rng.uniform(0, 100, n),
        "precipitation": rng.gamma(1.1, 3.2, n),
        "wind_speed_10m": rng.weibull(1.8, n) * 23,
        "wind_gusts_10m": rng.weibull(1.7, n) * 34,
        "pressure_msl": rng.normal(1008, 9, n),
        "relative_humidity_2m": rng.uniform(25, 100, n),
        "cloud_cover": rng.uniform(0, 100, n)
    })

    storm_score = (
        data["wind_gusts_10m"] / 75
        + data["wind_speed_10m"] / 55
        + data["precipitation_probability"] / 150
        + (1010 - data["pressure_msl"]) / 35
        + data["cloud_cover"] / 260
        + rng.normal(0, .17, n)
    )

    rain_score = (
        data["precipitation_probability"] / 65
        + data["precipitation"] / 18
        + data["relative_humidity_2m"] / 180
        + data["cloud_cover"] / 180
        + rng.normal(0, .18, n)
    )

    data["storm_label"] = (storm_score > 1.12).astype(int)
    data["rain_label"] = (rain_score > 1.65).astype(int)

    models = {}

    for label in ["storm_label", "rain_label"]:

        X_train, X_test, y_train, y_test = train_test_split(
            data[FEATURES],
            data[label],
            test_size=.2,
            random_state=42,
            stratify=data[label]
        )

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("forest", RandomForestClassifier(
                n_estimators=150,
                max_depth=11,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])

        model.fit(X_train, y_train)

        accuracy = accuracy_score(
            y_test,
            model.predict(X_test)
        )

        models[label] = (model, accuracy)

    return models


def risk(probability):

    if probability >= .75:
        return "HIGH", "🔴"

    elif probability >= .45:
        return "MODERATE", "🟠"

    return "LOW", "🟢"


def weather_icon(code):

    if code == 0:
        return "☀️"

    if code in [1, 2]:
        return "🌤️"

    if code == 3:
        return "☁️"

    if code in [45, 48]:
        return "🌫️"

    if code in [51, 53, 55, 61, 63, 65]:
        return "🌧️"

    if code in [71, 73, 75, 77]:
        return "❄️"

    if code in [80, 81, 82]:
        return "🌦️"

    if code in [95, 96, 99]:
        return "⛈️"

    return "🌡️"


# =========================================================
# SARVAM AI
# =========================================================

def get_sarvam_key():

    # Set SARVAM_API_KEY in Render Environment Variables.
    # Also supports Streamlit secrets if already configured.
    key = os.getenv("SARVAM_API_KEY", "")

    if not key:
        try:
            key = st.secrets["SARVAM_API_KEY"]
        except Exception:
            key = ""

    return key


def ask_sarvam(question, weather_context, history):

    api_key = get_sarvam_key()

    if not api_key:
        return (
            "Sarvam AI is not configured yet. "
            "Add SARVAM_API_KEY in your Render service's "
            "Environment settings and redeploy."
        )

    system_prompt = f"""
You are ATMOS AI, an expert weather intelligence assistant.

Your responsibilities:
1. Explain weather forecasts.
2. Explain storm and heavy rainfall risks.
3. Help users understand weather conditions.
4. Support English, Hindi and Marathi.
5. Give clear and concise safety guidance.
6. Never invent weather measurements.
7. Distinguish model estimates from official warnings.

CURRENT WEATHER DATA:
{weather_context}

Important:
The machine learning model is an experimental educational
model. Its probabilities are not official meteorological
probabilities. Direct users to official IMD warnings
for real emergency decisions.
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    for item in history[-10:]:
        messages.append(item)

    messages.append({
        "role": "user",
        "content": question
    })

    try:

        response = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={
                "api-subscription-key": api_key,
                "Content-Type": "application/json"
            },
            json={
                "model": "sarvam-105b-conversations",
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 1000
            },
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        return result["choices"][0]["message"]["content"]

    except requests.RequestException as error:

        return f"Sarvam API error: {error}"

    except (KeyError, IndexError, ValueError):

        return "Sarvam AI returned an unexpected response."


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">
        ADVANCED METEOROLOGICAL INTELLIGENCE SYSTEM
    </div>
    <h1>ATMOS AI</h1>
    <p>
        Predict the storm. Understand the rainfall.
        Prepare for the unexpected.
    </p>
    <div style="margin-top:18px;color:#8ed5ff;
    font-size:12px;letter-spacing:2px;">
        LIVE WEATHER • MACHINE LEARNING • SARVAM AI
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🌩️ ATMOS AI")

    st.caption("WEATHER CONTROL CENTRE")

    st.divider()

    city = st.text_input(
        "📍 Search location",
        value=st.session_state.get("city", "Pune")
    )

    refresh = st.button(
        "⟳ ANALYSE WEATHER",
        use_container_width=True
    )

    st.divider()

    st.markdown("### 🧠 AI / ML ENGINE")

    st.caption("Random Forest Classifier")

    st.markdown("### 🤖 SARVAM AI")

    st.caption("Multilingual weather assistant")

    st.markdown("### 📡 DATA SOURCE")

    st.caption("Open-Meteo Weather API")

    st.divider()

    st.markdown("### 📂 Train your own model")

    uploaded = st.file_uploader(
        "Upload labelled weather CSV",
        type=["csv"]
    )

    st.caption(
        "Required columns: seven weather features, "
        "storm_label and rain_label."
    )


# =========================================================
# LOCATION AND WEATHER
# =========================================================

if refresh or "weather_data" not in st.session_state or city != st.session_state.get("city"):

    try:

        with st.spinner("Connecting to meteorological satellites..."):

            location = geocode(city.strip())

            if not location:
                st.error("Location not found.")
                st.stop()

            weather_data = get_weather(
                location["latitude"],
                location["longitude"]
            )

            st.session_state.weather_data = weather_data
            st.session_state.location = location
            st.session_state.city = city

    except Exception as error:

        st.error(f"Weather connection failed: {error}")
        st.stop()


weather = st.session_state.weather_data
location = st.session_state.location
current = weather["current"]

hourly = pd.DataFrame(weather["hourly"])
daily = pd.DataFrame(weather["daily"])

hourly["time"] = pd.to_datetime(hourly["time"])

now = pd.Timestamp.now(
    tz=hourly["time"].dt.tz
) if hourly["time"].dt.tz is not None else pd.Timestamp.now()

future = hourly[hourly["time"] >= now].head(48).copy()

if future.empty:
    future = hourly.head(48).copy()


# =========================================================
# MODEL TRAINING / CSV UPLOAD
# =========================================================

if uploaded is not None:

    try:

        training_data = pd.read_csv(uploaded)

        required = FEATURES + ["storm_label", "rain_label"]

        missing = [
            col for col in required
            if col not in training_data.columns
        ]

        if missing:
            st.error("Missing columns: " + ", ".join(missing))
            st.stop()

        models = {}

        for label in ["storm_label", "rain_label"]:

            X_train, X_test, y_train, y_test = train_test_split(
                training_data[FEATURES],
                training_data[label].astype(int),
                test_size=.2,
                random_state=42,
                stratify=training_data[label]
            )

            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("forest", RandomForestClassifier(
                    n_estimators=180,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=42
                ))
            ])

            model.fit(X_train, y_train)

            models[label] = (
                model,
                accuracy_score(y_test, model.predict(X_test))
            )

        model_source = "CUSTOM HISTORICAL DATASET"

    except Exception as error:

        st.error(f"Dataset training failed: {error}")
        st.stop()

else:

    models = train_demo_models()
    model_source = "SYNTHETIC DEMO DATASET"


# =========================================================
# ML PREDICTIONS
# =========================================================

for feature in FEATURES:

    if feature not in future:
        future[feature] = np.nan

storm_model, storm_accuracy = models["storm_label"]
rain_model, rain_accuracy = models["rain_label"]

future["storm_probability"] = storm_model.predict_proba(
    future[FEATURES]
)[:, 1]

future["rain_probability"] = rain_model.predict_proba(
    future[FEATURES]
)[:, 1]

storm_probability = float(
    future["storm_probability"].max()
)

rain_probability = float(
    future["rain_probability"].max()
)

storm_level, storm_icon = risk(storm_probability)
rain_level, rain_icon = risk(rain_probability)


# =========================================================
# LOCATION HEADER
# =========================================================

st.markdown(
    f"""
    <div style="margin-top:28px;">
        <div style="font-size:12px;color:#75caff;
        letter-spacing:3px;font-weight:800;">
        LOCATION INTELLIGENCE
        </div>
        <h2 style="margin:5px 0;color:white;">
        📍 {location['name']}, {location['country']}
        </h2>
        <div class="muted">
        Updated: {current.get('time', 'Unavailable')}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LIVE WEATHER CARDS
# =========================================================

st.markdown('<div class="section-title">LIVE WEATHER CONDITIONS</div>',
            unsafe_allow_html=True)

cols = st.columns(5)

weather_metrics = [
    ("TEMPERATURE", f"{current.get('temperature_2m', '--')}°C", "🌡️"),
    ("HUMIDITY", f"{current.get('relative_humidity_2m', '--')}%", "💧"),
    ("WIND SPEED", f"{current.get('wind_speed_10m', '--')} km/h", "💨"),
    ("WIND GUSTS", f"{current.get('wind_gusts_10m', '--')} km/h", "🌬️"),
    ("PRESSURE", f"{current.get('pressure_msl', '--')} hPa", "📊")
]

for col, (label, value, icon) in zip(cols, weather_metrics):

    with col:

        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">{icon} {label}</div>
                <div class="big-value">{value}</div>
                <div class="muted">LIVE MEASUREMENT</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# RISK INTELLIGENCE
# =========================================================

st.markdown(
    '<div class="section-title">⚠️ AI RISK INTELLIGENCE</div>',
    unsafe_allow_html=True
)

a, b, c = st.columns([1, 1, 1.2])

with a:

    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">⛈️ STORM PREDICTION</div>
            <div class="big-value">{storm_icon} {storm_level}</div>
            <h2 style="color:#70caff;">
            {storm_probability:.0%}
            </h2>
            <div class="muted">Peak model score • Next 48 hours</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(storm_probability)

with b:

    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">🌧️ HEAVY RAINFALL</div>
            <div class="big-value">{rain_icon} {rain_level}</div>
            <h2 style="color:#70caff;">
            {rain_probability:.0%}
            </h2>
            <div class="muted">Peak model score • Next 48 hours</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(rain_probability)

with c:

    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">🧠 MODEL PERFORMANCE</div>
            <h2 style="color:#70caff;">RANDOM FOREST</h2>
            <p>Storm validation: {storm_accuracy:.1%}</p>
            <p>Rain validation: {rain_accuracy:.1%}</p>
            <div class="muted">{model_source}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

if storm_level == "HIGH" or rain_level == "HIGH":

    st.error(
        "HIGH MODELLED RISK: Monitor official IMD warnings "
        "and follow local emergency instructions."
    )

elif storm_level == "MODERATE" or rain_level == "MODERATE":

    st.warning(
        "MODERATE MODELLED RISK: Monitor changing weather "
        "conditions and updated forecasts."
    )

else:

    st.success(
        "LOW MODELLED RISK: Continue normal forecast monitoring."
    )


# =========================================================
# FORECAST ANALYTICS
# =========================================================

st.markdown(
    '<div class="section-title">📡 FORECAST ANALYTICS</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "⚡ RISK ANALYSIS",
    "🌧️ RAINFALL FORECAST",
    "💨 WIND ANALYSIS"
])

with tab1:

    chart_data = future.melt(
        id_vars=["time"],
        value_vars=[
            "storm_probability",
            "rain_probability"
        ],
        var_name="Risk",
        value_name="Probability"
    )

    chart_data["Risk"] = chart_data["Risk"].map({
        "storm_probability": "Storm risk",
        "rain_probability": "Heavy rainfall risk"
    })

    fig = px.line(
        chart_data,
        x="time",
        y="Probability",
        color="Risk",
        markers=True,
        template="plotly_dark",
        title="48-Hour AI Risk Outlook"
    )

    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:

    fig = px.bar(
        future,
        x="time",
        y="precipitation",
        color="precipitation_probability",
        color_continuous_scale="Blues",
        template="plotly_dark",
        title="Hourly Precipitation Forecast",
        labels={
            "time": "Local time",
            "precipitation": "Rainfall (mm)",
            "precipitation_probability": "Rain chance"
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

with tab3:

    fig = px.line(
        future,
        x="time",
        y=[
            "wind_speed_10m",
            "wind_gusts_10m"
        ],
        template="plotly_dark",
        title="Wind Speed and Gust Forecast"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 7-DAY FORECAST
# =========================================================

st.markdown(
    '<div class="section-title">📅 7-DAY WEATHER OUTLOOK</div>',
    unsafe_allow_html=True
)

daily["icon"] = daily["weather_code"].apply(weather_icon)

forecast_cols = st.columns(7)

for index, row in daily.iterrows():

    with forecast_cols[index]:

        st.markdown(
            f"""
            <div class="panel" style="text-align:center;padding:13px 5px;">
                <div class="muted">{row['time']}</div>
                <div style="font-size:29px;margin:12px 0;">
                    {row['icon']}
                </div>
                <b>{row['temperature_2m_max']}°</b>
                <div class="muted">
                    {row['temperature_2m_min']}° min
                </div>
                <div style="color:#79cfff;font-size:12px;margin-top:9px;">
                    🌧️ {row['precipitation_probability_max']}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# HOURLY FORECAST
# =========================================================

st.markdown(
    '<div class="section-title">🕒 HOURLY WEATHER DATA</div>',
    unsafe_allow_html=True
)

display = future[[
    "time",
    "temperature_2m",
    "precipitation_probability",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "storm_probability",
    "rain_probability"
]].copy()

display.columns = [
    "Time",
    "Temperature °C",
    "Rain chance %",
    "Rainfall mm",
    "Wind km/h",
    "Gusts km/h",
    "Storm score",
    "Rain score"
]

display["Storm score"] = display["Storm score"].map(
    lambda x: f"{x:.0%}"
)

display["Rain score"] = display["Rain score"].map(
    lambda x: f"{x:.0%}"
)

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# SARVAM AI WEATHER ASSISTANT
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="hero" style="
    min-height:160px;
    background:
    linear-gradient(90deg,rgba(3,13,30,.96),rgba(3,13,30,.5)),
    url('https://images.unsplash.com/photo-1470770841072-f978cf4d019e');
    background-size:cover;
    ">
        <div class="hero-tag">GENERATIVE WEATHER INTELLIGENCE</div>
        <h1 style="font-size:35px;">🤖 ATMOS AI ASSISTANT</h1>
        <p>Powered by Sarvam AI • English • हिंदी • मराठी</p>
    </div>
    """,
    unsafe_allow_html=True
)

weather_context = f"""
Location: {location['name']}, {location['country']}
Temperature: {current.get('temperature_2m')} Celsius
Humidity: {current.get('relative_humidity_2m')} percent
Wind speed: {current.get('wind_speed_10m')} km/h
Wind gusts: {current.get('wind_gusts_10m')} km/h
Pressure: {current.get('pressure_msl')} hPa
Storm model risk: {storm_level}, score {storm_probability:.0%}
Heavy rainfall model risk: {rain_level}, score {rain_probability:.0%}
"""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask ATMOS AI about storms, rainfall or weather..."
)

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("ATMOS AI is analysing your question..."):

            answer = ask_sarvam(
                question,
                weather_context,
                st.session_state.chat_history[:-1]
            )

            st.markdown(answer)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })


# =========================================================
# CUSTOM PREDICTION LAB
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔬 ADVANCED PREDICTION LAB</div>',
    unsafe_allow_html=True
)

st.caption(
    "Test different meteorological conditions using the trained ML models."
)

with st.expander("OPEN PREDICTION LAB"):

    x1, x2, x3 = st.columns(3)

    with x1:

        probability = st.slider(
            "Precipitation probability (%)",
            0, 100, 70
        )

        precipitation = st.number_input(
            "Rainfall (mm/hour)",
            0.0, 150.0, 8.0
        )

        humidity = st.slider(
            "Humidity (%)",
            0, 100, 85
        )

    with x2:

        wind = st.number_input(
            "Wind speed (km/h)",
            0.0, 200.0, 25.0
        )

        gust = st.number_input(
            "Wind gusts (km/h)",
            0.0, 250.0, 40.0
        )

        cloud = st.slider(
            "Cloud cover (%)",
            0, 100, 90
        )

    with x3:

        pressure = st.number_input(
            "Pressure (hPa)",
            850.0, 1100.0, 1005.0
        )

        predict = st.button(
            "⚡ RUN PREDICTION",
            use_container_width=True
        )

    if predict:

        sample = pd.DataFrame([[
            probability,
            precipitation,
            wind,
            gust,
            pressure,
            humidity,
            cloud
        ]], columns=FEATURES)

        storm = float(
            storm_model.predict_proba(sample)[0, 1]
        )

        rain = float(
            rain_model.predict_proba(sample)[0, 1]
        )

        r1, r2 = st.columns(2)

        with r1:

            st.metric(
                "Storm risk",
                risk(storm)[0],
                f"{storm:.0%} model score"
            )

        with r2:

            st.metric(
                "Heavy rainfall risk",
                risk(rain)[0],
                f"{rain:.0%} model score"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;padding:25px;">
        <div style="color:#68caff;font-size:22px;
        font-weight:900;letter-spacing:4px;">
            ATMOS AI
        </div>
        <div class="muted" style="margin-top:10px;">
            ADVANCED WEATHER INTELLIGENCE SYSTEM
        </div>
        <div class="muted" style="margin-top:8px;">
            Built with Python • Streamlit • Machine Learning • Sarvam AI
        </div>
        <div class="muted" style="margin-top:15px;">
            Experimental predictions are not official weather warnings.
            Follow IMD advisories for safety decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
