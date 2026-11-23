import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

load_dotenv()  # Loads SARVAM_API_KEY from the local .env file, if present

st.set_page_config(page_title="Atmos AI | Weather Intelligence", page_icon="🌩️", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.stApp {background: linear-gradient(145deg,#071426 0%,#0b1d33 55%,#102a43 100%); color:#eef6ff;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
.block-container {padding-top: 1.6rem; max-width: 1450px;}
.hero {padding:26px 30px;border:1px solid rgba(125,190,255,.22);border-radius:22px;
background:linear-gradient(115deg,rgba(22,57,91,.95),rgba(13,34,59,.88));}
.hero h1 {font-size:2.35rem; margin:0;color:#f4f9ff;}
.hero p {color:#a9c5df;margin:.45rem 0 0 0;}
.metric-card {background:rgba(19,45,72,.88);border:1px solid rgba(135,190,235,.18);
border-radius:17px;padding:17px 19px;min-height:112px;}
.metric-label {color:#9dbbd5;font-size:.85rem;}
.metric-value {font-size:1.8rem;font-weight:750;color:#f3f8ff;margin-top:5px;}
.metric-note {color:#8eabc5;font-size:.78rem;margin-top:2px;}
.section {font-size:1.18rem;font-weight:700;color:#eaf4ff;margin:1rem 0 .6rem;}
.small-muted {color:#9bb4cc;font-size:.83rem;}
div[data-testid="stSidebar"] {background:#09192b;}
.stButton button {border-radius:11px;font-weight:650;}
</style>
""", unsafe_allow_html=True)

FEATURES = ["precipitation_probability", "precipitation", "wind_speed_10m",
            "wind_gusts_10m", "pressure_msl", "relative_humidity_2m", "cloud_cover"]

@st.cache_data(ttl=1800)
def geocode(city):
    url="https://geocoding-api.open-meteo.com/v1/search"
    r=requests.get(url,params={"name":city,"count":1,"language":"en","format":"json"},timeout=15)
    r.raise_for_status()
    results=r.json().get("results",[])
    if not results: return None
    x=results[0]
    return {"name":x["name"],"country":x.get("country",""),"latitude":x["latitude"],"longitude":x["longitude"]}

@st.cache_data(ttl=600)
def get_weather(lat,lon):
    url="https://api.open-meteo.com/v1/forecast"
    params={
        "latitude":lat,"longitude":lon,
        "current":"temperature_2m,relative_humidity_2m,precipitation,rain,showers,weather_code,wind_speed_10m,wind_gusts_10m,pressure_msl",
        "hourly":"temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,rain,showers,wind_speed_10m,wind_gusts_10m,pressure_msl,cloud_cover",
        "forecast_days":3,"timezone":"auto"
    }
    r=requests.get(url,params=params,timeout=20)
    r.raise_for_status()
    return r.json()

def make_demo_models():
    # Educational baseline only: synthetic labels approximate broad weather-risk patterns.
    rng=np.random.default_rng(42)
    n=12000
    df=pd.DataFrame({
        "precipitation_probability":rng.uniform(0,100,n),
        "precipitation":rng.gamma(1.1,3.2,n),
        "wind_speed_10m":rng.weibull(1.8,n)*23,
        "wind_gusts_10m":rng.weibull(1.7,n)*34,
        "pressure_msl":rng.normal(1008,9,n),
        "relative_humidity_2m":rng.uniform(25,100,n),
        "cloud_cover":rng.uniform(0,100,n)
    })
    storm_score=(df.wind_gusts_10m/75 + df.wind_speed_10m/55 +
                 df.precipitation_probability/150 + (1010-df.pressure_msl)/35 +
                 df.cloud_cover/260 + rng.normal(0,.17,n))
    rain_score=(df.precipitation_probability/65 + df.precipitation/18 +
                df.relative_humidity_2m/180 + df.cloud_cover/180 + rng.normal(0,.18,n))
    df["storm_label"]=(storm_score>1.12).astype(int)
    df["rain_label"]=(rain_score>1.65).astype(int)
    models={}
    for label in ["storm_label","rain_label"]:
        X_train,X_test,y_train,y_test=train_test_split(df[FEATURES],df[label],test_size=.2,random_state=42,stratify=df[label])
        model=Pipeline([("imputer",SimpleImputer(strategy="median")),
                        ("rf",RandomForestClassifier(n_estimators=160,max_depth=11,min_samples_leaf=3,
                                                     class_weight="balanced",random_state=42,n_jobs=-1))])
        model.fit(X_train,y_train)
        models[label]=(model,accuracy_score(y_test,model.predict(X_test)))
    return models

def risk_label(p):
    if p>=.75: return "HIGH", "#ff626d"
    if p>=.45: return "MODERATE", "#ffbd59"
    return "LOW", "#43d6a0"

def metric(label,value,note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div style="color:#71c5ff;font-weight:700;letter-spacing:2px;font-size:.78rem;">WEATHER INTELLIGENCE • AI / ML</div>
  <h1>🌩️ ATMOS AI</h1>
  <p>Storm detection · Heavy rainfall outlook · Forecast intelligence</p>
</div>
""",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌐 Control Centre")
    city=st.text_input("Search city", "Pune")
    st.caption("Examples: Pune, Mumbai, Delhi, Bengaluru")
    run=st.button("Analyse weather",use_container_width=True,type="primary")
    st.divider()
    st.markdown("### 🧠 Model settings")
    st.caption("Upload a labelled CSV to train the models on your own dataset.")
    upload=st.file_uploader("Training dataset (CSV)",type=["csv"])
    st.caption("Required feature columns: "+", ".join(FEATURES))
    st.caption("Also include storm_label and rain_label (0/1).")
    st.divider()
    st.markdown("**Data source**")
    st.caption("Live forecast: Open-Meteo API")
    st.caption("Demo model: Random Forest baseline")

if "city" not in st.session_state: st.session_state.city="Pune"
if run or "weather" not in st.session_state or st.session_state.get("city") != city:
    try:
        place=geocode(city.strip())
        if not place: st.error("City not found. Try another city name."); st.stop()
        weather=get_weather(place["latitude"],place["longitude"])
        st.session_state.weather=weather
        st.session_state.place=place
        st.session_state.city=city
    except Exception as e:
        st.error(f"Could not load live weather data: {e}")
        st.info("Check your internet connection and try again.")
        st.stop()

if upload is not None:
    try:
        train= pd.read_csv(upload)
        required=FEATURES+["storm_label","rain_label"]
        missing=[c for c in required if c not in train.columns]
        if missing: st.error("Dataset is missing columns: "+", ".join(missing)); st.stop()
        models={}
        for label in ["storm_label","rain_label"]:
            Xtr,Xte,ytr,yte=train_test_split(train[FEATURES],train[label].astype(int),test_size=.2,random_state=42,stratify=train[label])
            model=Pipeline([("imputer",SimpleImputer(strategy="median")),
                            ("rf",RandomForestClassifier(n_estimators=180,max_depth=12,class_weight="balanced",random_state=42,n_jobs=-1))])
            model.fit(Xtr,ytr); models[label]=(model,accuracy_score(yte,model.predict(Xte)))
        model_source="Your uploaded dataset"
    except Exception as e:
        st.error(f"Could not train from uploaded dataset: {e}")
        st.stop()
else:
    with st.spinner("Preparing AI models..."):
        models=make_demo_models()
    model_source="Synthetic educational baseline"

w=st.session_state.weather
place=st.session_state.place
cur=w["current"]; hourly=pd.DataFrame(w["hourly"])
hourly["time"]=pd.to_datetime(hourly["time"])
now=pd.Timestamp.now(tz=hourly["time"].dt.tz) if hourly["time"].dt.tz is not None else pd.Timestamp.now()
future=hourly[hourly["time"]>=now].head(48).copy()
if future.empty: future=hourly.head(48).copy()

st.markdown(f"### 📍 {place['name']}, {place['country']}  <span class='small-muted'>| Updated {cur.get('time','')}</span>",unsafe_allow_html=True)
st.markdown('<div class="small-muted">Live conditions and a rolling 48-hour risk outlook</div>',unsafe_allow_html=True)
c1,c2,c3,c4,c5=st.columns(5)
with c1: metric("Temperature",f"{cur.get('temperature_2m','—')}°C","Current")
with c2: metric("Humidity",f"{cur.get('relative_humidity_2m','—')}%","Relative humidity")
with c3: metric("Wind",f"{cur.get('wind_speed_10m','—')} km/h","10 m wind")
with c4: metric("Wind gusts",f"{cur.get('wind_gusts_10m','—')} km/h","Peak gust")
with c5: metric("Pressure",f"{cur.get('pressure_msl','—')} hPa","Mean sea-level")

# Build hourly feature frame and model probabilities
for col in FEATURES:
    if col not in future: future[col]=np.nan
storm_model,storm_acc=models["storm_label"]
rain_model,rain_acc=models["rain_label"]
future["storm_probability"]=storm_model.predict_proba(future[FEATURES])[:,1]
future["rain_probability"]=rain_model.predict_proba(future[FEATURES])[:,1]
peak_storm=float(future["storm_probability"].max())
peak_rain=float(future["rain_probability"].max())
slabel,scolor=risk_label(peak_storm)
rlabel,rcolor=risk_label(peak_rain)

st.markdown('<div class="section">⚠️ AI Risk Assessment</div>',unsafe_allow_html=True)
a,b,c=st.columns([1,1,1.2])
with a:
    st.markdown(f'<div class="metric-card"><div class="metric-label">STORM RISK • NEXT 48 HOURS</div><div class="metric-value" style="color:{scolor}">{slabel}</div><div class="metric-note">Peak model probability: {peak_storm:.0%}</div></div>',unsafe_allow_html=True)
    st.progress(peak_storm)
with b:
    st.markdown(f'<div class="metric-card"><div class="metric-label">HEAVY RAIN RISK • NEXT 48 HOURS</div><div class="metric-value" style="color:{rcolor}">{rlabel}</div><div class="metric-note">Peak model probability: {peak_rain:.0%}</div></div>',unsafe_allow_html=True)
    st.progress(peak_rain)
with c:
    st.markdown('<div class="metric-card"><div class="metric-label">SYSTEM STATUS</div><div class="metric-value">● Monitoring</div><div class="metric-note">Model source: '+model_source+'</div><div class="metric-note">Storm model validation accuracy: '+f'{storm_acc:.1%}'+'</div><div class="metric-note">Rain model validation accuracy: '+f'{rain_acc:.1%}'+'</div></div>',unsafe_allow_html=True)

if slabel=="HIGH" or rlabel=="HIGH":
    st.error("HIGH RISK SIGNAL: The model indicates elevated storm or heavy-rain potential in the next 48 hours. Follow official local weather warnings.")
elif slabel=="MODERATE" or rlabel=="MODERATE":
    st.warning("MODERATE RISK: Conditions may become unsettled. Monitor updated forecasts and local advisories.")
else:
    st.success("LOW MODELLED RISK: No strong signal in this model run. Continue to monitor changing conditions.")

st.markdown('<div class="section">📈 Forecast intelligence — next 48 hours</div>',unsafe_allow_html=True)
tab1,tab2,tab3=st.tabs(["Risk probability","Rainfall & precipitation","Wind & pressure"])
with tab1:
    plot=future[["time","storm_probability","rain_probability"]].melt("time",var_name="Risk type",value_name="Probability")
    plot["Risk type"]=plot["Risk type"].map({"storm_probability":"Storm risk","rain_probability":"Heavy rainfall risk"})
    fig=px.line(plot,x="time",y="Probability",color="Risk type",markers=True,
                labels={"time":"Local time","Probability":"Model probability"})
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",legend_title_text="")
    fig.update_yaxes(tickformat=".0%",range=[0,1])
    st.plotly_chart(fig,use_container_width=True)
with tab2:
    fig=px.bar(future,x="time",y="precipitation",color="precipitation_probability",
               color_continuous_scale="Blues",labels={"time":"Local time","precipitation":"Precipitation (mm)","precipitation_probability":"Rain chance"})
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
with tab3:
    fig=px.line(future,x="time",y=["wind_speed_10m","wind_gusts_10m","pressure_msl"],
                labels={"time":"Local time","value":"Reading","variable":"Measurement"})
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)

st.markdown('<div class="section">🕒 Hourly forecast table</div>',unsafe_allow_html=True)
show=future[["time","temperature_2m","precipitation_probability","precipitation","wind_speed_10m","wind_gusts_10m","storm_probability","rain_probability"]].copy()
show.columns=["Local time","Temp °C","Rain chance %","Rain mm","Wind km/h","Gust km/h","Storm risk","Heavy rain risk"]
show["Rain chance %"]=show["Rain chance %"].round(0)
show["Storm risk"]=show["Storm risk"].map(lambda x:f"{x:.0%}")
show["Heavy rain risk"]=show["Heavy rain risk"].map(lambda x:f"{x:.0%}")
st.dataframe(show,use_container_width=True,hide_index=True)

st.markdown('<div class="section">🔬 Prediction lab</div>',unsafe_allow_html=True)
st.caption("Enter weather conditions to estimate modelled storm and heavy-rain risk. This is an educational decision-support prototype, not an official warning service.")
with st.expander("Run a custom prediction"):
    x1,x2,x3=st.columns(3)
    with x1:
        pp=st.slider("Precipitation probability (%)",0,100,70)
        mm=st.number_input("Precipitation (mm/hour)",0.0,150.0,8.0,0.5)
        hum=st.slider("Relative humidity (%)",0,100,85)
    with x2:
        wind=st.number_input("Wind speed (km/h)",0.0,200.0,25.0,1.0)
        gust=st.number_input("Wind gusts (km/h)",0.0,250.0,40.0,1.0)
        cloud=st.slider("Cloud cover (%)",0,100,90)
    with x3:
        pressure=st.number_input("Sea-level pressure (hPa)",850.0,1100.0,1005.0,0.5)
        st.write("")
        st.write("")
        predict=st.button("⚡ Predict risk",type="primary",use_container_width=True)
    if predict:
        sample=pd.DataFrame([[pp,mm,wind,gust,pressure,hum,cloud]],columns=FEATURES)
        sp=float(storm_model.predict_proba(sample)[0,1]); rp=float(rain_model.predict_proba(sample)[0,1])
        sl,sc=risk_label(sp); rl,rc=risk_label(rp)
        q1,q2=st.columns(2)
        with q1: st.metric("Storm risk",sl,f"{sp:.0%} model probability")
        with q2: st.metric("Heavy rainfall risk",rl,f"{rp:.0%} model probability")

st.markdown("---")
st.markdown('<div class="section">🤖 Sarvam AI Weather Assistant</div>',unsafe_allow_html=True)
st.caption("Ask questions about this forecast in English, Hindi, or Marathi. Sarvam AI receives the current forecast and experimental model risk values as context.")

sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
if not sarvam_key:
    try:
        sarvam_key = str(st.secrets.get("SARVAM_API_KEY", "")).strip()
    except Exception:
        sarvam_key = ""

if not sarvam_key:
    st.info("Sarvam AI is not configured yet. Add your key to a local .env file as SARVAM_API_KEY=your_key, or add SARVAM_API_KEY to Streamlit Cloud Secrets, then restart the app.")
else:
    if "sarvam_messages" not in st.session_state:
        st.session_state.sarvam_messages = []

    for message in st.session_state.sarvam_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about storm risk, rainfall, or this forecast...")
    if question:
        st.session_state.sarvam_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Sarvam AI is analysing the forecast..."):
                try:
                    from sarvamai import SarvamAI
                    client = SarvamAI(api_subscription_key=sarvam_key)
                    context = (
                        f"Location: {place['name']}, {place['country']}. "
                        f"Current temperature: {cur.get('temperature_2m')} C; "
                        f"humidity: {cur.get('relative_humidity_2m')}%; "
                        f"wind: {cur.get('wind_speed_10m')} km/h; "
                        f"gusts: {cur.get('wind_gusts_10m')} km/h; "
                        f"pressure: {cur.get('pressure_msl')} hPa. "
                        f"Experimental ML peak storm risk next 48h: {peak_storm:.0%}; "
                        f"experimental ML peak heavy-rain risk next 48h: {peak_rain:.0%}. "
                        f"Risk model source: {model_source}. "
                        "The default model may be trained on synthetic educational data. "
                    )
                    history = st.session_state.sarvam_messages[-8:]
                    messages = [{
                        "role": "system",
                        "content": (
                            "You are Atmos AI's weather education assistant. Explain the supplied weather context clearly. "
                            "Do not invent observations, forecast values, or official alerts. Clearly say that modelled risk "
                            "scores are experimental and are not calibrated probabilities unless validated against real historical data. "
                            "Never present yourself as an official meteorological authority. For safety, tell users to follow "
                            "official IMD/local authority warnings. Reply in the language used by the user when possible. "
                            "Current forecast context: " + context
                        )
                    }] + history
                    response = client.chat.completions(
                        model="sarvam-105b-conversations",
                        messages=messages,
                    )
                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.sarvam_messages.append({"role": "assistant", "content": answer})
                except Exception as exc:
                    st.error(f"Sarvam AI request failed. Check your API key, account credits, and internet connection. Details: {exc}")

st.markdown("**About this project**")
st.markdown("""
Atmos AI combines live meteorological forecast variables with Random Forest classifiers to demonstrate storm and heavy-rain risk estimation. The default models are trained on synthetic educational data and must not be treated as operational meteorological forecasts. For a credible academic or real-world model, train and validate against historical observations from a trusted weather agency, using time-based validation and location-aware testing.
""")
st.caption("Live weather data provided by Open-Meteo. Predictions are experimental and are not a substitute for IMD alerts or emergency guidance.")
