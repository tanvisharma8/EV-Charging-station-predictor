# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:19:55 2025

@author: Prisha D
"""

# app.py — Final working Streamlit app (Matplotlib + Folium + Waiting Time Prediction)
import streamlit as st
import pandas as pd
import numpy as np
import re
import folium
import joblib
import io
import matplotlib.pyplot as plt
from math import radians, sin, cos, asin, sqrt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="EV Charging Station Finder", layout="wide")

# -------------------------
# Helpers
# -------------------------
def extract_number(s):
    if pd.isna(s): 
        return np.nan
    m = re.search(r'(\d+\.?\d*)', str(s))
    return float(m.group(1)) if m else np.nan



def parse_vehicle_list(val):
    if pd.isna(val):
        return []
    s = str(val).strip()
    s = s.strip("[]")
    s = s.replace("'", "").replace('"', "")
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items

def haversine(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except:
        return np.nan
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def folium_map_html_from_df(df_map, center_lat, center_lon, zoom_start=10):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    for _, r in df_map.iterrows():
        try:
            popup_html = f"<b>{r.get('name','')}</b><br>City: {r.get('city','')}<br>Capacity: {r.get('capacity_kw','')} kW"
            folium.Marker(
                [r['latitude'], r['longitude']],
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
        except Exception:
            continue
    return m._repr_html_()
# -------------------------
# Load & preprocess (Cleaned version)
# -------------------------
import os
import pandas as pd
import numpy as np
import re
import streamlit as st

@st.cache_data
def load_prepare(local_path="ev_final.csv",
                 fallback_url="https://raw.githubusercontent.com/tanvisharma8/EV-Charging-station-predictor/main/EV_Charging_Station_ML_Project/ev_final.csv"):

    # --- Load CSV ---
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        st.warning("Local CSV not found, loading from GitHub URL...")
        df = pd.read_csv(fallback_url)

    df.columns = [c.strip() for c in df.columns]

    # Standardize station name
    if "name" not in df.columns and "station_name" in df.columns:
        df.rename(columns={"station_name":"name"}, inplace=True)

    # Ensure necessary columns exist
    for c in ["capacity","vehicle_type","power_type","type","latitude","longitude",
              "city","vendor_name","address","open","close","cost_per_unit"]:
        if c not in df.columns:
            df[c] = np.nan

    # --- Capacity ---
    def extract_number(s):
        if pd.isna(s): return np.nan
        m = re.search(r'(\d+\.?\d*)', str(s))
        return float(m.group(1)) if m else np.nan
    df['capacity_kw'] = df['capacity'].apply(extract_number)
    df['capacity_kw'] = df['capacity_kw'].fillna(df['capacity_kw'].median())

    # --- Vehicle count ---
    def parse_vehicle_list(val):
        if pd.isna(val): return []
        s = str(val).strip().strip("[]").replace("'", "").replace('"', "")
        return [x.strip() for x in s.split(",") if x.strip()]
    df['vehicle_list'] = df['vehicle_type'].apply(parse_vehicle_list)
    df['vehicle_count'] = df['vehicle_list'].apply(len)

    # --- Power type ---
    df['power_type'] = df['power_type'].fillna('').astype(str)
    if 'type' in df.columns:
        df['type'] = df['type'].fillna('').astype(str)
    df['is_dc'] = np.where(df['power_type'].str.strip() != '', df['power_type'], df['type'])
    df['is_dc'] = df['is_dc'].astype(str).str.upper().str.contains('DC').astype(int)

    # --- Coordinates ---
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude','longitude']).reset_index(drop=True)

    # --- Cost cleaning ---
    def clean_cost(x):
        if pd.isna(x): return np.nan
        x = str(x).replace("₹","").replace("Rs","").replace("rs","")
        x = x.replace("/unit","").replace("per unit","").replace("kW","").replace("-","").strip()
        return pd.to_numeric(x, errors='coerce')
    df['cost_per_unit'] = df['cost_per_unit'].apply(clean_cost)

    # --- Open hours ---
    def safe_open_hours(o,c):
        try:
            o_pd = pd.to_datetime(o, errors='coerce')
            c_pd = pd.to_datetime(c, errors='coerce')
            if pd.isna(o_pd) or pd.isna(c_pd): return np.nan
            diff = (c_pd - o_pd).total_seconds()
            if diff < 0: diff += 24*3600
            return diff/3600.0
        except:
            return np.nan
    df['open_hours'] = df.apply(lambda r: safe_open_hours(r.get('open'), r.get('close')), axis=1)
    df['open_hours'] = df['open_hours'].fillna(24.0)

    # --- Fill string columns ---
    df['city'] = df['city'].fillna('').astype(str)
    df['vendor_name'] = df['vendor_name'].fillna('').astype(str)
    df['address'] = df['address'].fillna('').astype(str)

    # --- Synthetic waiting time ---
    np.random.seed(42)
    noise = np.random.normal(0, 4, len(df))
    df["wait_time"] = (25 - 0.12*df["capacity_kw"] + 4*df["is_dc"] + 3*df["vehicle_count"] + noise).clip(1, 60)

    return df

    # --- Load dataset ---
    df = load_prepare()


    # -------------------------
    # Clean capacity
    # -------------------------
    def extract_number(s):
        if pd.isna(s): return np.nan
        m = re.search(r'(\d+\.?\d*)', str(s))
        return float(m.group(1)) if m else np.nan
    df['capacity_kw'] = df['capacity'].apply(extract_number)
    df['capacity_kw'] = df['capacity_kw'].fillna(df['capacity_kw'].median())

    # -------------------------
    # Vehicle type count
    # -------------------------
    def parse_vehicle_list(val):
        if pd.isna(val):
            return []
        s = str(val).strip().strip("[]").replace("'", "").replace('"', "")
        return [x.strip() for x in s.split(",") if x.strip()]
    df['vehicle_list'] = df['vehicle_type'].apply(parse_vehicle_list)
    df['vehicle_count'] = df['vehicle_list'].apply(len)

    # -------------------------
    # Power type
    # -------------------------
    df['power_type'] = df['power_type'].fillna('').astype(str)
    df['type'] = df['type'].fillna('').astype(str)
    df['is_dc'] = np.where(df['power_type'].str.strip() != '', df['power_type'], df['type'])
    df['is_dc'] = df['is_dc'].astype(str).str.upper().str.contains('DC').astype(int)

    # -------------------------
    # Latitude / Longitude
    # -------------------------
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude','longitude']).reset_index(drop=True)

    # -------------------------
    # Cost per unit
    # -------------------------
    def clean_cost(x):
        if pd.isna(x): return np.nan
        x = str(x).replace("₹", "").replace("Rs", "").replace("rs", "")
        x = x.replace("/unit", "").replace("per unit", "")
        x = x.replace("kW", "").replace("-", "").strip()
        return pd.to_numeric(x, errors='coerce')
    df['cost_per_unit'] = df['cost_per_unit'].apply(clean_cost)
    df['cost_per_unit'] = df['cost_per_unit'].fillna(df['cost_per_unit'].median())

    # -------------------------
    # Open hours
    # -------------------------
    def safe_open_hours(o, c):
        try:
            o_pd = pd.to_datetime(o, errors='coerce')
            c_pd = pd.to_datetime(c, errors='coerce')
            if pd.isna(o_pd) or pd.isna(c_pd):
                return 24.0
            diff = c_pd - o_pd
            secs = diff.total_seconds()
            if secs < 0: secs += 24*3600
            return secs / 3600.0
        except:
            return 24.0
    df['open_hours'] = df.apply(lambda r: safe_open_hours(r.get('open'), r.get('close')), axis=1)

    # -------------------------
    # Fill missing text fields
    # -------------------------
    for col in ['city', 'vendor_name', 'address']:
        df[col] = df[col].fillna('').astype(str)

    # -------------------------
    # Remove outliers (optional)
    # -------------------------
    df = df[df['capacity_kw'] <= df['capacity_kw'].quantile(0.99)]
    df = df[df['vehicle_count'] <= df['vehicle_count'].quantile(0.99)]
    df = df[df['cost_per_unit'] <= df['cost_per_unit'].quantile(0.99)]

    # -------------------------
    # Synthetic waiting time
    # -------------------------
    np.random.seed(42)
    noise = np.random.normal(0, 4, len(df))
    df["wait_time"] = (
        25
        - 0.12 * df["capacity_kw"]
        + 4 * df["is_dc"]
        + 3 * df["vehicle_count"]
        + noise
    ).clip(1, 60)

    return df

# Load cleaned data
df = load_prepare("ev_final.csv")

# -------------------------
# Train WAITING TIME model
# -------------------------
@st.cache_data
def train_wait_model(df_in):
    tmp = df_in[['capacity_kw', 'vehicle_count', 'is_dc', 'open_hours', 'wait_time']].dropna()

    X = tmp[['capacity_kw', 'vehicle_count', 'is_dc', 'open_hours']]
    y = tmp['wait_time']

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(Xs, y)

    return model, scaler

wait_model, wait_scaler = train_wait_model(df)

# -------------------------
# City centroids
# -------------------------
city_centroids = df.groupby(df['city'].str.strip().str.lower())[["latitude","longitude"]].mean().to_dict('index')
def get_city_centroid(city_name):
    if not isinstance(city_name, str): return None, None
    key = city_name.strip().lower()
    if key in city_centroids:
        return city_centroids[key]['latitude'], city_centroids[key]['longitude']
    return None, None

# -------------------------
# Recommendation function (unchanged)
# -------------------------
def recommend(df_source, lat, lon, req_power, max_price, prefer_dc, nearest_only, top_n):

    d = df_source.copy()
    d['distance_km'] = d.apply(lambda r: haversine(lat, lon, r['latitude'], r['longitude']), axis=1)

    cand = d.copy()
    cand = cand[(cand['capacity_kw'] >= (req_power*0.3)) | (cand['capacity_kw'] >= (req_power*0.5))]

    if nearest_only:
        return cand.sort_values('distance_km').head(top_n)

    def norm(s):
        if s.max() - s.min() < 1e-9:
            return pd.Series(0.5, index=s.index)
        return (s - s.min())/(s.max()-s.min())

    cap_score = (cand['capacity_kw']/req_power).clip(0,1)
    dist_score = 1 - norm(cand['distance_km'])
    power_pref = cand['is_dc'].apply(lambda v: 1.0 if (prefer_dc==True and v==1) or (prefer_dc==False and v==0) else 0.5)

    vehicle_score = norm(cand['vehicle_count'])

    cand['final_score'] = (
        0.35*dist_score +
        0.35*cap_score +
        0.10*power_pref +
        0.10*vehicle_score +
        0.10*(1 - norm(cand['wait_time']))
    )

    return cand.sort_values('final_score', ascending=False).head(top_n)

# -------------------------
# UI
# -------------------------
page = st.sidebar.radio(
    "Choose",
    ["Recommendation", "Dashboard", "Model", "Save Model"],
    key="main_menu_radio"
)

# -----------------------------------------------------
# MODEL PAGE — WAITING TIME PREDICTION (Random Forest)
# -----------------------------------------------------
if page == "Model":
    st.title("⏳ Waiting Time Prediction Model")
    st.write("Predict estimated waiting time for a charging station.")

    # --- Input fields ---
    c1, c2, c3, c4 = st.columns(4)
    cap_i = c1.number_input("🔌 Capacity (kW)", min_value=0.0, value=15.0)
    vc_i = c2.number_input("🚗 Vehicle count", min_value=0, value=1)
    dc_i = c3.selectbox("⚡ Charger Type", ["AC Charger (0)", "DC Fast Charger (1)"])
    oh_i = c4.number_input("⏳ Open hours", min_value=0.0, max_value=24.0, value=24.0)

    dc_value = 1 if "DC" in dc_i else 0

    # --- Train/test split & model metrics ---
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Prepare features and target
    X = df[['capacity_kw', 'vehicle_count', 'is_dc', 'open_hours']]
    y = df['wait_time']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    # Metrics function
    def regression_metrics(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        accuracy = r2 * 100
        return r2, mae, rmse, accuracy

    r2_train, mae_train, rmse_train, acc_train = regression_metrics(y_train, y_train_pred)
    r2_test, mae_test, rmse_test, acc_test = regression_metrics(y_test, y_test_pred)

    # Display model metrics
    st.subheader("📊 Model Performance")
    st.write("**Training Data:**")
    st.write(f"R²: {r2_train:.3f} | MAE: {mae_train:.2f} min | RMSE: {rmse_train:.2f} min | Accuracy: {acc_train:.1f}%")
    st.write("**Testing Data:**")
    st.write(f"R²: {r2_test:.3f} | MAE: {mae_test:.2f} min | RMSE: {rmse_test:.2f} min | Accuracy: {acc_test:.1f}%")

    # --- Predict waiting time for user input ---
    if st.button("🔮 Predict Waiting Time"):
        Xsamp = pd.DataFrame([[cap_i, vc_i, dc_value, oh_i]],
                             columns=['capacity_kw', 'vehicle_count', 'is_dc', 'open_hours'])
        pred = rf_model.predict(Xsamp)[0]

        st.success("Prediction Complete!")
        st.metric("⏱ Estimated Waiting Time (minutes)", f"{pred:.1f}")

# -----------------------------------------------------
# Save Model
# -----------------------------------------------------
if page == "Save Model":
    st.title("💾 Save trained WAITING TIME model")
    if st.button("Save model and scaler"):
        try:
            joblib.dump(wait_model, "ev_wait_model.pkl")
            joblib.dump(wait_scaler, "ev_wait_scaler.pkl")
            st.success("Saved ev_wait_model.pkl and ev_wait_scaler.pkl")
        except Exception as e:
            st.error("Save failed: " + str(e))


# -------------------------
# Recommendation function
# -------------------------
def recommend(df_source, lat, lon, req_power, max_price, prefer_dc, nearest_only, top_n):
    d = df_source.copy()
    d['distance_km'] = d.apply(lambda r: haversine(lat, lon, r['latitude'], r['longitude']), axis=1)

    # filter by price (if finite)
    if np.isfinite(max_price):
        cand = d[d['cost_per_unit'] <= max_price].copy()
        if cand.empty:
            cand = d.copy()
    else:
        cand = d.copy()

    # relaxed capacity filter
    cand = cand[(cand['capacity_kw'] >= (req_power*0.3)) | (cand['capacity_kw'] >= (req_power*0.5))].copy()
    if cand.empty:
        cand = d.copy()

    if nearest_only:
        return cand.sort_values('distance_km').head(top_n)

    def norm(s):
        if s.max() - s.min() < 1e-9:
            return pd.Series(0.5, index=s.index)
        return (s - s.min())/(s.max()-s.min())

    cap_score = (cand['capacity_kw']/req_power).clip(0,1)
    dist_score = 1 - norm(cand['distance_km'])
    price_score = 1 - norm(cand['cost_per_unit'])
    if prefer_dc is None:
        power_pref = pd.Series(0.7, index=cand.index)
    else:
        power_pref = cand['is_dc'].apply(lambda v: 1.0 if (v==1 and prefer_dc) or (v==0 and not prefer_dc) else 0.3)
    vehicle_score = norm(cand['vehicle_count'])

    cand['final_score'] = (0.30*dist_score + 0.30*cap_score + 0.20*price_score + 0.10*power_pref + 0.10*vehicle_score)
    cand['final_score'] += cand['capacity_kw'].apply(lambda c: 0.02 if c >= req_power else 0.0)
    return cand.sort_values('final_score', ascending=False).head(top_n)



# Recommendation page
if page == "Recommendation":
    st.title("🔍 EV Charging Station Recommendation")
    st.write("Enter city name or lat,lon and filters.")

    col1, col2 = st.columns([2,1])
    with col1:
        user_loc = st.text_input("City or 'lat,lon' (leave blank to use data center):", value="")
    with col2:
        pref = st.selectbox("Power preference", ["No preference","Prefer DC","Prefer AC"])
        prefer_dc = None if pref=="No preference" else (True if pref=="Prefer DC" else False)

    req_power = st.slider("Required power (kW)", 1, 300, 15)
    max_price = st.number_input("Max price per unit (enter large for no limit)", min_value=0.0, value=float(df['cost_per_unit'].max()))
    nearest_only = st.checkbox("Nearest-only mode", value=False)
    top_n = st.number_input("How many results?", 1, 50, value=10)

    if st.button("Find Stations"):
        # get coordinates
        if user_loc.strip()=="":
            st.info("Using dataset center as location.")
            user_lat = df['latitude'].mean(); user_lon = df['longitude'].mean()
        else:
            parsed = None
            if "," in user_loc:
                parts = [p.strip() for p in user_loc.split(",")]
                if len(parts) >= 2:
                    try:
                        parsed = (float(parts[0]), float(parts[1]))
                    except:
                        parsed = None
            if parsed:
                user_lat, user_lon = parsed
            else:
                latc, lonc = get_city_centroid(user_loc)
                if latc is not None:
                    user_lat, user_lon = latc, lonc
                    st.success(f"Using centroid of '{user_loc}': ({user_lat:.4f}, {user_lon:.4f})")
                else:
                    st.warning("City not found — using dataset center.")
                    user_lat = df['latitude'].mean(); user_lon = df['longitude'].mean()

        recs = recommend(df, user_lat, user_lon, req_power, max_price, prefer_dc, nearest_only, int(top_n))
        if recs.empty:
            st.warning("No recommendations found.")
        else:
            show_cols = ['name','vendor_name','address','city','capacity','capacity_kw','power_type','vehicle_type','vehicle_count','cost_per_unit','distance_km','final_score']
            show_cols = [c for c in show_cols if c in recs.columns]
            st.subheader("Top Recommendations")
            st.dataframe(recs[show_cols].reset_index(drop=True))

            # CSV download
            csv = recs[show_cols].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download recommendations as CSV", data=csv, file_name="ev_recommendations.csv", mime="text/csv")

            # folium map
            st.subheader("Map of Recommendations")
            try:
                html_map = folium_map_html_from_df(recs, user_lat, user_lon, zoom_start=11)
                st.components.v1.html(html_map, height=500)
            except Exception as e:
                st.error("Map error: " + str(e))

# -------------------------  
# Dashboard (Interactive)  
# -------------------------  
elif page == "Dashboard":  
    st.title("📈 Interactive Dashboard")  

    st.subheader("⚡ Capacity vs Estimated Waiting Time")
    fig, ax = plt.subplots(figsize=(9,4))

    # Scatter
    ax.scatter(df['capacity_kw'], df['wait_time'], 
           c=df['is_dc'], cmap='coolwarm', alpha=0.7)

    ax.set_xlabel("Capacity (kW)")
    ax.set_ylabel("Estimated Waiting Time (minutes)")
    ax.set_title("Charging Station Capacity vs Waiting Time")
    ax.grid(True)

    # Legend for AC/DC
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='AC', markerfacecolor='blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='DC', markerfacecolor='red', markersize=10)]
    ax.legend(handles=legend_elements, title="Charger Type")

    st.pyplot(fig)


    # --- 2. AC vs DC Counts ---  
    st.subheader("🔌 AC vs DC Stations")  
    power_filter = st.sidebar.multiselect("Select Power Type to Display", options=['AC','DC'], default=['AC','DC'])  
    df_power = df.copy()  
    df_power['power_category'] = np.where(df_power['is_dc']==1, 'DC', 'AC')  
    df_filtered = df_power[df_power['power_category'].isin(power_filter)]  
    counts = df_filtered['power_category'].value_counts()  

    fig2, ax2 = plt.subplots(figsize=(5,3))  
    bars = ax2.bar(counts.index.astype(str), counts.values, color=['orange','green'])  
    ax2.set_ylabel("Count")  
    ax2.set_title("Filtered AC/DC Station Counts")  
    for bar in bars:  
        height = bar.get_height()  
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height}', ha='center', va='bottom')  
    st.pyplot(fig2)  

    # --- 3. Top N Cities by Average Capacity ---  
    st.subheader("📊 Top Cities by Average Capacity")  
    top_n = st.sidebar.slider("Select Top N cities", min_value=5, max_value=50, value=10)  
    city_capacity = df.groupby('city', dropna=True)['capacity_kw'].mean().reset_index()  
    city_capacity = city_capacity.sort_values('capacity_kw', ascending=False).head(top_n)  

    fig3, ax3 = plt.subplots(figsize=(10,5))  
    bars = ax3.bar(city_capacity['city'].astype(str), city_capacity['capacity_kw'], color='skyblue')  
    ax3.set_ylabel("Average Capacity (kW)")  
    ax3.set_xlabel("City")  
    ax3.set_title(f"Top {top_n} Cities by Average Capacity")  
    ax3.set_xticklabels(city_capacity['city'].astype(str), rotation=45, ha='right')  
    for bar in bars:  
        height = bar.get_height()  
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', ha='center', va='bottom')  
    st.pyplot(fig3)  


 

    # --- 5. All Stations on Map ---  
    st.subheader("🗺️ Map of Stations")  
    try:  
        center_lat = df['latitude'].mean()  
        center_lon = df['longitude'].mean()  
        html_all = folium_map_html_from_df(df.sample(min(len(df),300)), center_lat, center_lon, zoom_start=5)  
        st.components.v1.html(html_all, height=500)  
    except Exception as e:  
        st.error("Map error: " + str(e))  



st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ — ABT")
