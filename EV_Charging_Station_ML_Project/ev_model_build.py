"""
ev_project.py

Single-file EV project (Spyder friendly).

- Place ev_final.csv in the same folder.
- Run the file in Spyder. It will:
    1) Clean & preprocess data
    2) Train a Linear Regression model to predict cost_per_unit
    3) Save model and scaler
    4) Provide an interactive CLI recommendation system (city name or "lat,lon")
"""

import pandas as pd
import numpy as np
import re
from math import radians, sin, cos, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Utility functions
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except:
        return np.nan
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2.0)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.0)**2
    c = 2 * asin(sqrt(a))
    R = 6371.0
    return R * c

def safe_parse_list(cell):
    """Parse vehicle_type string into list without using unsafe eval."""
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return cell
    s = str(cell).strip()
    # If it looks like python list, extract items inside
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        if inner == "":
            return []
        parts = re.split(r''',\s*(?![^()]*\))''', inner)
        cleaned = [p.strip().strip("'\"") for p in parts if p.strip()]
        return cleaned
    # fallback: comma separated
    if ',' in s:
        return [x.strip().strip("'\"") for x in s.split(',') if x.strip()]
    # single token
    return [s] if s else []

def extract_kw(x):
    """Extract numeric kW from capacity textual field, else NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r'(\d+\.?\d*)', s)
    if m:
        try:
            return float(m.group(1))
        except:
            return np.nan
    return np.nan

# -------------------------
# 1) Load & initial cleaning
# -------------------------
print("Loading dataset 'ev_final.csv' ...")
df = pd.read_csv(r"C:\Users\Prisha D\Desktop\DEPLOY\ev_final.csv")

# Remove exact duplicates (keep first)
if 'uid' in df.columns:
    df = df.drop_duplicates(subset='uid')
else:
    df = df.drop_duplicates()

# Ensure basic columns exist
required = ['latitude','longitude','capacity','vehicle_type','power_type','cost_per_unit','city','open','close']
for c in required:
    if c not in df.columns:
        df[c] = np.nan

# Numeric cost_per_unit
df['cost_per_unit'] = pd.to_numeric(df['cost_per_unit'], errors='coerce')

# If all costs are NaN (unlikely), set zeros to avoid failure
if df['cost_per_unit'].isna().all():
    df['cost_per_unit'] = 0.0

# Fill cost_per_unit NaN with median (robust)
df['cost_per_unit'] = df['cost_per_unit'].fillna(df['cost_per_unit'].median())

# capacity numeric
df['capacity_kw'] = df['capacity'].apply(extract_kw)
# fill capacity with median
if df['capacity_kw'].isna().all():
    df['capacity_kw'] = df['capacity_kw'].fillna(0.0)
else:
    df['capacity_kw'] = df['capacity_kw'].fillna(df['capacity_kw'].median())

# vehicle list & count
df['vehicle_list'] = df['vehicle_type'].apply(safe_parse_list)
df['vehicle_count'] = df['vehicle_list'].apply(len)

# is_dc flag
df['is_dc'] = df['power_type'].astype(str).str.upper().str.contains('DC', na=False).astype(int)

# convert lat/lon to numeric and drop rows without coords (we need coords for recommendations)
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df = df.dropna(subset=['latitude','longitude']).reset_index(drop=True)

# parse times to compute open_hours safely (some formats may be invalid)
def safe_time_diff_hours(o, c):
    try:
        o_pd = pd.to_datetime(o, errors='coerce')
        c_pd = pd.to_datetime(c, errors='coerce')
        if pd.isna(o_pd) or pd.isna(c_pd):
            return np.nan
        diff = c_pd - o_pd
        # if diff negative (closing next day), add 24h
        secs = diff.total_seconds()
        if secs < 0:
            secs += 24*3600
        return secs / 3600.0
    except:
        return np.nan

df['open_hours'] = df.apply(lambda r: safe_time_diff_hours(r.get('open'), r.get('close')), axis=1)
df['open_hours'] = df['open_hours'].fillna(df['open_hours'].median() if df['open_hours'].notna().any() else 24.0)

# final numeric columns ensured
for col in ['capacity_kw','vehicle_count','is_dc','open_hours','cost_per_unit']:
    if col not in df.columns:
        df[col] = 0.0
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

print("Data loaded and preprocessed. Rows:", len(df))

# -------------------------
# 2) Train linear regression to predict cost_per_unit
#    (this is optional but included as requested)
# -------------------------
print("\nTraining Linear Regression to predict cost_per_unit ...")

# Choose simple numeric features suitable for Spyder single-file:
features = ['capacity_kw', 'vehicle_count', 'is_dc', 'open_hours']

X = df[features].copy()
y = df['cost_per_unit'].copy()

# Safety: ensure no NaN in y
y = y.fillna(y.median())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# train
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# evaluate
y_pred = lr.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Linear Regression trained. R2 = {r2:.4f}, MSE = {mse:.4f}")

# Save model and scaler
joblib.dump(lr, "ev_cost_model.pkl")
joblib.dump(scaler, "ev_scaler.pkl")
print("Saved model -> ev_cost_model.pkl and scaler -> ev_scaler.pkl")

# -------------------------
# 3) Recommendation system (city name or lat,lon input)
# -------------------------
# We'll compute a combined score using normalized components:
# - distance (closer better)
# - capacity match (>=required gets full)
# - price (lower better)
# - power type match (DC preference)
# - vehicle_count advantage

# Precompute some columns for efficiency
df['capacity_match_dummy'] = df['capacity_kw']  # will be used in ratio
# For price normalization later, we'll use dataset max/min within filtered candidates

def recommend_stations(user_location, required_power_kw=15.0, max_price=float('inf'),
                       prefer_dc=None, top_n=10, city_fallback_use_dataset=False):
    """
    user_location: "CityName" OR "lat,lon" OR tuple (lat,lon)
    required_power_kw: float
    max_price: float (max allowed cost_per_unit)
    prefer_dc: None / True / False
    top_n: int
    """
    loc = user_location
    # Determine user coords or city
    user_lat = None
    user_lon = None
    city_mode = False
    user_city = None

    if isinstance(loc, (list, tuple)) and len(loc) == 2:
        try:
            user_lat = float(loc[0]); user_lon = float(loc[1])
        except:
            user_lat = user_lon = None
    else:
        s = str(loc).strip()
        if s == "":
            # fallback to first row city center
            user_city = df['city'].fillna('').iloc[0]
            city_mode = True
        elif ',' in s and not any(c.isalpha() for c in s.replace(' ', '')):
            # looks like coordinates as "lat,lon"
            parts = s.split(',')
            try:
                user_lat = float(parts[0].strip()); user_lon = float(parts[1].strip())
            except:
                city_mode = True
                user_city = s.lower()
        else:
            city_mode = True
            user_city = s.lower()

    candidates = df.copy()

    # If city_mode and city exists in dataset, narrow to that city (but if none, leave all)
    if city_mode and user_city:
        matched = candidates[candidates['city'].fillna('').str.lower() == user_city]
        if len(matched) > 0:
            candidates = matched.copy()
            # set user_lat/lon to city mean for distance calc
            user_lat = matched['latitude'].mean()
            user_lon = matched['longitude'].mean()
            # print small message outside function if desired
        else:
            # city not found: keep all candidates but inform user
            # (distance will be computed if user provided coordinates; else distance component ignored)
            pass

    # Filter by max_price strictly (if max_price is finite)
    if np.isfinite(max_price):
        tmp = candidates[candidates['cost_per_unit'] <= max_price]
        if len(tmp) > 0:
            candidates = tmp.copy()
        else:
            # if no stations meet the strict price, relax to full set but keep user informed
            candidates = candidates.copy()

    # Compute distance score if user coords available
    if (user_lat is not None) and (user_lon is not None):
        candidates['distance_km'] = candidates.apply(
            lambda r: haversine(user_lat, user_lon, r['latitude'], r['longitude']), axis=1)
        # transform to score: smaller distance better => dist_score in [0,1] with 1 best
        # we normalize by a reasonable scale (e.g., max distance within candidates)
        maxd = candidates['distance_km'].replace([np.inf, np.nan], np.nan).max()
        if pd.isna(maxd) or maxd == 0:
            candidates['dist_score'] = 1.0
        else:
            # invert proportionally
            candidates['dist_score'] = 1.0 - (candidates['distance_km'] / (maxd + 1e-9))
            candidates['dist_score'] = candidates['dist_score'].clip(0.0, 1.0)
    else:
        # neutral distance score
        candidates['distance_km'] = np.nan
        candidates['dist_score'] = 0.5

    # Capacity match: ratio capped at 1.0
    def cap_match_ratio(cap_kw):
        try:
            return min(float(cap_kw) / max(1e-6, float(required_power_kw)), 1.0)
        except:
            return 0.0
    candidates['capacity_match'] = candidates['capacity_kw'].apply(cap_match_ratio)

    # Price score: lower price better => normalized inversely within candidates
    min_price = candidates['cost_per_unit'].min() if len(candidates) > 0 else 0.0
    max_price_cand = candidates['cost_per_unit'].max() if len(candidates) > 0 else 1.0
    if max_price_cand - min_price < 1e-6:
        candidates['price_score'] = 0.5
    else:
        candidates['price_score'] = 1.0 - ((candidates['cost_per_unit'] - min_price) / (max_price_cand - min_price))
        candidates['price_score'] = candidates['price_score'].clip(0.0, 1.0)

    # Power type preference
    if prefer_dc is None:
        candidates['power_pref'] = 0.7  # neutral
    else:
        if prefer_dc:
            candidates['power_pref'] = candidates['is_dc'].apply(lambda v: 1.0 if v==1 else 0.2)
        else:
            candidates['power_pref'] = candidates['is_dc'].apply(lambda v: 1.0 if v==0 else 0.2)

    # Vehicle count normalized
    max_vehicle = candidates['vehicle_count'].max() if len(candidates)>0 else 1
    if max_vehicle == 0:
        candidates['vehicle_score'] = 0.0
    else:
        candidates['vehicle_score'] = candidates['vehicle_count'] / (max_vehicle + 1e-9)

    # Combine into final score (weights can be tuned)
    # weights: distance 30%, capacity 30%, price 20%, power_pref 10%, vehicle_score 10%
    candidates['final_score'] = (
        0.30 * candidates['dist_score'] +
        0.30 * candidates['capacity_match'] +
        0.20 * candidates['price_score'] +
        0.10 * candidates['power_pref'] +
        0.10 * candidates['vehicle_score']
    )

    # slight boost if capacity >= required
    candidates['final_score'] += candidates['capacity_kw'].apply(lambda c: 0.02 if c >= required_power_kw else 0.0)

    # sort and return top_n
    candidates_sorted = candidates.sort_values(by='final_score', ascending=False).reset_index(drop=True)
    return candidates_sorted.head(top_n)

# -------------------------
# 4) Interactive CLI wrapper
# -------------------------
def run_cli():
    print("\nEV Station Recommendation (CLI)\n")
    print("Enter location as either a city name (e.g., 'Delhi') or coordinates 'lat,lon' (e.g., '28.6448,77.2167').")
    loc = input("Location (city or lat,lon) [default: first city in dataset]: ").strip()
    if loc == "":
        loc = df['city'].fillna('').iloc[0]
        print("Using default location:", loc)

    try:
        req_power = float(input("Required power (kW) [default 15]: ").strip() or 15.0)
    except:
        req_power = 15.0
    try:
        max_price_in = input("Max price per unit (enter number or leave blank for no limit): ").strip()
        max_price = float(max_price_in) if max_price_in != "" else float('inf')
    except:
        max_price = float('inf')
    pref = input("Prefer DC? (yes/no/skip) [skip]: ").strip().lower()
    if pref in ['yes','y']:
        prefer_dc = True
    elif pref in ['no','n']:
        prefer_dc = False
    else:
        prefer_dc = None
    try:
        top_n = int(input("How many results to show [default 10]: ").strip() or 10)
    except:
        top_n = 10

    print("\nFinding recommendations ...\n")
    recs = recommend_stations(loc, required_power_kw=req_power, max_price=max_price,
                              prefer_dc=prefer_dc, top_n=top_n)

    if recs.empty:
        print("No recommendations found. Try changing your filters.")
        return

    # Print readable output
    for i, r in recs.iterrows():
        print(f"--- Rank {i+1}  Score: {r['final_score']:.4f} ---")
        print(f"Name       : {r.get('name','')}")
        print(f"Vendor     : {r.get('vendor_name','')}")
        print(f"City       : {r.get('city','')}")
        print(f"Address    : {r.get('address','')}")
        print(f"Capacity   : {r.get('capacity','')}  [{r.get('capacity_kw',np.nan)} kW]")
        print(f"Power type : {r.get('power_type','')}")
        print(f"Vehicles   : {r.get('vehicle_type','')}")
        print(f"Price/unit : {r.get('cost_per_unit',np.nan)}")
        dist = r.get('distance_km', np.nan)
        if not pd.isna(dist):
            print(f"Distance   : {dist:.3f} km")
        else:
            print("Distance   : N/A")
        print()

# -------------------------
# 5) Optional quick demo predictions
# -------------------------
def demo_predict_cost_example():
    # load model and scaler (we have them in memory as lr and scaler)
    sample = pd.DataFrame([[15.0, 1.0, 1.0, 24.0]], columns=['capacity_kw','vehicle_count','is_dc','open_hours'])
    sample_scaled = scaler.transform(sample)
    pred = lr.predict(sample_scaled)[0]
    print(f"Example predicted cost_per_unit for sample {sample.iloc[0].to_dict()} -> ₹ {pred:.2f}")

# -------------------------
# Main interactive entry
# -------------------------
if __name__ == "__main__":
    print("\n--------- EV PROJECT (single-file) ---------")
    print("Options:")
    print("1. Run Recommendation CLI")
    print("2. Demo cost prediction (simple example)")
    print("3. Exit")
    choice = input("Choose [1/2/3] (default 1): ").strip() or "1"
    if choice == "1":
        run_cli()
    elif choice == "2":
        demo_predict_cost_example()
    else:
        print("Exiting. You can import functions from this file for further use.")
