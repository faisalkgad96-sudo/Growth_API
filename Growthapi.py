import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import altair as alt
import time
import pickle
import os
from io import BytesIO
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
from github import Github

st.set_page_config(page_title="Master Dashboard v10.6", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

# ============================================================================
# 1. CONFIGURATION & STATE
# ============================================================================
OPERATIONAL_AREAS = ["Maadi", "Alexandria", "Downtown 1", "Zahraa El Maadi", "Masr El Gedida"]
TIME_OFFSET = "+02:00"
MASTER_FILE_NAME = "master_state.pkl"

# Initialize Session State
if 'master_s_df' not in st.session_state: st.session_state.master_s_df = pd.DataFrame()
if 'master_r_df' not in st.session_state: st.session_state.master_r_df = pd.DataFrame()
if 'master_l_df' not in st.session_state: st.session_state.master_l_df = pd.DataFrame()
if 'master_a_df' not in st.session_state: st.session_state.master_a_df = pd.DataFrame()
if 'last_update_time' not in st.session_state: st.session_state.last_update_time = None
if 'data_initialized' not in st.session_state: st.session_state.data_initialized = False

# ============================================================================
# 2. GITHUB CLOUD SYNC
# ============================================================================
def push_to_github(data_obj, commit_msg):
    """Pushes the updated master state file back to GitHub."""
    try:
        # Retrieve Secrets
        token = st.secrets["GITHUB_TOKEN"]
        repo_name = st.secrets["REPO_NAME"]
        
        g = Github(token)
        repo = g.get_repo(repo_name)
        
        # Serialize Data
        buffer = BytesIO()
        pickle.dump(data_obj, buffer)
        content = buffer.getvalue()
        
        # Check if file exists to Update or Create
        try:
            contents = repo.get_contents(MASTER_FILE_NAME)
            repo.update_file(contents.path, commit_msg, content, contents.sha)
            return True, "Updated existing Master File."
        except:
            repo.create_file(MASTER_FILE_NAME, commit_msg, content)
            return True, "Created new Master File."
            
    except Exception as e:
        return False, str(e)

# ============================================================================
# 3. HELPERS
# ============================================================================
def safe_dedupe(df, target_col):
    if df.empty: return df
    if target_col in df.columns:
        df = df.drop_duplicates(subset=target_col, keep='last')
    else:
        col_map = {c.lower(): c for c in df.columns}
        if target_col.lower() in col_map:
            df = df.drop_duplicates(subset=col_map[target_col.lower()], keep='last')
        else:
            df = df.drop_duplicates()
    return df.reset_index(drop=True)

@st.cache_data(ttl=3600)
def load_static_files(points_files, fleet_file):
    points_df = pd.DataFrame()
    if points_files:
        dfs = []
        for f in points_files:
            try:
                if f.name.endswith('.csv'): d = pd.read_csv(f)
                else: d = pd.read_excel(f)
                d.columns = d.columns.str.lower().str.strip()
                dfs.append(d)
            except: pass
        if dfs: points_df = pd.concat(dfs, ignore_index=True)

    fleet_df = pd.DataFrame()
    if fleet_file:
        try:
            if fleet_file.name.endswith('.csv'): d = pd.read_csv(fleet_file)
            else: d = pd.read_excel(fleet_file)
            d.columns = d.columns.str.lower().str.strip()
            c_col = next((c for c in d.columns if c in ['code', 'vehicle id', 'scooter']), None)
            a_col = next((c for c in d.columns if c in ['area', 'assigned area']), None)
            if c_col and a_col:
                d = d.rename(columns={c_col: 'scooter_code', a_col: 'assigned_area'})
                d['scooter_code'] = d['scooter_code'].astype(str).str.strip().str.lower()
                d['assigned_area'] = d['assigned_area'].astype(str).str.strip()
                fleet_df = d[d['assigned_area'].isin(OPERATIONAL_AREAS)].copy()
        except: pass
    return points_df, fleet_df

def fetch_chunk(base, head, payload):
    try:
        r = requests.post(base, headers=head, json=payload, timeout=45)
        if r.status_code == 200:
            df = pd.read_excel(BytesIO(r.content))
            if not df.empty: df.columns = df.columns.str.lower().str.strip()
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def fetch_api_data_batched(token, start_dt, end_dt):
    base = "https://dashboard.rabbit-api.app/export"
    head = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    date_ranges = []
    curr = start_dt
    while curr <= end_dt:
        nxt = min(curr + timedelta(days=6), end_dt)
        date_ranges.append((curr, nxt))
        curr = nxt + timedelta(days=1)

    prog_bar = st.progress(0)
    status_text = st.empty()
    total_batches = len(date_ranges)
    all_s, all_r, all_l, all_a = [], [], [], []
    
    for idx, (s, e) in enumerate(date_ranges):
        s_str = f"{s}T00:00:00{TIME_OFFSET}"
        e_str = f"{e}T23:59:59{TIME_OFFSET}"
        pct = idx / total_batches
        prog_bar.progress(pct)
        status_text.markdown(f"**📥 Fetching Batch {idx+1}/{total_batches}:** {s} to {e}...")

        p_sess = {"module": "AppSessions", "format": "excel", "filters": json.dumps({"startDate": s_str, "endDate": e_str, "lat": None, "lng": None, "radius": None, "inRide": "all", "error": "", "areas": []})}
        all_s.append(fetch_chunk(base, head, p_sess))

        r_fields = "id,userId,area,userSignupDateLocal,scooter,duration,totalCost,actualCost,paidWithBalance,paidWithPromocode,paidWithSubscription,start_date_local,end_date_local,startLat,startLong,stopLat,stopLong,rating"
        p_ride = {"module": "RidesSucess", "format": "excel", "fields": r_fields, "filters": json.dumps({"startDate": s_str, "endDate": e_str})}
        all_r.append(fetch_chunk(base, head, p_ride))

        p_logs = {"module": "VehicleLogs", "format": "excel", "filters": json.dumps({"startDate": s_str, "endDate": e_str, "isOperating": "All", "logCase": "All"})}
        all_l.append(fetch_chunk(base, head, p_logs))

        p_attend = {"module": "Attendance", "format": "excel", "filters": json.dumps({"startDate": s_str, "endDate": e_str})}
        all_a.append(fetch_chunk(base, head, p_attend))

    prog_bar.progress(1.0)
    status_text.success("✅ Fetch Complete! Merging...")
    time.sleep(0.5)
    status_text.empty()
    prog_bar.empty()
    
    final_s = pd.concat(all_s, ignore_index=True) if all_s else pd.DataFrame()
    final_r = pd.concat(all_r, ignore_index=True) if all_r else pd.DataFrame()
    final_l = pd.concat(all_l, ignore_index=True) if all_l else pd.DataFrame()
    final_a = pd.concat(all_a, ignore_index=True) if all_a else pd.DataFrame()
    
    return final_s, final_r, final_l, final_a, None

# ============================================================================
# 4. PROCESSING ENGINES
# ============================================================================
def build_tree_match(df, points_df, radius, lat_col, lon_col, prefix=None):
    if df.empty or points_df is None or points_df.empty: return df
    df = df.reset_index(drop=True)
    df.columns = df.columns.str.lower().str.strip()
    lat_col = lat_col.lower()
    lon_col = lon_col.lower()
    cols = points_df.columns
    p_lat = next((c for c in ['lat', 'latitude'] if c in cols), None)
    p_lon = next((c for c in ['lng', 'lon', 'longitude'] if c in cols), None)
    if not p_lat: return df
    if lat_col not in df.columns or lon_col not in df.columns: return df
    valid = df.dropna(subset=[lat_col, lon_col]).copy()
    valid = valid[(valid[lat_col]!=0) & (valid[lon_col]!=0)]
    if valid.empty: return df
    def to_cart(lat, lon):
        rad_lat, rad_lon = np.radians(lat), np.radians(lon)
        a = 6371000.0
        x = a * np.cos(rad_lat) * np.cos(rad_lon)
        y = a * np.cos(rad_lat) * np.sin(rad_lon)
        z = a * np.sin(rad_lat)
        return np.column_stack([x, y, z])
    p_cart = to_cart(points_df[p_lat].values, points_df[p_lon].values)
    t_cart = to_cart(valid[lat_col].values, valid[lon_col].values)
    tree = cKDTree(p_cart)
    dists, idxs = tree.query(t_cart, k=1)
    mask = dists <= radius
    matched_idxs = idxs[mask]
    p_area = next((c for c in ['area'] if c in cols), None)
    p_neigh = next((c for c in ['neighborhood', 'neighbourhood', 'point name', 'name'] if c in cols), None)
    if prefix:
        col_name = f'{prefix.lower()} neighborhood'
        df[col_name] = 'Unknown'
        if p_neigh: valid.loc[mask, col_name] = points_df.iloc[matched_idxs][p_neigh].values
    else:
        df['assigned area'] = 'Out of Fence'
        if p_area: valid.loc[mask, 'assigned area'] = points_df.iloc[matched_idxs][p_area].values
    df.update(valid)
    return df

@st.cache_data(show_spinner=False)
def process_data(df_s, df_r, points_df, radius):
    if df_s is None: df_s = pd.DataFrame()
    if df_r is None: df_r = pd.DataFrame()
    if not df_s.empty:
        in_ride_col = next((c for c in df_s.columns if c == 'in ride'), None)
        if in_ride_col: df_s = df_s[df_s[in_ride_col].astype(str).str.strip().str.lower() != 'yes']
        df_s['assigned area'] = 'Out of Fence'
        if points_df is not None and not points_df.empty: 
            s_lat = next((c for c in df_s.columns if c in ['lat', 'latitude']), 'lat')
            s_lon = next((c for c in df_s.columns if c in ['lng', 'longitude', 'long']), 'lng')
            df_s = build_tree_match(df_s, points_df, radius, s_lat, s_lon)
    if not df_r.empty and points_df is not None and not points_df.empty:
        r_start_lat = next((c for c in df_r.columns if 'start' in c and 'lat' in c), 'startlat')
        r_start_lon = next((c for c in df_r.columns if 'start' in c and ('lon' in c or 'lng' in c)), 'startlong')
        r_stop_lat = next((c for c in df_r.columns if ('stop' in c or 'end' in c) and 'lat' in c), 'stoplat')
        r_stop_lon = next((c for c in df_r.columns if ('stop' in c or 'end' in c) and ('lon' in c or 'lng' in c)), 'stoplong')
        df_r = build_tree_match(df_r, points_df, radius, r_start_lat, r_start_lon, 'Start')
        df_r = build_tree_match(df_r, points_df, radius, r_stop_lat, r_stop_lon, 'End')
    return df_s, df_r

@st.cache_data(show_spinner=False)
def process_supply_and_urgent(logs_df, fleet_df, s_date, e_date):
    if fleet_df is None or fleet_df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if logs_df is None or logs_df.empty: return fleet_df, pd.DataFrame(), pd.DataFrame()
    v_col = next((c for c in logs_df.columns if c in ['vehicle', 'scooter', 'code']), 'vehicle')
    op_col = next((c for c in logs_df.columns if 'operating' in c), 'operating')
    act_col = next((c for c in logs_df.columns if c == 'action'), 'action')
    date_col = next((c for c in logs_df.columns if 'local' in c and 'date' in c), 'date (local)')
    reason_col = next((c for c in logs_df.columns if 'urgent' in c and 'reason' in c), 'urgent reason')
    logs_df['clean_code'] = logs_df[v_col].astype(str).str.strip().str.lower()
    active_logs = logs_df[logs_df[op_col].astype(str).str.lower() == 'yes']
    active_codes = active_logs['clean_code'].unique()
    fleet_df['Is Active'] = fleet_df['scooter_code'].isin(active_codes)
    urgent_logs = logs_df[logs_df[act_col].isin(['ENTERED_URGENT', 'EXITED_URGENT'])].copy()
    urgent_final = pd.DataFrame()
    if not urgent_logs.empty:
        urgent_logs[date_col] = pd.to_datetime(urgent_logs[date_col], errors='coerce')
        urgent_logs = urgent_logs.sort_values(date_col)
        last_status = urgent_logs.drop_duplicates(subset=['clean_code'], keep='last')
        current_urgent = last_status[last_status[act_col] == 'ENTERED_URGENT'].copy()
        urgent_breakdown = pd.merge(current_urgent, fleet_df, left_on='clean_code', right_on='scooter_code', how='inner')
        urgent_breakdown['Urgent Reason'] = urgent_breakdown[reason_col].fillna("Unknown")
        urgent_final = urgent_breakdown[['scooter_code', 'assigned_area', 'Urgent Reason', date_col]]
    logs_with_area = pd.merge(logs_df, fleet_df, left_on='clean_code', right_on='scooter_code', how='inner')
    logs_with_area['Log_Date'] = pd.to_datetime(logs_with_area[date_col]).dt.date
    logs_with_area['Hour_Bucket'] = pd.to_datetime(logs_with_area[date_col]).dt.floor('H')
    daily_active_logs = logs_with_area[logs_with_area[op_col].astype(str).str.lower() == 'yes']
    daily_active_count = daily_active_logs.groupby(['Log_Date', 'assigned_area'])['clean_code'].nunique().reset_index()
    daily_active_count.rename(columns={'clean_code': 'Daily_Active_Baseline', 'assigned_area': 'Area'}, inplace=True)
    start_dt = pd.to_datetime(s_date)
    end_dt = pd.to_datetime(e_date) + timedelta(days=1) - timedelta(seconds=1)
    hourly_range = pd.date_range(start=start_dt, end=end_dt, freq='H')
    master_trend = pd.MultiIndex.from_product([hourly_range, OPERATIONAL_AREAS], names=['Hour_Bucket', 'Area']).to_frame(index=False)
    master_trend['Log_Date'] = master_trend['Hour_Bucket'].dt.date
    master_trend = pd.merge(master_trend, daily_active_count, on=['Log_Date', 'Area'], how='left').fillna(0)
    urgent_in_bucket = logs_with_area[(logs_with_area[act_col] == 'ENTERED_URGENT') | (logs_with_area[reason_col].notna())]
    urgent_trend = urgent_in_bucket.groupby(['Hour_Bucket', 'assigned_area'])['clean_code'].nunique().reset_index()
    urgent_trend.rename(columns={'clean_code': 'Urgent_Count', 'assigned_area': 'Area'}, inplace=True)
    master_trend = pd.merge(master_trend, urgent_trend, on=['Hour_Bucket', 'Area'], how='left').fillna(0)
    master_trend['Net_Available'] = (master_trend['Daily_Active_Baseline'] - master_trend['Urgent_Count']).clip(lower=0)
    return fleet_df, urgent_final, master_trend

@st.cache_data(show_spinner=False)
def process_ride_utilization_metrics(rides_df, logs_df, fleet_df, s_date, e_date):
    if fleet_df is None or fleet_df.empty: return pd.DataFrame()
    if logs_df is None or logs_df.empty: return pd.DataFrame()
    v_col_l = next((c for c in logs_df.columns if c in ['vehicle', 'scooter', 'code']), 'vehicle')
    op_col_l = next((c for c in logs_df.columns if 'operating' in c), 'operating')
    date_col_l = next((c for c in logs_df.columns if 'local' in c and 'date' in c), 'date (local)')
    if not rides_df.empty:
        r_date_col = next((c for c in rides_df.columns if 'start' in c and 'local' in c), 'start_date_local')
        r_scooter_col = next((c for c in rides_df.columns if c in ['scooter', 'code', 'vehicle id']), 'scooter')
        rides_df['Ride_Date'] = pd.to_datetime(rides_df[r_date_col]).dt.date
    logs_df['clean_code'] = logs_df[v_col_l].astype(str).str.strip().str.lower()
    logs_with_area = pd.merge(logs_df, fleet_df, left_on='clean_code', right_on='scooter_code', how='inner')
    logs_with_area['Log_Date'] = pd.to_datetime(logs_with_area[date_col_l]).dt.date
    active_population = logs_with_area[logs_with_area[op_col_l].astype(str).str.lower() == 'yes'][['Log_Date', 'assigned_area', 'clean_code']].drop_duplicates()
    active_population.columns = ['Date', 'Area', 'Scooter']
    if not rides_df.empty:
        daily_ride_counts = rides_df.groupby(['Ride_Date', r_scooter_col]).size().reset_index(name='Ride_Count')
        daily_ride_counts.columns = ['Date', 'Scooter', 'Ride_Count']
        merged = pd.merge(active_population, daily_ride_counts, on=['Date', 'Scooter'], how='left')
        merged['Ride_Count'] = merged['Ride_Count'].fillna(0)
    else:
        merged = active_population.copy()
        merged['Ride_Count'] = 0
    count_0 = merged[merged['Ride_Count'] == 0].groupby(['Date', 'Area'])['Scooter'].nunique().reset_index(name='Exact_0_Rides')
    count_1 = merged[merged['Ride_Count'] == 1].groupby(['Date', 'Area'])['Scooter'].nunique().reset_index(name='Exact_1_Ride')
    count_2_plus = merged[merged['Ride_Count'] >= 2].groupby(['Date', 'Area'])['Scooter'].nunique().reset_index(name='Plus_2_Rides')
    start_dt = pd.to_datetime(s_date).date()
    end_dt = pd.to_datetime(e_date).date()
    date_range = pd.date_range(start_dt, end_dt).date
    skeleton = pd.MultiIndex.from_product([date_range, OPERATIONAL_AREAS], names=['Date', 'Area']).to_frame(index=False)
    final_util = pd.merge(skeleton, count_0, on=['Date', 'Area'], how='left').fillna(0)
    final_util = pd.merge(final_util, count_1, on=['Date', 'Area'], how='left').fillna(0)
    final_util = pd.merge(final_util, count_2_plus, on=['Date', 'Area'], how='left').fillna(0)
    return final_util

@st.cache_data(show_spinner=False)
def process_attendance(df_att, s_date, e_date):
    if df_att is None or df_att.empty: return pd.DataFrame()
    name_col = next((c for c in df_att.columns if 'name' in c), 'name')
    area_col = next((c for c in df_att.columns if 'area' in c), 'area')
    shift_col = next((c for c in df_att.columns if 'shift' in c), 'shift')
    swap_col = next((c for c in df_att.columns if 'swap' in c and 'count' in c), 'battery swap count')
    date_col = next((c for c in df_att.columns if 'check-in date' in c and 'local' in c), 'check-in date (local)')
    df_att['clean_name'] = df_att[name_col].astype(str).str.lower()
    df_att['clean_shift'] = df_att[shift_col].astype(str).str.lower()
    try:
        df_att['Att_Date'] = pd.to_datetime(df_att[date_col], dayfirst=True, errors='coerce').dt.date
    except:
        df_att['Att_Date'] = pd.to_datetime(df_att[date_col], errors='coerce').dt.date
    valid_att = df_att[(~df_att['clean_name'].str.contains("tech", na=False)) & (~df_att['clean_shift'].str.contains("supervisor", na=False))].copy()
    att_metrics = valid_att.groupby(['Att_Date', area_col]).agg(Active_Shifts=(name_col, 'count'), Total_Swaps=(swap_col, 'sum')).reset_index()
    att_metrics.rename(columns={area_col: 'Area', 'Att_Date': 'Date'}, inplace=True)
    att_metrics['Productivity'] = np.where(att_metrics['Active_Shifts']>0, att_metrics['Total_Swaps']/att_metrics['Active_Shifts'], 0)
    start_dt = pd.to_datetime(s_date).date()
    end_dt = pd.to_datetime(e_date).date()
    date_range = pd.date_range(start_dt, end_dt).date
    skeleton = pd.MultiIndex.from_product([date_range, OPERATIONAL_AREAS], names=['Date', 'Area']).to_frame(index=False)
    final_att = pd.merge(skeleton, att_metrics, on=['Date', 'Area'], how='left').fillna(0)
    return final_att

def get_time_bucket(df, time_col, interval):
    if df.empty: return df
    df[time_col] = pd.to_datetime(df[time_col])
    if interval == "Hourly": return df[time_col].dt.floor('H')
    elif interval == "Daily": return df[time_col].dt.date
    elif interval == "Monthly": return df[time_col].dt.to_period('M').dt.to_timestamp()
    return df[time_col]

# ============================================================================
# 4. MAIN LAYOUT & LOGIC
# ============================================================================

# 1. SIDEBAR & AUTO-LOAD
with st.sidebar:
    st.header("🛠️ Setup")
    api_token = st.text_input("1. API Token", type="password")
    p_files = st.file_uploader("2. Distribution Points", accept_multiple_files=True)
    f_file = st.file_uploader("3. Fleet File", type=['xlsx', 'csv'])
    radius = st.slider("Matching Radius (m)", 100, 1000, 200)
    
    # AUTO-LOAD FROM REPO
    if not st.session_state.data_initialized and os.path.exists(MASTER_FILE_NAME):
        try:
            with st.spinner("Found Master File in Cloud. Loading..."):
                state_data = pd.read_pickle(MASTER_FILE_NAME)
                st.session_state.master_s_df = state_data['sessions']
                st.session_state.master_r_df = state_data['rides']
                st.session_state.master_l_df = state_data['logs']
                st.session_state.master_a_df = state_data['attendance']
                st.session_state.last_update_time = state_data.get('timestamp', datetime.now())
                st.session_state.data_initialized = True
                st.success("✅ Cloud Data Loaded!")
        except Exception as e:
            st.error(f"Error loading Cloud File: {e}")

col_title, col_refresh = st.columns([3, 1])
with col_title:
    st.title("⚡ Master Dashboard v10.6")
with col_refresh:
    st.write("") 
    if st.session_state.last_update_time:
        st.caption(f"Last updated: {st.session_state.last_update_time.strftime('%Y-%m-%d %I:%M %p')}")
    
    # REFRESH LOGIC
    if st.button("🔄 Refresh New Data", type="secondary", use_container_width=True):
        if not api_token:
            st.error("Missing API Token!")
        elif not st.session_state.data_initialized:
            st.error("Launch First!")
        else:
            refresh_start = datetime.now().date() 
            if not st.session_state.master_s_df.empty:
                try:
                    time_col = next((c for c in st.session_state.master_s_df.columns if 'created' in c and 'local' in c), 'created at (local)')
                    refresh_start = pd.to_datetime(st.session_state.master_s_df[time_col]).max().date()
                except: pass
            refresh_end = datetime.now().date()
            
            new_s, new_r, new_l, new_a, err = fetch_api_data_batched(api_token, refresh_start, refresh_end)
            
            if err:
                st.error(f"Refresh failed: {err}")
            else:
                if not new_s.empty:
                    st.session_state.master_s_df = pd.concat([st.session_state.master_s_df, new_s])
                    st.session_state.master_s_df = safe_dedupe(st.session_state.master_s_df, 'id')
                if not new_r.empty:
                    st.session_state.master_r_df = pd.concat([st.session_state.master_r_df, new_r])
                    st.session_state.master_r_df = safe_dedupe(st.session_state.master_r_df, 'id')
                if not new_l.empty:
                    st.session_state.master_l_df = pd.concat([st.session_state.master_l_df, new_l])
                    st.session_state.master_l_df = safe_dedupe(st.session_state.master_l_df, 'uuid')
                if not new_a.empty:
                    st.session_state.master_a_df = pd.concat([st.session_state.master_a_df, new_a])
                    st.session_state.master_a_df = safe_dedupe(st.session_state.master_a_df, 'name')
                
                st.session_state.last_update_time = datetime.now()
                st.success("Data Refreshed!")
                st.rerun()

    # CLOUD PUSH BUTTON (Appears only if data exists)
    if st.session_state.data_initialized:
        if st.button("☁️ Push to Cloud (GitHub)", type="primary", use_container_width=True):
            if "GITHUB_TOKEN" not in st.secrets:
                st.error("No GitHub Token found in Secrets!")
            else:
                with st.spinner("Pushing Master File to GitHub..."):
                    save_obj = {
                        'sessions': st.session_state.master_s_df,
                        'rides': st.session_state.master_r_df,
                        'logs': st.session_state.master_l_df,
                        'attendance': st.session_state.master_a_df,
                        'timestamp': datetime.now()
                    }
                    success, msg = push_to_github(save_obj, f"Auto-update: {datetime.now()}")
                    if success:
                        st.success(f"Success! {msg}")
                    else:
                        st.error(f"Failed: {msg}")

# 2. DATE FILTERS (For API Launch or Visualization)
c_d1, c_d2, c_btn = st.columns([2, 2, 1])
s_date = c_d1.date_input("Start Date", datetime.now().date() - timedelta(days=7))
e_date = c_d2.date_input("End Date", datetime.now().date())

# 3. LAUNCH BUTTON (API LOAD)
if c_btn.button("🚀 Launch via API", type="secondary", use_container_width=True):
    if not api_token or not f_file:
        st.error("Missing Token or Fleet File!")
    else:
        points_df, fleet_df = load_static_files(p_files, f_file)
        raw_s, raw_r, raw_l, raw_a, err = fetch_api_data_batched(api_token, s_date, e_date)
        
        if err: st.error(err)
        else:
            st.session_state.master_s_df = raw_s.reset_index(drop=True)
            st.session_state.master_r_df = raw_r.reset_index(drop=True)
            st.session_state.master_l_df = raw_l.reset_index(drop=True)
            st.session_state.master_a_df = raw_a.reset_index(drop=True)
            st.session_state.data_initialized = True
            st.session_state.last_update_time = datetime.now()
            st.success("Analysis Complete!")

# 4. DASHBOARD RENDER
if st.session_state.data_initialized:
    if f_file: 
        points_df, fleet_df = load_static_files(p_files, f_file)
    
    # Visualization Filtering
    view_s = st.session_state.master_s_df
    view_r = st.session_state.master_r_df
    view_l = st.session_state.master_l_df
    view_a = st.session_state.master_a_df
    
    final_s, final_r = process_data(view_s.copy(), view_r.copy(), points_df, radius)
    final_fleet, final_urgent, final_trend = process_supply_and_urgent(view_l.copy(), fleet_df, s_date, e_date)
    final_util = process_ride_utilization_metrics(final_r, view_l.copy(), fleet_df, s_date, e_date)
    final_att = process_attendance(view_a.copy(), s_date, e_date)

    st.write("### 🔍 Global Filters")
    with st.container():
        f1, f2, f3 = st.columns(3)
        all_areas = []
        if not final_s.empty: all_areas.extend(final_s['assigned area'].unique())
        if not final_r.empty: 
            r_area_col = next((c for c in final_r.columns if c == 'area'), 'area')
            all_areas.extend(final_r[r_area_col].unique())
        all_areas = sorted(list(set(all_areas)))
        if 'Out of Fence' in all_areas: all_areas.remove('Out of Fence')
        sel_areas = f1.multiselect("Areas", all_areas, default=all_areas)
        sel_time = f3.selectbox("Time Grouping", ["Daily", "Hourly", "Monthly"])

    fs_df = final_s[final_s['assigned area'].isin(sel_areas)].copy() if not final_s.empty else pd.DataFrame()
    fr_df = final_r[final_r[r_area_col].isin(sel_areas)].copy() if not final_r.empty else pd.DataFrame()
    f_supply = final_fleet[final_fleet['assigned_area'].isin(sel_areas)].copy() if not final_fleet.empty else pd.DataFrame()
    f_urgent = final_urgent[final_urgent['assigned_area'].isin(sel_areas)].copy() if not final_urgent.empty else pd.DataFrame()
    f_trend = final_trend[final_trend['Area'].isin(sel_areas)].copy() if not final_trend.empty else pd.DataFrame()
    f_util = final_util[final_util['Area'].isin(sel_areas)].copy() if not final_util.empty else pd.DataFrame()
    f_att = final_att[final_att['Area'].isin(sel_areas)].copy() if not final_att.empty else pd.DataFrame()

    if not fs_df.empty:
        t_col = next((c for c in fs_df.columns if 'created' in c), None)
        if t_col: fs_df = fs_df[(pd.to_datetime(fs_df[t_col]).dt.date >= s_date) & (pd.to_datetime(fs_df[t_col]).dt.date <= e_date)]
    if not fr_df.empty:
        t_col = next((c for c in fr_df.columns if 'start' in c and 'local' in c), None)
        if t_col: fr_df = fr_df[(pd.to_datetime(fr_df[t_col]).dt.date >= s_date) & (pd.to_datetime(fr_df[t_col]).dt.date <= e_date)]

    tab_demand, tab_supply = st.tabs(["📈 Demand Side", "🛴 Supply Side"])

    # --- DEMAND ---
    with tab_demand:
        dm1, dm2, dm3, dm4 = st.columns(4)
        dm1.metric("Sessions", f"{len(fs_df):,}")
        dm2.metric("Rides", f"{len(fr_df):,}")
        dm3.metric("Missed Opps", f"{len(fs_df)-len(fr_df):,}")
        conv = (len(fr_df)/len(fs_df)*100) if len(fs_df)>0 else 0
        dm4.metric("Conversion", f"{conv:.1f}%")

        if not fs_df.empty and not fr_df.empty:
            st.write("#### Performance Trends")
            view_mode = st.radio("Select Chart View:", ["📈 Trendlines", "🔥 Heatmaps"], horizontal=True)
            
            s_time = next((c for c in fs_df.columns if 'created' in c and 'local' in c), 'created at (local)')
            r_time = next((c for c in fr_df.columns if 'start' in c and 'local' in c), 'start_date_local')

            fs_df['Bucket'] = get_time_bucket(fs_df, s_time, sel_time)
            fr_df['Bucket'] = get_time_bucket(fr_df, r_time, sel_time)

            s_grp = fs_df.groupby(['Bucket', 'assigned area']).size().reset_index(name='Sessions')
            s_grp.rename(columns={'assigned area': 'Area'}, inplace=True)
            r_grp = fr_df.groupby(['Bucket', r_area_col]).size().reset_index(name='Rides')
            r_grp.rename(columns={r_area_col: 'Area'}, inplace=True)
            
            trend = pd.merge(s_grp, r_grp, on=['Bucket', 'Area'], how='outer').fillna(0)
            trend['Missed Opps'] = trend['Sessions'] - trend['Rides']
            trend['Fulfillment'] = np.where(trend['Sessions']>0, trend['Rides']/trend['Sessions'], 0)
            trend['Bucket'] = pd.to_datetime(trend['Bucket'])

            if view_mode == "📈 Trendlines":
                def plot_trend(y_col, title, fmt=".0f"):
                    return alt.Chart(trend).mark_line(point=True).encode(
                        x=alt.X('Bucket', axis=alt.Axis(format='%b %d %H:%M' if sel_time=='Hourly' else '%b %d')),
                        y=alt.Y(y_col), color='Area', tooltip=['Bucket', 'Area', alt.Tooltip(y_col, format=fmt)]
                    ).properties(title=title, height=300).interactive()

                st.altair_chart(plot_trend('Sessions', 'Sessions'), use_container_width=True)
                st.altair_chart(plot_trend('Rides', 'Rides'), use_container_width=True)
                st.altair_chart(plot_trend('Missed Opps', 'Missed Opps'), use_container_width=True)
                st.altair_chart(plot_trend('Fulfillment', 'Fulfillment %', '.1%'), use_container_width=True)
            else:
                hm_data = trend.copy()
                if sel_time == 'Hourly':
                    hm_data['X_Axis'] = hm_data['Bucket'].dt.hour
                    x_title = "Hour of Day"
                    hm_data['Tooltip_X'] = hm_data['X_Axis'].astype(str) + ":00"
                elif sel_time == 'Daily':
                    hm_data['X_Axis'] = hm_data['Bucket'].dt.strftime('%b %d')
                    x_title = "Date"
                    hm_data['Tooltip_X'] = hm_data['X_Axis']
                else:
                    hm_data['X_Axis'] = hm_data['Bucket'].dt.strftime('%Y-%m')
                    x_title = "Month"
                    hm_data['Tooltip_X'] = hm_data['X_Axis']

                hm_agg = hm_data.groupby(['Area', 'X_Axis', 'Tooltip_X']).agg({'Sessions': 'sum','Rides': 'sum','Missed Opps': 'sum'}).reset_index()
                hm_agg['Fulfillment'] = np.where(hm_agg['Sessions']>0, hm_agg['Rides']/hm_agg['Sessions'], 0)

                def plot_heatmap(y_col, title, fmt=".0f"):
                    hm_agg[f'{y_col}_Norm'] = hm_agg.groupby('Area')[y_col].transform(lambda x: x / x.max() if x.max() > 0 else 0)
                    return alt.Chart(hm_agg).mark_rect().encode(
                        x=alt.X('X_Axis:O', title=x_title, sort=None),
                        y=alt.Y('Area:N', title='Area'),
                        color=alt.Color(f'{y_col}_Norm', legend=None, scale=alt.Scale(scheme='blues')),
                        tooltip=['Tooltip_X', 'Area', alt.Tooltip(y_col, format=fmt, title=y_col)]
                    ).properties(title=title, height=350).interactive()

                st.altair_chart(plot_heatmap('Sessions', 'Sessions Heatmap'), use_container_width=True)
                st.altair_chart(plot_heatmap('Rides', 'Rides Heatmap'), use_container_width=True)
                st.altair_chart(plot_heatmap('Missed Opps', 'Missed Opps Heatmap'), use_container_width=True)
                st.altair_chart(plot_heatmap('Fulfillment', 'Fulfillment Heatmap', '.1%'), use_container_width=True)

    # --- SUPPLY ---
    with tab_supply:
        if not f_supply.empty:
            total_fleet = len(f_supply)
            active_fleet = f_supply['Is Active'].sum()
            total_avail_hours = f_trend['Net_Available'].sum() if not f_trend.empty else 0
            hours_diff = (pd.to_datetime(e_date) - pd.to_datetime(s_date)).total_seconds() / 3600
            hours_diff = max(hours_diff, 24)
            avg_avail_fleet = total_avail_hours / hours_diff
            
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Total Fleet", f"{total_fleet:,}")
            sm2.metric("Active Fleet", f"{active_fleet:,}")
            sm3.metric("Available Hours", f"{total_avail_hours:,.0f}")
            sm4.metric("Avg Available Fleet", f"{avg_avail_fleet:.1f}")
            st.divider()
            
            if not f_trend.empty:
                f_trend['Chart_Bucket'] = get_time_bucket(f_trend, 'Hour_Bucket', sel_time)
                
                chart_agg_avail = f_trend.groupby(['Chart_Bucket', 'Area'])['Net_Available'].sum().reset_index()
                st.write("#### 📉 Available Hours Trend")
                avail_chart = alt.Chart(chart_agg_avail).mark_line(point=True).encode(
                    x=alt.X('Chart_Bucket', axis=alt.Axis(format='%b %d %H:%M' if sel_time=='Hourly' else '%b %d')),
                    y=alt.Y('Net_Available', title='Available Hours'), color='Area', tooltip=['Chart_Bucket', 'Area', 'Net_Available']
                ).properties(height=350).interactive()
                st.altair_chart(avail_chart, use_container_width=True)
                
                chart_agg_active = f_trend.groupby(['Chart_Bucket', 'Area'])['Daily_Active_Baseline'].max().reset_index()
                st.write("#### 🛴 Active Scooters Trend")
                active_chart = alt.Chart(chart_agg_active).mark_line(point=True).encode(
                    x=alt.X('Chart_Bucket', axis=alt.Axis(format='%b %d %H:%M' if sel_time=='Hourly' else '%b %d')),
                    y=alt.Y('Daily_Active_Baseline', title='Active Fleet Count'), color='Area', tooltip=['Chart_Bucket', 'Area', 'Daily_Active_Baseline']
                ).properties(height=350).interactive()
                st.altair_chart(active_chart, use_container_width=True)

            if not f_util.empty:
                st.divider()
                st.write("#### 🚲 Active Fleet Utilization Funnel")
                f_util['Chart_Bucket'] = pd.to_datetime(f_util['Date'])
                c_u1, c_u2, c_u3 = st.columns(3)
                with c_u1:
                    u0 = alt.Chart(f_util).mark_line(point=True).encode(x=alt.X('Chart_Bucket'), y=alt.Y('Exact_0_Rides'), color='Area', tooltip=['Chart_Bucket', 'Area', 'Exact_0_Rides']).properties(title="= 0 Rides", height=280).interactive()
                    st.altair_chart(u0, use_container_width=True)
                with c_u2:
                    u1 = alt.Chart(f_util).mark_line(point=True).encode(x=alt.X('Chart_Bucket'), y=alt.Y('Exact_1_Ride'), color='Area', tooltip=['Chart_Bucket', 'Area', 'Exact_1_Ride']).properties(title="= 1 Ride", height=280).interactive()
                    st.altair_chart(u1, use_container_width=True)
                with c_u3:
                    u2 = alt.Chart(f_util).mark_line(point=True).encode(x=alt.X('Chart_Bucket'), y=alt.Y('Plus_2_Rides'), color='Area', tooltip=['Chart_Bucket', 'Area', 'Plus_2_Rides']).properties(title="≥ 2 Rides", height=280).interactive()
                    st.altair_chart(u2, use_container_width=True)

            if not f_att.empty:
                st.divider()
                st.write("#### 🔋 Operations & Productivity")
                f_att['Chart_Bucket'] = pd.to_datetime(f_att['Date'])
                c1 = alt.Chart(f_att).mark_line(point=True).encode(x='Chart_Bucket', y='Active_Shifts', color='Area', tooltip=['Date', 'Area', 'Active_Shifts']).properties(title="Active Shifts", height=250).interactive()
                st.altair_chart(c1, use_container_width=True)
                c2 = alt.Chart(f_att).mark_line(point=True).encode(x='Chart_Bucket', y='Total_Swaps', color='Area', tooltip=['Date', 'Area', 'Total_Swaps']).properties(title="Total Swaps", height=250).interactive()
                st.altair_chart(c2, use_container_width=True)
                c3 = alt.Chart(f_att).mark_line(point=True).encode(x='Chart_Bucket', y='Productivity', color='Area', tooltip=['Date', 'Area', alt.Tooltip('Productivity', format='.1f')]).properties(title="Productivity", height=250).interactive()
                st.altair_chart(c3, use_container_width=True)

            if not f_urgent.empty:
                st.divider()
                with st.expander("🚨 View Currently Urgent Scooters"):
                    st.dataframe(f_urgent)
