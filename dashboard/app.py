# dashboard/app.py
import mimetypes
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import gc

# ==============================================================================
# ADD PROJECT ROOT TO SYS.PATH
# ==============================================================================
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==============================================================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==============================================================================
st.set_page_config(page_title="Sarawak Hydroclimatology", page_icon="🌍", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
    h1 { font-size: 1.8rem !important; padding-bottom: 0 !important; }
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label { font-weight: bold; color: #4ba3e3; }
    hr { margin-top: 0.5rem; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌍 Sarawak Regional Hydroclimate Dashboard</h1>", unsafe_allow_html=True)

# ==============================================================================
# HELP SECTION
# ==============================================================================
with st.expander("ℹ️ Dashboard Help & User Guide"):
    st.markdown("""
    **Welcome to the Sarawak Hydroclimate Dashboard!** This tool visualizes the performance of Satellite Precipitation Estimates (SPEs), Machine Learning bias correction, and future climate risk.
    
    - **📊 Raw Data Characteristics**: Statistical overview, distributions, and station-level peer comparison.
    - **🗺️ Hydroclimate Explorer**: Explore raw and fused precipitation time series.
    - **📐 TCA Weight Analysis**: Analyze Triple Collocation Analysis weights.
    - **🤖 ML Model Evaluation**: Compare ML bias correction performance.
    - **🧠 Explainable AI (xAI)**: Understand ML decision-making using SHAP values.
    - **🌵 Drought Event Analysis**: Historical drought onset, termination, and return periods.
    - **🔮 Future Drought Risk Projection**: CMIP6-based future drought frequency and severity shifts.
    """)

st.markdown("---")

# ==============================================================================
# DATA PROCESSING DICTIONARIES
# ==============================================================================
LULC_MAP = {
    1: "Evergreen Forest", 2: "Open Shrublands", 3: "Deciduous Forest", 
    4: "Mixed Forests", 5: "Cropland/Natural Mosaic", 8: "Deciduous Broadleaf", 
    9: "Deciduous Needleleaf", 10: "Evergreen Broadleaf", 12: "Croplands", 
    13: "Urban & Built-Up", 14: "Cropland/Natural Mosaic", 15: "Snow & Ice", 
    16: "Barren or Sparsely Vegetated", 17: "Water Bodies"
}

PRECIP_SOURCES = {
    'precip_chirps': 'Raw CHIRPS', 'precip_era5_land': 'Raw ERA5-Land',
    'precip_persiann': 'Raw PERSIANN-CDR', 'P_merged_global': 'TCA Fused (Global)',
    'P_merged_season': 'TCA Fused (Seasonal)', 'P_merged_elev': 'TCA Fused (Elevation)',
    'P_merged_hier': 'TCA Fused (Hierarchical)'
}

# ==============================================================================
# DATA LOADING
# ==============================================================================
@st.cache_data
def load_data():
    parquet_path = project_root / "outputs" / "processed" / "sarawak_TCA_fused.parquet"
    if not parquet_path.exists():
        st.error(f"File not found: {parquet_path}. Please run TCA fusion first.")
        return pd.DataFrame()
    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'])
    if 'lulc_class' in df.columns and 'Land_Use' not in df.columns:
        df['Land_Use'] = df['lulc_class'].map(LULC_MAP).fillna('Unknown')
    elif 'Land_Use' not in df.columns:
        df['Land_Use'] = 'Unknown'
    return df

with st.spinner("Loading TCA-Fused Parquet Data..."):
    df = load_data()
    gc.collect()

if 'Gauge_Observed' in df.columns:
    PRECIP_SOURCES['Gauge_Observed'] = 'In-Situ Gauge (Truth)'

# ==============================================================================
# TAB SETUP (UPDATED WITH NEW DATA CHARACTERISTICS TAB)
# ==============================================================================
tab_data, tab_explore, tab_tca, tab_ml, tab_xai, tab_drought, tab_future = st.tabs([
    "📊 Data Characteristics",
    "🗺️ Hydroclimate Explorer", 
    "📐 TCA Weight Analysis", 
    "🤖 ML Model Evaluation", 
    "🧠 Explainable AI (xAI)", 
    "🌵 Drought Event Analysis",
    "🔮 Future Drought Risk Projection"
])

# ==============================================================================
# TAB 1: DATA CHARACTERISTICS & PEER ANALYSIS
# TODO: Probability Density Function (overlaying each (all SPE + in-situ) PDF density function graph; each density overlay one another and sigma labeled and central mean is indicated)
#      - Activate tooltip
#      - Multi-selection of data visibility (no limit on how many to select)
#      - Boxplot: maintain and multi-selection to sync  and check if it can be combined and overlaid together in one single docked plot
# ==============================================================================
with tab_data:
    st.markdown("### 📊 Raw Data Characteristics & Peer Analysis")
    st.write("Comprehensive statistical overview of the raw hydroclimatic datasets, comparing directly acquired Satellite Precipitation Estimates (SPEs) against in-situ gauge observations across spatial and temporal dimensions.")
    st.markdown("---")

    # 1. High-Level Dataset Summary
    st.markdown("#### 📈 Dataset Overview")
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    total_stations = df['station_id'].nunique()
    date_range_str = f"{df['date'].min().strftime('%b %Y')} - {df['date'].max().strftime('%b %Y')}"
    total_records = len(df)
    
    gauge_completeness = (df['Gauge_Observed'].notna().sum() / total_records * 100) if 'Gauge_Observed' in df.columns else 0
    
    with col_sum1:
        st.metric("Total Stations", total_stations)
    with col_sum2:
        st.metric("Temporal Coverage", date_range_str)
    with col_sum3:
        st.metric("Total Records", f"{total_records:,}")
    with col_sum4:
        st.metric("Gauge Data Completeness", f"{gauge_completeness:.1f}%")

    st.markdown("---")

    # 2. Distribution Comparison (Box Plot)
    st.markdown("#### 📦 Precipitation Distribution Comparison")
    st.caption("Box plots showing the spread, median, and outliers of monthly precipitation across different data sources. Highlights systematic biases in raw SPEs compared to ground truth.")
    
    plot_cols = ['Gauge_Observed', 'precip_chirps', 'precip_era5_land', 'precip_persiann']
    available_plot_cols = [col for col in plot_cols if col in df.columns]
    
    if available_plot_cols:
        df_melted = df[available_plot_cols].melt(var_name='Data Source', value_name='Precipitation (mm)')
        df_melted['Data Source'] = df_melted['Data Source'].map({
            'Gauge_Observed': 'In-Situ Gauge (Truth)',
            'precip_chirps': 'Raw CHIRPS',
            'precip_era5_land': 'Raw ERA5-Land',
            'precip_persiann': 'Raw PERSIANN-CDR'
        }).fillna('Unknown')
        
        fig_box = px.box(
            df_melted, 
            x='Data Source', 
            y='Precipitation (mm)', 
            color='Data Source',
            color_discrete_map={
                'In-Situ Gauge (Truth)': '#d62728',
                'Raw CHIRPS': '#1f77b4',
                'Raw ERA5-Land': '#ff7f0e',
                'Raw PERSIANN-CDR': '#2ca02c'
            },
            points="outliers",
            height=400
        )
        fig_box.update_layout(showlegend=False, xaxis_title="", yaxis_title="Monthly Precipitation (mm)")
        st.plotly_chart(fig_box, width="stretch")

    st.markdown("---")

    # 3. Correlation Matrix
    st.markdown("#### 🔗 Cross-Product Correlation Matrix")
    st.caption("Pearson correlation coefficients between all precipitation products. Higher values (closer to 1.0) indicate stronger linear agreement with the in-situ gauge network.")
    
    corr_df = df[available_plot_cols].corr().round(3)
    corr_df = corr_df.rename(index={
        'Gauge_Observed': 'Gauge', 'precip_chirps': 'CHIRPS', 
        'precip_era5_land': 'ERA5-Land', 'precip_persiann': 'PERSIANN'
    }, columns={
        'Gauge_Observed': 'Gauge', 'precip_chirps': 'CHIRPS', 
        'precip_era5_land': 'ERA5-Land', 'precip_persiann': 'PERSIANN'
    })
    
    fig_corr = px.imshow(
        corr_df, 
        text_auto=True, 
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        height=400
    )
    fig_corr.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig_corr, width="stretch")

    st.markdown("---")

    # 4. Station-Level Peer Comparison Table
    # TODO: Describe how the data is spatially collocate and specify that there are no interpolation applied
    # TODO: Create a plot that shows how many measurement per station for all station - synchronize sorting function with the table; multi-select for different data product
    st.markdown("#### 🏆 Station-Level Peer Comparison")
    st.caption("Aggregate statistics per station. Use the search and sort features to identify stations with high bias, low data completeness, or specific elevation characteristics.")
    
    # Dynamically build aggregation dictionary based on available columns
    agg_dict = {
        'Station_Name': ('station_name', 'first'),
        'Elevation_m': ('elev_m', 'first'),
        'Land_Use': ('Land_Use', 'first'),
        'Total_Months': ('date', 'count')
    }
    
    if 'Gauge_Observed' in df.columns:
        agg_dict['Mean_Gauge'] = ('Gauge_Observed', 'mean')
        agg_dict['Gauge_Months'] = ('Gauge_Observed', 'count')
    if 'precip_chirps' in df.columns:
        agg_dict['Mean_CHIRPS'] = ('precip_chirps', 'mean')
    if 'precip_era5_land' in df.columns:
        agg_dict['Mean_ERA5'] = ('precip_era5_land', 'mean')
    if 'precip_persiann' in df.columns:
        agg_dict['Mean_PERSIANN'] = ('precip_persiann', 'mean')
        
    station_stats = df.groupby('station_id').agg(**agg_dict).reset_index()
    
    # Calculate Bias Percentages if Gauge data exists
    if 'Gauge_Observed' in df.columns and 'Mean_Gauge' in station_stats.columns:
        station_stats['Data_Completeness_%'] = (station_stats['Gauge_Months'] / station_stats['Total_Months'] * 100).round(1)
        
        if 'Mean_CHIRPS' in station_stats.columns:
            station_stats['CHIRPS_Bias_%'] = ((station_stats['Mean_CHIRPS'] - station_stats['Mean_Gauge']) / station_stats['Mean_Gauge'] * 100).round(1)
        if 'Mean_ERA5' in station_stats.columns:
            station_stats['ERA5_Bias_%'] = ((station_stats['Mean_ERA5'] - station_stats['Mean_Gauge']) / station_stats['Mean_Gauge'] * 100).round(1)
        if 'Mean_PERSIANN' in station_stats.columns:
            station_stats['PERSIANN_Bias_%'] = ((station_stats['Mean_PERSIANN'] - station_stats['Mean_Gauge']) / station_stats['Mean_Gauge'] * 100).round(1)
        
        display_cols = ['Station_Name', 'Elevation_m', 'Land_Use', 'Mean_Gauge', 'Mean_CHIRPS', 'Mean_ERA5', 'Mean_PERSIANN', 'Data_Completeness_%', 'CHIRPS_Bias_%', 'ERA5_Bias_%', 'PERSIANN_Bias_%']
        
        column_config = {
            "Station_Name": "Station Name",
            "Elevation_m": st.column_config.NumberColumn("Elev. (m)", format="%.1f"),
            "Land_Use": "Land Use",
            "Mean_Gauge": st.column_config.NumberColumn("Mean Gauge (mm)", format="%.1f"),
            "Mean_CHIRPS": st.column_config.NumberColumn("Mean CHIRPS (mm)", format="%.1f"),
            "Mean_ERA5": st.column_config.NumberColumn("Mean ERA5 (mm)", format="%.1f"),
            "Mean_PERSIANN": st.column_config.NumberColumn("Mean PERSIANN (mm)", format="%.1f"),
            "Data_Completeness_%": st.column_config.ProgressColumn("Gauge Completeness", min_value=0, max_value=100, format="%.1f%%"),
            "CHIRPS_Bias_%": st.column_config.NumberColumn("CHIRPS Bias (%)", format="%.1f%%", help="Positive = Overestimation vs Gauge"),
            "ERA5_Bias_%": st.column_config.NumberColumn("ERA5 Bias (%)", format="%.1f%%", help="Positive = Overestimation vs Gauge"),
            "PERSIANN_Bias_%": st.column_config.NumberColumn("PERSIANN Bias (%)", format="%.1f%%", help="Positive = Overestimation vs Gauge")
        }
    else:
        # Fallback if no gauge data is present
        display_cols = ['Station_Name', 'Elevation_m', 'Land_Use', 'Total_Months']
        if 'Mean_CHIRPS' in station_stats.columns: display_cols.append('Mean_CHIRPS')
        if 'Mean_ERA5' in station_stats.columns: display_cols.append('Mean_ERA5')
        if 'Mean_PERSIANN' in station_stats.columns: display_cols.append('Mean_PERSIANN')
        
        column_config = {
            "Station_Name": "Station Name",
            "Elevation_m": st.column_config.NumberColumn("Elev. (m)", format="%.1f"),
            "Land_Use": "Land Use",
            "Total_Months": "Total Months"
        }

    st.dataframe(
        station_stats[display_cols],
        hide_index=True,
        width="stretch",
        column_config=column_config
    )
    
# ==============================================================================
# TAB 2: HYDROCLIMATE EXPLORER
# TODO: Minimize the map as much as possible and dock it at the right side near the dashboard toggle; expandable to select stations to inspect and redock
# ==============================================================================
with tab_explore:
    col_top1, col_top2, col_top3 = st.columns([1.5, 1, 1])

    with col_top1:
        min_date, max_date = df['date'].min().date(), df['date'].max().date()
        date_range = st.slider("📅 Timeline", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")

    with col_top2:
        plot_grouping = st.radio("📊 Chart Display Mode", options=["By Station", "By Dataset"], index=0)

    with col_top3:
        temporal_agg = st.radio("⏱️ Temporal Resolution", options=["Monthly", "Annual Totals"], index=0)

    c1, c2, c3 = st.columns(3)

    with c1:
        selected_precip = st.multiselect("🌧️ Precipitation Sources", options=list(PRECIP_SOURCES.keys()), format_func=lambda x: PRECIP_SOURCES[x], default=['precip_chirps', 'P_merged_hier'])

    with c2:
        station_dict = df.drop_duplicates('station_id').set_index('station_id').get('station_name', pd.Series(index=df['station_id'].unique(), data="Unknown")).to_dict()
        all_stations = sorted(df['station_id'].unique())
        selected_stations = st.multiselect(
            "📍 Select Stations (Max 5)", 
            options=all_stations, 
            format_func=lambda x: f"{x} - {station_dict.get(x, 'Unknown')}",
            default=[all_stations[0]] if all_stations else [], 
            max_selections=5
        )

    with c3:
        show_et = st.selectbox("☀️ Evapotranspiration Data", options=["None", "ERA5-Land PET (mm)"], index=1)

    st.markdown("---")

    df_filtered = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])].copy()

    if temporal_agg == "Annual Totals":
        df_filtered['year'] = df_filtered['date'].dt.year
        agg_dict = {src: 'sum' for src in PRECIP_SOURCES.keys() if src in df_filtered.columns}
        if 'pet' in df_filtered.columns:
            agg_dict['pet'] = 'sum'
        meta_cols = ['lon', 'lat', 'elev_m', 'Land_Use']
        for col in meta_cols:
            if col in df_filtered.columns:
                agg_dict[col] = 'first'
        df_filtered = df_filtered.groupby(['station_id', 'year']).agg(agg_dict).reset_index()
        df_filtered['date'] = pd.to_datetime(df_filtered['year'].astype(str) + '-01-01')

    def calculate_metrics(data, sources, truth_col='Gauge_Observed'):
        results = []
        if truth_col not in data.columns: return pd.DataFrame()
        for src in sources:
            if src == truth_col or src not in data.columns: continue
            valid_mask = data[src].notna() & data[truth_col].notna()
            sim, obs = data.loc[valid_mask, src], data.loc[valid_mask, truth_col]
            if len(sim) > 1:
                results.append({
                    "Product": PRECIP_SOURCES[src], 
                    "RMSE (mm)": round(np.sqrt(((sim - obs) ** 2).mean()), 2), 
                    "MAE (mm)": round(np.abs(sim - obs).mean(), 2), 
                    "CC (r)": round(np.corrcoef(sim, obs)[0, 1], 3)
                })
        return pd.DataFrame(results)

    col_map, col_plots = st.columns([1, 2.2])

    with col_map:
        st.markdown("### 🗺️ Spatial Collocation")
        map_agg = {src: 'sum' for src in PRECIP_SOURCES.keys() if src in df_filtered.columns}
        if 'pet' in df_filtered.columns: map_agg['pet'] = 'sum'
        summary_df = df_filtered.groupby(['station_id', 'lon', 'lat', 'elev_m', 'Land_Use'], dropna=False).agg(map_agg).reset_index()
        
        for col in map_agg.keys():
            summary_df[f"{col}_str"] = summary_df[col].apply(lambda x: f"{x:,.1f} mm" if pd.notnull(x) else "N/A")
        
        center_lat = summary_df['lat'].mean() if not summary_df.empty else 0
        center_lon = summary_df['lon'].mean() if not summary_df.empty else 0
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6, pitch=0)
        
        unselected_df = summary_df[~summary_df['station_id'].isin(selected_stations)]
        focus_df = summary_df[summary_df['station_id'].isin(selected_stations)]
        
        layers = []
        if not unselected_df.empty:
            layers.append(pdk.Layer("ScatterplotLayer", data=unselected_df, get_position="[lon, lat]", get_color="[120, 120, 120, 140]", get_radius=4000, pickable=True))
        if not focus_df.empty:
            layers.append(pdk.Layer("ScatterplotLayer", data=focus_df, get_position="[lon, lat]", get_color="[255, 50, 50, 255]", get_radius=8000, pickable=True))

        tooltip_html = "<b>📍 Station: {station_id}</b><br/><b>Land Use:</b> {Land_Use}<br/><b>Elev:</b> {elev_m} m<hr style='margin: 8px 0px;'/>"
        for src in selected_precip:
            if src in summary_df.columns: tooltip_html += f"<b>{PRECIP_SOURCES[src]}:</b> {{{src}_str}}<br/>"
                
        st.pydeck_chart(pdk.Deck(map_style='road', initial_view_state=view_state, layers=layers, tooltip={"html": tooltip_html}))
        
        if 'Gauge_Observed' in df.columns and selected_stations:
            st.markdown("### 📈 Performance Metrics")
            metrics_df = calculate_metrics(df_filtered[df_filtered['station_id'].isin(selected_stations)], selected_precip)
            if not metrics_df.empty: 
                st.dataframe(metrics_df, hide_index=True, width="stretch")
                st.download_button(
                    label="📥 Download Metrics CSV",
                    data=metrics_df.to_csv(index=False).encode('utf-8'),
                    file_name='station_metrics.csv',
                    mime='text/csv'
                )

    with col_plots:
        st.markdown("### 📊 Hydroclimatic Time Series") 
        # TODO: PET graph change bar chart to needle chart and overlay the plot upon one another with y-axis left and right (for their different metrics)
        # TODO: For the Annual Totals; remove the 'Jan' month indicator and just maintain the year to avoid misleading comprehension
        # TODO: Comment out TCA data from choices because it is not discussed yet
        # TODO: Try to combine to the first page
        if not selected_stations or not selected_precip:
            st.warning("👈 Please select at least one station and one precipitation source.")
        else:
            color_map = {'precip_chirps': '#1f77b4', 'precip_era5_land': '#ff7f0e', 'precip_persiann': '#2ca02c', 'P_merged_hier': '#9467bd', 'Gauge_Observed': '#d62728'}
            if plot_grouping == "By Station":
                for station in selected_stations:
                    station_data = df_filtered[df_filtered['station_id'] == station].sort_values('date')
                    fig = make_subplots(rows=2 if show_et != "None" else 1, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3] if show_et != "None" else [1.0])
                    
                    for src in selected_precip:
                        fig.add_trace(go.Scatter(x=station_data['date'], y=station_data[src], name=PRECIP_SOURCES[src], mode='lines+markers' if temporal_agg == "Annual Totals" else 'lines', line=dict(color=color_map.get(src, None), width=2)), row=1, col=1)
                    
                    if show_et != "None" and 'pet' in station_data.columns:
                        fig.add_trace(go.Bar(x=station_data['date'], y=station_data['pet'], name="PET", marker_color='rgba(255, 165, 0, 0.6)'), row=2, col=1)
                    
                    fig.update_layout(title=dict(text=f"<b>Station: {station}</b>"), height=350, margin={"r":20,"t":40,"l":20,"b":20}, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)), hovermode='x unified')
                    st.plotly_chart(fig, width="stretch")


# ==============================================================================
# TAB 3: TCA WEIGHT ANALYSIS (OPTIMIZED)
# TODO: Conventional bar plot representation to be included to aid understanding
# TODO: Taylor Diagram - increase the font label, change the marker size and symbol and color
# ==============================================================================
with tab_tca:
    st.markdown("### 📐 Triple Collocation Analysis (TCA) Weight Allocation")
    st.write("This module evaluates the inverse-variance weights assigned to each satellite product by the TCA framework. Weights are derived by analyzing the mutual log-space covariance among the three independent products.")
    
    strat_level = st.radio("Select Stratification Level to Analyze:", 
                           options=["Global", "Seasonal", "Elevation", "Hierarchical"], 
                           horizontal=True)
    
    @st.cache_data
    def compute_all_tca_weights(file_path):
        df_temp = pd.read_parquet(file_path)
        valid = df_temp[(df_temp['precip_chirps'] > 0) & 
                        (df_temp['precip_era5_land'] > 0) & 
                        (df_temp['precip_persiann'] > 0)].copy()
        
        def get_w(data, target_col):
            if len(data) < 5:
                return [np.nan, np.nan, np.nan]
            try:
                X = np.log(data[['precip_chirps', 'precip_era5_land', 'precip_persiann']].values)
                y = np.log(data[target_col].values)
                w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                return w
            except Exception:
                return [np.nan, np.nan, np.nan]
        
        w_g = get_w(valid, 'P_merged_global')
        df_g = pd.DataFrame([{"Stratum": "Global", "CHIRPS": w_g[0], "ERA5-Land": w_g[1], "PERSIANN-CDR": w_g[2]}])
        
        records_s = []
        for s in sorted(valid['season'].dropna().unique()):
            w = get_w(valid[valid['season'] == s], 'P_merged_season')
            records_s.append({"Stratum": s, "CHIRPS": w[0], "ERA5-Land": w[1], "PERSIANN-CDR": w[2]})
        df_s = pd.DataFrame(records_s)
        
        records_e = []
        for e in sorted(valid['elev_bin'].dropna().unique(), reverse=True):
            w = get_w(valid[valid['elev_bin'] == e], 'P_merged_elev')
            records_e.append({"Stratum": e, "CHIRPS": w[0], "ERA5-Land": w[1], "PERSIANN-CDR": w[2]})
        df_e = pd.DataFrame(records_e)
        
        records_h = []
        for h in sorted(valid['stratum_hier'].dropna().unique()):
            w = get_w(valid[valid['stratum_hier'] == h], 'P_merged_hier')
            records_h.append({"Stratum": h, "CHIRPS": w[0], "ERA5-Land": w[1], "PERSIANN-CDR": w[2]})
        df_h = pd.DataFrame(records_h)
        
        return df_g, df_s, df_e, df_h

    project_root = Path(__file__).parent.parent
    file_path = str(project_root / "outputs" / "processed" / "sarawak_TCA_fused.parquet")
    
    with st.spinner("Calculating TCA weights..."):
        df_g, df_s, df_e, df_h = compute_all_tca_weights(file_path)
        gc.collect()
    
    weights_map = {
        "Global": df_g,
        "Seasonal": df_s,
        "Elevation": df_e,
        "Hierarchical": df_h
    }
    weights_df = weights_map.get(strat_level, df_g)
    
    if weights_df.empty or weights_df['CHIRPS'].isna().all():
        st.warning("⚠️ No valid TCA weights could be calculated for this stratum. This usually happens if a specific combination has insufficient wet-day observations.")
    else:
        col_w1, col_w2 = st.columns([1.2, 1])
        
        with col_w1:
            st.markdown(f"#### 🌡️ {strat_level} Weight Allocation Heatmap")
            heat_data = weights_df[['CHIRPS', 'ERA5-Land', 'PERSIANN-CDR']].values
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_data,
                x=['CHIRPS', 'ERA5-Land', 'PERSIANN-CDR'],
                y=weights_df['Stratum'],
                colorscale='Viridis',
                text=np.round(heat_data, 3),
                texttemplate="%{text}",
                hoverongaps=False
            ))
            fig_heat.update_layout(height=450, margin={"r":20,"t":20,"l":20,"b":20}, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_heat, width="stretch")
            st.caption("Values closer to 1.0 (yellow) indicate lower random error variance (higher reliability) for that product in that specific stratum.")

        with col_w2:
            st.markdown("#### 🔺 Ternary Simplex Clustering")
            fig_tern = px.scatter_ternary(
                weights_df, a="PERSIANN-CDR", b="CHIRPS", c="ERA5-Land", 
                hover_name="Stratum", color="Stratum" if strat_level != "Global" else None, size_max=15
            )
            fig_tern.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
            fig_tern.update_layout(
                height=450, margin={"r":20,"t":40,"l":20,"b":20},
                ternary=dict(sum=1, aaxis=dict(title="PERSIANN-CDR", tickformat=".2f", tickfont=dict(size=10)),
                             baxis=dict(title="CHIRPS", tickformat=".2f", tickfont=dict(size=10)),
                             caxis=dict(title="ERA5-Land", tickformat=".2f", tickfont=dict(size=10)))
            )
            st.plotly_chart(fig_tern, width="stretch")
            st.caption("Visualizes the convex hull of error weights, demonstrating physical equifinality across spatial strata.")

        with st.expander("🔍 View Raw TCA Weight Data"):
            st.dataframe(weights_df.style.format({"CHIRPS": "{:.3f}", "ERA5-Land": "{:.3f}", "PERSIANN-CDR": "{:.3f}"}), width="stretch")


# ==============================================================================
# TAB 4: ML BIAS CORRECTION EVALUATION
# ==============================================================================
with tab_ml:
    st.markdown("### 🤖 Machine Learning Bias Correction Performance")
    st.write("This module evaluates the effectiveness of ML bias correction by comparing raw Hierarchical TCA products against ML-corrected products using In-Situ Gauges as ground truth.")

    @st.cache_data
    def load_ml_data():
        project_root = Path(__file__).parent.parent
        ml_parquet = project_root / "outputs" / "processed" / "sarawak_ML_corrected.parquet"
        ml_leaderboard = project_root / "outputs" / "processed" / "ml_leaderboard.csv"
        
        df_ml = pd.DataFrame()
        leaderboard = pd.DataFrame()
        
        if ml_parquet.exists():
            df_ml = pd.read_parquet(ml_parquet)
            df_ml['date'] = pd.to_datetime(df_ml['date'])
        if ml_leaderboard.exists():
            leaderboard = pd.read_csv(ml_leaderboard)
            
        return df_ml, leaderboard

    with st.spinner("🔄 Loading ML evaluation data and computing metrics..."):
        df_ml, leaderboard = load_ml_data()
        gc.collect()

    if df_ml.empty:
        st.error("⚠️ `sarawak_ML_corrected.parquet` not found. Please run `05_ml_bias_correction.ipynb` first.")
    else:
        valid_ml = df_ml.dropna(subset=['Gauge_Observed', 'P_merged_hier', 'P_corrected']).copy()
        
        st.markdown("---")
        st.markdown("#### 🎯 Model Selection")
        
        if not leaderboard.empty:
            available_models = sorted(leaderboard['ML_Model'].unique())
            selected_models = st.multiselect(
                "🔍 Select ML Models to Compare", 
                options=available_models, 
                default=[available_models[0]] if available_models else [],
                help="Select models to visualize. Each model will have a unique marker shape."
            )
        else:
            selected_models = []
            st.warning("No models found in leaderboard.")
        
        if not selected_models:
            st.stop()
        
        st.markdown("---")
        st.markdown("#### 🏆 Model Performance Leaderboard")
        
        st.caption("📊 **Composite Score:** Key metrics (R, R², RMSE, KGE) are normalized to a 0–1 scale. Error metrics are inverted so lower errors yield higher scores. The final score is the average of these normalized values (Rank 1 = Best balanced performance).")
        
        col_lb1, col_lb2 = st.columns([1.5, 1])
        
        with col_lb1:
            if not leaderboard.empty:
                display_lb = leaderboard[leaderboard['ML_Model'].isin(selected_models)].copy()
                display_lb = display_lb[['Fusion_Method', 'ML_Model', 'RMSE', 'R2', 'KGE', 'Composite_Score']].copy()
                display_lb = display_lb.sort_values('Composite_Score', ascending=False)
                
                def highlight_best(s):
                    is_max = s == s.max()
                    return ['background-color: #d4edda' if v else '' for v in is_max]
                
                st.dataframe(display_lb.style.apply(highlight_best, subset=['Composite_Score']), hide_index=True, width="stretch")
                
                st.download_button(
                    label="📥 Download Leaderboard CSV",
                    data=display_lb.to_csv(index=False).encode('utf-8'),
                    file_name='ml_leaderboard.csv',
                    mime='text/csv'
                )

        with col_lb2:
            if not leaderboard.empty and 'Composite_Score' in leaderboard.columns:
                st.markdown("**Composite Score Ranking**")
                fig_bar = px.bar(
                    leaderboard[leaderboard['ML_Model'].isin(selected_models)].sort_values('Composite_Score', ascending=True), 
                    x='Composite_Score', y='ML_Model', color='Fusion_Method', orientation='h', text='Composite_Score'
                )
                fig_bar.update_layout(height=300, margin={"r":20,"t":20,"l":20,"b":20}, showlegend=False)
                fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                st.plotly_chart(fig_bar, width="stretch")

        st.markdown("---")
        st.markdown("#### 📊 Unified Performance Comparison")
        
        st.caption("📊 **Scatter Plot:** Red = Raw, Green = Corrected. Marker shapes represent different ML models. Points hugging the 1:1 red dashed line indicate better agreement. *(Data is intelligently sampled for instant browser rendering).*")
        
        marker_shapes = {
            'Linear Regression': 'circle', 'Random Forest': 'square', 'Gradient Boosting': 'diamond', 
            'Neural Network (MLP)': 'triangle-up', 'SVR': 'pentagon'
        }
        
        fig_scatter = go.Figure()
        MAX_POINTS = 15000 
        
        for model in selected_models:
            model_data = valid_ml[valid_ml['model_used'] == model] if 'model_used' in valid_ml.columns else valid_ml
            
            for variant, col_name in [('Raw (Hierarchical)', 'P_merged_hier'), (f'Corrected ({model})', 'P_corrected')]:
                subset = model_data[['Gauge_Observed', col_name]].dropna()
                if len(subset) > MAX_POINTS:
                    subset = subset.sample(MAX_POINTS, random_state=42)
                
                is_raw = 'Raw' in variant
                color = '#ef553b' if is_raw else '#00cc96'
                
                fig_scatter.add_trace(go.Scattergl(
                    x=subset['Gauge_Observed'], y=subset[col_name], mode='markers', name=f"{model} - {variant}",
                    marker=dict(symbol=marker_shapes.get(model, 'circle'), size=4, color=color, opacity=0.35, line=dict(color='white', width=0.2))
                ))
        
        max_val = max(valid_ml['Gauge_Observed'].max(), valid_ml['P_merged_hier'].max(), valid_ml['P_corrected'].max())
        fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='1:1 Perfect Agreement', line=dict(color='red', dash='dash', width=2)))
        
        fig_scatter.update_layout(height=500, xaxis_title="In-Situ Gauge (mm)", yaxis_title="Estimated Precipitation (mm)", hovermode='closest', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_scatter, width="stretch")
        
        st.markdown("---")
        st.markdown("#### 📐 Unified Taylor Diagram")
        
        st.caption("📐 **Taylor Diagram:** The Reference point (red star) at (1,0) represents perfect agreement. **Open markers** = Raw products. **Filled marker** = ML-Corrected P_merged. Points closer to the reference indicate better performance. Radial distance = normalized standard deviation. Azimuthal angle = correlation coefficient. Green dashed arcs = centered RMS difference (labels aligned along each arc).")
        
        all_taylor_products = {
            'precip_chirps': 'CHIRPS (Raw)',
            'precip_persiann': 'PERSIANN-CDR (Raw)',
            'precip_era5_land': 'ERA5-Land (Raw)',
            'P_merged_global': 'P_merged (Global)',
            'P_merged_hier': 'P_merged (Hierarchical)',
            'P_corrected': 'ML-Corrected P_merged'
        }
        
        selected_taylor_products = st.multiselect(
            "🔍 Select Products to Display on Taylor Diagram",
            options=list(all_taylor_products.keys()),
            default=['precip_chirps', 'precip_persiann', 'precip_era5_land', 'P_merged_hier', 'P_corrected'],
            format_func=lambda x: all_taylor_products[x],
            help="Toggle which products appear on the Taylor Diagram."
        )
        
        obs = valid_ml['Gauge_Observed']
        obs_std = np.std(obs)
        
        product_defs = {
            'precip_chirps':       {'label': 'CHIRPS',             'symbol': 'circle-open',      'color': '#1f77b4'},
            'precip_persiann':     {'label': 'PERSIANN-CDR',       'symbol': 'square-open',      'color': '#ff7f0e'},
            'precip_era5_land':    {'label': 'ERA5-Land',          'symbol': 'diamond-open',     'color': '#2ca02c'},
            'P_merged_global':     {'label': 'P_merged (Global)',  'symbol': 'triangle-up-open', 'color': '#8c564b'},
            'P_merged_hier':       {'label': 'P_merged (Hier)',    'symbol': 'triangle-down-open','color': '#9467bd'},
            'P_corrected':         {'label': 'ML-Corrected',       'symbol': 'star',             'color': '#d62728'}
        }
        
        taylor_df = pd.DataFrame()
        
        for col_key in selected_taylor_products:
            if col_key not in valid_ml.columns:
                continue
            info = product_defs[col_key]
            est = valid_ml[col_key].dropna()
            obs_aligned = valid_ml.loc[est.index, 'Gauge_Observed'].dropna()
            common_idx = est.index.intersection(obs_aligned.index)
            est = est.loc[common_idx]
            obs_v = obs_aligned.loc[common_idx]
            
            if len(est) < 10 or np.std(est) == 0:
                continue
                
            r = np.corrcoef(obs_v, est)[0, 1]
            std_norm = np.std(est) / obs_std
            rmsd = np.sqrt(np.mean((est - obs_v)**2)) / obs_std
            
            taylor_df = pd.concat([taylor_df, pd.DataFrame([{
                'Product': info['label'], 'R': r, 'Std': std_norm, 
                'RMSD': rmsd, 'Color': info['color'], 'Symbol': info['symbol']
            }])], ignore_index=True)
        
        fig_taylor = go.Figure()
        
        rmsd_color, corr_color, std_color, grid_color = '#006400', '#2F4F4F', '#4B0082', '#696969'
        max_std = max(taylor_df['Std'].max(), 1.5) * 1.2 if not taylor_df.empty else 2.0
        theta = np.linspace(0, np.pi/2, 100)
        
        rmsd_values = [0.5, 1.0, 1.5]
        for rmsd_val in rmsd_values:
            if rmsd_val < max_std * 1.5:
                x_arc = 1 + rmsd_val * np.cos(np.pi - theta)
                y_arc = rmsd_val * np.sin(np.pi - theta)
                mask = (x_arc >= 0) & (x_arc <= max_std) & (y_arc >= 0) & (y_arc <= max_std)
                
                fig_taylor.add_trace(go.Scatter(
                    x=x_arc[mask], y=y_arc[mask], mode='lines',
                    line=dict(color=rmsd_color, dash='dash', width=1.5),
                    name=f'RMSD={rmsd_val}', showlegend=False 
                ))
                
                label_theta = np.pi / 4
                label_x = 1 + rmsd_val * np.cos(np.pi - label_theta)
                label_y = rmsd_val * np.sin(np.pi - label_theta)
                
                fig_taylor.add_annotation(
                    x=label_x, y=label_y,
                    text=f'{rmsd_val}',
                    showarrow=False,
                    font=dict(size=10, color=rmsd_color),
                    textangle=45,
                    xanchor='center', 
                    yanchor='middle',
                    bgcolor='rgba(255,255,255,0.9)',
                    borderpad=2
                )
        
        for r_val in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
            angle = np.arccos(r_val)
            fig_taylor.add_trace(go.Scatter(
                x=[0, max_std * np.cos(angle)], y=[0, max_std * np.sin(angle)],
                mode='lines', line=dict(color=corr_color, width=1.5, dash='dash'),
                showlegend=False
            ))
            fig_taylor.add_annotation(
                x=max_std * 0.92 * np.cos(angle), y=max_std * 0.92 * np.sin(angle),
                text=f'{r_val}', showarrow=False,
                font=dict(size=10, color='#000000'), textangle=np.degrees(angle) - 90
            )
        
        for std_val in [0.5, 1.0, 1.5]:
            if std_val <= max_std:
                theta_circle = np.linspace(0, np.pi/2, 100)
                fig_taylor.add_trace(go.Scatter(
                    x=std_val * np.cos(theta_circle), y=std_val * np.sin(theta_circle),
                    mode='lines', line=dict(color=std_color, width=1.5, dash='dot'),
                    showlegend=False
                ))
        
        fig_taylor.add_trace(go.Scatter(
            x=[1], y=[0], mode='markers+text',
            marker=dict(size=14, color='red', symbol='star'),
            name='Reference (Perfect)', text=['Reference'],
            textposition='bottom center'
        ))
        
        for _, row in taylor_df.iterrows():
            y_val = row['Std'] * np.sin(np.arccos(row['R']))
            
            fig_taylor.add_trace(go.Scatter(
                x=[row['Std']], y=[y_val], mode='markers+text',
                marker=dict(
                    size=10,
                    color=row['Color'], symbol=row['Symbol'],
                    line=dict(width=2, color=row['Color']), opacity=1.0
                ),
                name=row['Product'],
                text=[f"{row['Product'][:10]}"], textposition='top center',
                textfont=dict(size=9, color='#000000'),
                hovertemplate=f"Product: {row['Product']}<br>R: {row['R']:.3f}<br>Std: {row['Std']:.3f}<br>RMSD: {row['RMSD']:.3f}<extra></extra>"
            ))
        
        fig_taylor.update_layout(
            height=700,
            xaxis=dict(
                title=dict(text='Normalized Standard Deviation', font=dict(size=14, color='#000000')),
                range=[0, max_std], zeroline=True, zerolinecolor='#000000', zerolinewidth=1.5,
                showgrid=True, gridcolor=grid_color, gridwidth=1,
                showline=True, linecolor='#000000', linewidth=1.5,
                ticks='outside', tickcolor='#000000', tickwidth=1.5,
                showticklabels=True, tickfont=dict(size=11, color='#000000')
            ),
            yaxis=dict(
                title=dict(text='Normalized Standard Deviation', font=dict(size=14, color='#000000')),
                range=[0, max_std], zeroline=True, zerolinecolor='#000000', zerolinewidth=1.5,
                showgrid=True, gridcolor=grid_color, gridwidth=1,
                showline=True, linecolor='#000000', linewidth=1.5,
                ticks='outside', tickcolor='#000000', tickwidth=1.5,
                showticklabels=True, tickfont=dict(size=11, color='#000000'),
                scaleanchor="x", scaleratio=1
            ),
            legend=dict(
                orientation="v", yanchor="top", y=1, xanchor="left", x=1.05,
                font=dict(size=10, color='#000000'),
                bgcolor='rgba(255,255,255,0.95)', bordercolor='#000000', borderwidth=1
            ),
            showlegend=True, plot_bgcolor='rgba(255,255,255,1)',
            paper_bgcolor='rgba(255,255,255,1)', margin=dict(l=80, r=40, t=80, b=80)
        )
        
        fig_taylor.add_annotation(
            x=max_std * 0.85, y=max_std * 0.5, text="Correlation",
            showarrow=False, font=dict(size=12, color='#000000'), textangle=-30
        )
        fig_taylor.add_annotation(
            x=1, y=-0.1, text="Reference", showarrow=False,
            font=dict(size=11, color='#000000')
        )
        
        st.plotly_chart(fig_taylor, width="stretch")
        
        try:
            st.download_button(
                label="📥 Download Taylor Diagram PNG",
                data=fig_taylor.to_image(format="png", scale=2),
                file_name='taylor_diagram.png', mime='image/png'
            )
        except Exception:
            st.caption("💡 *Note: To enable PNG download, install the Kaleido package by running `pip install kaleido` in your terminal.*")
        
        st.markdown("---")
        st.markdown("#### 📊 Performance Metrics Summary")
        
        metrics_rows = []
        for model in selected_models:
            model_data = valid_ml[valid_ml['model_used'] == model] if 'model_used' in valid_ml.columns else valid_ml
            for variant_name, est_col in [('Raw P_merged', 'P_merged_hier'), ('ML-Corrected', 'P_corrected')]:
                est = model_data[est_col]
                obs_v = model_data['Gauge_Observed']
                r = np.corrcoef(obs_v, est)[0, 1]
                rmse = np.sqrt(np.mean((est - obs_v)**2))
                bias = np.mean(est - obs_v)
                r_kge = r
                alpha = np.std(est) / np.std(obs_v)
                beta = np.mean(est) / np.mean(obs_v)
                kge = 1 - np.sqrt((r_kge - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                metrics_rows.append({'Model': model, 'Variant': variant_name, 'R': round(r, 3), 'RMSE (mm)': round(rmse, 2), 'Bias (mm)': round(bias, 2), 'KGE': round(kge, 3)})
        
        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(metrics_df, hide_index=True, width="stretch")
        
        st.download_button(
            label="📥 Download Metrics Summary CSV",
            data=metrics_df.to_csv(index=False).encode('utf-8'),
            file_name='ml_metrics_summary.csv',
            mime='text/csv'
        )

# ==============================================================================
# TAB 5: EXPLAINABLE AI (xAI) - SIDE-BY-SIDE MULTI-MODEL COMPARISON
# ==============================================================================
with tab_xai:
    st.markdown("### 🧠 Explainable AI: Decoding the Bias Correction")
    st.write("This module uses SHAP to explain *why* the ML model corrects the satellite data. Positive SHAP values mean the model *adds* rain (correcting underestimation); negative values mean it *removes* rain.")

    project_root = Path(__file__).parent.parent
    xai_dir = project_root / "outputs" / "processed"
    
    available_models = []
    for f in xai_dir.glob("shap_importance_*.csv"):
        model_name = f.stem.replace("shap_importance_", "").replace("_", " ")
        available_models.append(model_name)
    
    if not available_models:
        st.error("⚠️ No xAI data found. Please run `06_xai_analysis.ipynb` first.")
        st.stop()

    st.markdown("---")
    st.markdown("#### 🎯 Select Models to Compare Side-by-Side")
    selected_models = st.multiselect(
        "Choose up to 3 ML models to analyze:",
        options=available_models,
        default=[available_models[0]] if available_models else [],
        help="Select up to 3 models to compare their decision-making logic side-by-side without scrolling."
    )

    if not selected_models:
        st.warning("Please select at least one model from the dropdown above.")
        st.stop()
        
    if len(selected_models) > 3:
        st.warning("⚠️ For optimal side-by-side viewing, please select a maximum of 3 models.")
        selected_models = selected_models[:3]

    cols = st.columns(len(selected_models))

    for col, model_name in zip(cols, selected_models):
        with col:
            st.markdown(f"### 🤖 {model_name}")
            
            safe_name = model_name.replace(" ", "_")
            imp_csv = xai_dir / f"shap_importance_{safe_name}.csv"
            bee_csv = xai_dir / f"shap_beeswarm_data_{safe_name}.csv"
            
            if not imp_csv.exists() or not bee_csv.exists():
                st.error(f"Data missing for {model_name}")
                continue
                
            imp_df = pd.read_csv(imp_csv)
            bee_df = pd.read_csv(bee_csv)
            
            st.markdown("#### 📊 Feature Importance")
            st.caption("Which physical factors drive the bias correction?")
            
            fig_imp = px.bar(imp_df, x='Mean |SHAP|', y='Feature', orientation='h', 
                             color='Feature', color_discrete_sequence=px.colors.qualitative.Bold)
            fig_imp.update_layout(height=280, showlegend=False, xaxis_title="Impact", margin=dict(l=0, r=0, t=20, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown("#### 🐝 SHAP Beeswarm")
            st.caption("Right = adds rain, Left = removes rain. Red = High feature value.")
            
            fig_bee = px.scatter(bee_df, x='SHAP Value', y='Feature', color='Feature Value',
                                 color_continuous_scale='RdBu_r', opacity=0.5, height=320)
            fig_bee.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_bee.update_layout(margin=dict(l=0, r=0, t=20, b=20))
            st.plotly_chart(fig_bee, use_container_width=True)
            
            st.markdown(f"#### 🔗 3. Feature Interaction & Dependence ({model_name})")
            st.caption("Select a feature to see how its value impacts the model's bias correction. The color reveals interaction effects with other variables.")
            
            top_features = imp_df['Feature'].tail(4).tolist()
            
            target_feature = st.selectbox(
                "Select Feature to Analyze:",
                options=top_features,
                index=len(top_features)-1,
                help="Choose which physical factor's impact you want to visualize.",
                key=f"feature_select_{safe_name}"
            )
            
            feat_data = bee_df[bee_df['Feature'] == target_feature].copy()
            
            if target_feature == 'elev_m':
                color_col = 'Season'
                color_map = {'NEM': '#d62728', 'SWM': '#2ca02c', 'Inter': '#ff7f0e'}
                x_label = 'Elevation (m)'
            elif target_feature == 'month':
                color_col = 'Feature Value'
                color_map = None
                x_label = 'Month (1-12)'
            elif target_feature == 'LULC Class':
                color_col = 'Season'
                color_map = {'NEM': '#d62728', 'SWM': '#2ca02c', 'Inter': '#ff7f0e'}
                x_label = 'LULC Class'
            else:
                color_col = 'Season'
                color_map = {'NEM': '#d62728', 'SWM': '#2ca02c', 'Inter': '#ff7f0e'}
                x_label = target_feature.replace('_', ' ').title()
            
            if color_map:
                fig_dep = px.scatter(
                    feat_data, x='Feature Value', y='SHAP Value', color=color_col,
                    color_discrete_map=color_map, opacity=0.4, height=400,
                    labels={'Feature Value': x_label, 'SHAP Value': 'Impact on Bias Correction'}
                )
            else:
                fig_dep = px.scatter(
                    feat_data, x='Feature Value', y='SHAP Value', color=color_col,
                    color_continuous_scale='Viridis', opacity=0.4, height=400,
                    labels={'Feature Value': x_label, 'SHAP Value': 'Impact on Bias Correction'}
                )
            
            fig_dep.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_dep, use_container_width=True, key=f"dep_plot_{safe_name}")
            
            st.info(f"""
            **🔬 Interpretation for {target_feature}:** 
            Look at how the SHAP values (y-axis) change as the {x_label} (x-axis) increases. 
            The coloring shows if this relationship changes depending on the **{color_col}**. 
            - Points **above y=0** mean the model *adds* precipitation to correct satellite underestimation.
            - Points **below y=0** mean the model *removes* precipitation to correct overestimation.
            """)


# ==============================================================================
# TAB 6: DROUGHT EVENT ANALYSIS (INTERACTIVE NEEDLE PLOT)
# ==============================================================================
with tab_drought:
    st.markdown("### 🌵 Drought Event Analysis & Forecasting")
    st.write("Interactive analysis of drought onset, termination, and return periods using Run Theory (Yevjevich, 1967).")
    st.markdown("---")

    project_root = Path(__file__).parent.parent
    events_csv = project_root / "outputs" / "processed" / "drought_events_summary.csv"
    data_csv = project_root / "outputs" / "processed" / "sarawak_ML_corrected.parquet"

    if not events_csv.exists() or not data_csv.exists():
        st.error("⚠️ Drought data not found. Please run `07_drought_event_extraction.ipynb` first.")
        st.stop()

    events_df = pd.read_csv(events_csv)
    events_df['station_id'] = events_df['station_id'].astype(str)
    events_df['onset'] = pd.to_datetime(events_df['onset'])
    events_df['termination'] = pd.to_datetime(events_df['termination'])
    
    df_full = pd.read_parquet(data_csv)
    df_full['date'] = pd.to_datetime(df_full['date'])
    df_full['station_id'] = df_full['station_id'].astype(str)

    stations_with_events = events_df[['station_id', 'station_name']].drop_duplicates()
    station_map = dict(zip(stations_with_events['station_id'], stations_with_events['station_name']))
    
    display_options = [f"{name} ({sid})" for sid, name in station_map.items()]
    display_options.sort()

    st.markdown("#### 🎛️ Dashboard Filters")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_display = st.selectbox("Select Station:", options=display_options)
        selected_station = selected_display.split('(')[-1].strip(')')
        
    stn_data = df_full[df_full['station_id'] == selected_station].set_index('date').sort_index()
    stn_events = events_df[events_df['station_id'] == selected_station].copy()
    
    stn_meta = df_full[df_full['station_id'] == selected_station].iloc[0]
    stn_lat = stn_meta['lat']
    stn_lon = stn_meta['lon']
    stn_elev = stn_meta['elev_m']
    
    stn_lulc_name = stn_meta.get('Land_Use')
    stn_lulc_code = stn_meta.get('lulc_class')
    if pd.isna(stn_lulc_name) and pd.notna(stn_lulc_code):
        try:
            stn_lulc = LULC_MAP.get(int(stn_lulc_code), 'Unknown')
        except (ValueError, TypeError):
            stn_lulc = 'Unknown'
    else:
        stn_lulc = stn_lulc_name if pd.notna(stn_lulc_name) else 'Unknown'

    stn_data_stats = df_full[df_full['station_id'] == selected_station]
    precip_min = stn_data_stats['P_corrected'].min()
    precip_max = stn_data_stats['P_corrected'].max()
    precip_mean = stn_data_stats['P_corrected'].mean()
    
    if 'pet' in stn_data_stats.columns:
        pet_min = stn_data_stats['pet'].min()
        pet_max = stn_data_stats['pet'].max()
        pet_mean = stn_data_stats['pet'].mean()
    else:
        pet_min = pet_max = pet_mean = np.nan

    st.caption(f"📊 Found **{len(stn_events)}** drought events for {selected_display}.")

    with col2:
        col_map, col_info = st.columns([2, 1])
        
        with col_map:
            map_df = pd.DataFrame({'lat': [stn_lat], 'lon': [stn_lon], 'station': [selected_display]})
            view_state = pdk.ViewState(latitude=stn_lat, longitude=stn_lon, zoom=10, pitch=0)
            layers = [pdk.Layer("ScatterplotLayer", data=map_df, get_position="[lon, lat]", get_color="[255, 50, 50, 255]", get_radius=2000, pickable=True)]
            st.pydeck_chart(pdk.Deck(map_style='road', initial_view_state=view_state, layers=layers, tooltip={"html": f"<b>Station:</b> {selected_display}"}, height=250))
            
        with col_info:
            st.markdown("##### 📍 Station Info")
            st.metric("Elevation", f"{stn_elev:.1f} m")
            st.metric("Land Cover", stn_lulc)
            st.metric("Avg Precip", f"{precip_mean:.1f} mm")
            st.metric("Precip Range", f"{precip_min:.1f} - {precip_max:.1f} mm")
            if not np.isnan(pet_mean):
                st.metric("Avg PET", f"{pet_mean:.1f} mm")
                st.metric("PET Range", f"{pet_min:.1f} - {pet_max:.1f} mm")

    st.markdown("---")

    zoom_key = f"x_range_{selected_station}"
    if zoom_key not in st.session_state:
        st.session_state[zoom_key] = [stn_data.index.min(), stn_data.index.max()]

    st.markdown("#### 📊 Interactive SPI-12 Needle Plot")
    st.caption("Blue bars = Wet/Normal (SPI > -1.0). Red bars = Drought (SPI ≤ -1.0). Markers show onset (▼) and termination (▲).")

    @st.cache_data
    def get_spi_data(station_data):
        from src.drought_indices import calculate_spi
        return calculate_spi(station_data['P_corrected'], scale=12)

    with st.spinner("Calculating SPI-12..."):
        stn_data['SPI_12'] = get_spi_data(stn_data)

    fig = go.Figure()
    colors = ['red' if v <= -1.0 else '#1f77b4' for v in stn_data['SPI_12']]
    
    fig.add_trace(go.Bar(x=stn_data.index, y=stn_data['SPI_12'], name='SPI-12', marker_color=colors, opacity=0.8))
    fig.add_hline(y=-1.0, line_dash="dash", line_color="black", line_width=2, annotation_text="Moderate Drought Threshold")
    fig.add_hline(y=-1.5, line_dash="dot", line_color="darkred", line_width=1.5, annotation_text="Severe")
    fig.add_hline(y=-2.0, line_dash="dot", line_color="red", line_width=1.5, annotation_text="Extreme")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    if len(stn_events) > 0:
        for i, event in stn_events.iterrows():
            try:
                onset_spi = stn_data.loc[event['onset'], 'SPI_12']
                termination_spi = stn_data.loc[event['termination'], 'SPI_12']
                
                fig.add_vrect(x0=event['onset'], x1=event['termination'], fillcolor="red", opacity=0.15, line_width=0)
                
                fig.add_trace(go.Scatter(
                    x=[event['onset']], y=[onset_spi], mode='markers+text',
                    marker=dict(symbol='triangle-down', size=12, color='darkred', line=dict(width=2, color='black')),
                    name=f'Onset: {event["onset"].strftime("%Y-%m")}', text=[f'Onset<br>Sev: {event["severity"]:.1f}'],
                    textposition='top center', textfont=dict(size=9, color='darkred'),
                    hovertemplate=f'Onset: {event["onset"].strftime("%Y-%m-%d")}<br>SPI: {onset_spi:.2f}<br>Severity: {event["severity"]:.1f}<br>Duration: {event["duration"]} months<extra></extra>'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[event['termination']], y=[termination_spi], mode='markers+text',
                    marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='black')),
                    name=f'Termination: {event["termination"].strftime("%Y-%m")}', text=[f'End<br>Dur: {int(event["duration"])}m'],
                    textposition='bottom center', textfont=dict(size=9, color='green'),
                    hovertemplate=f'Termination: {event["termination"].strftime("%Y-%m-%d")}<br>SPI: {termination_spi:.2f}<br>Severity: {event["severity"]:.1f}<br>Duration: {event["duration"]} months<extra></extra>'
                ))
            except Exception:
                continue
    else:
        st.info("No drought events met the criteria for this station.")

    fig.update_layout(
        height=500, xaxis_range=st.session_state[zoom_key], yaxis_title="SPI-12 Value", xaxis_title="Date",
        template="plotly_white", hovermode="x unified", yaxis=dict(range=[-3.5, 3.5])
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### 📋 Detected Drought Events")
    
    if len(stn_events) > 0:
        st.markdown("🔍 **Click an event below to focus the graph on that specific drought period:**")
        
        cols = st.columns(4)
        for idx, event in stn_events.iterrows():
            col_idx = idx % 4
            with cols[col_idx]:
                btn_label = f"{event['category']} ({event['onset'].strftime('%Y-%m')})"
                if st.button(btn_label, key=f"btn_drought_{selected_station}_{idx}"):
                    padding = pd.Timedelta(days=90)
                    st.session_state[zoom_key] = [event['onset'] - padding, event['termination'] + padding]
                    st.rerun()
                    
        if st.button("🔄 Reset Graph to Full Historical View", use_container_width=True):
            st.session_state[zoom_key] = [stn_data.index.min(), stn_data.index.max()]
            st.rerun()
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        display_df = stn_events[['onset', 'termination', 'duration', 'severity', 'category', 'return_period_years']].copy()
        display_df['return_period_years'] = display_df['return_period_years'].round(1)
        display_df['onset'] = display_df['onset'].dt.strftime('%Y-%m')
        display_df['termination'] = display_df['termination'].dt.strftime('%Y-%m')
        
        st.dataframe(
            display_df, 
            hide_index=True, 
            width="stretch",
            column_config={
                "onset": st.column_config.TextColumn("Onset", help="The month and year the drought began (Index ≤ -1.0)."),
                "termination": st.column_config.TextColumn("Termination", help="The month and year the drought ended (Index > -1.0)."),
                "duration": st.column_config.NumberColumn("Duration (Months)", help="Total consecutive months the drought persisted."),
                "severity": st.column_config.NumberColumn("Severity", help="Cumulative absolute deficit of the index during the drought period."),
                "category": st.column_config.TextColumn("Category", help="Classification: Moderate (-1.0 to -1.49), Severe (-1.5 to -1.99), or Extreme (≤ -2.0)."),
                "return_period_years": st.column_config.NumberColumn("Return Period (Years)", help="Estimated average recurrence interval for a drought of this severity or worse at this station.")
            }
        )
    else:
        st.warning("No drought events found for the selected station based on the current thresholds.")

# ==============================================================================
# TAB 7: FUTURE DROUGHT RISK PROJECTION (CMIP6)
# ==============================================================================
with tab_future:
    st.markdown("### 🔮 Future Drought Risk Projection (CMIP6)")
    st.write("Assessing how climate change scenarios (SSPs) will alter drought frequency, severity, and return periods in Sarawak.")
    st.info("🔬 **Academic Standard Note:** Return periods represent the annual probability of exceedance (e.g., a 10-year return period = 10% chance per year). Under climate change, historical baselines no longer predict future risks (non-stationarity).")
    st.markdown("---")

    future_events_csv = project_root / "outputs" / "processed" / "future_drought_events_corrected.csv"
    data_csv = project_root / "outputs" / "processed" / "sarawak_ML_corrected.parquet"
    cmip6_dir = project_root / "outputs" / "processed" / "cmip6_ensemble"
    
    if not future_events_csv.exists() or not data_csv.exists():
        st.error("⚠️ Future drought data or historical metadata not found.")
        st.markdown("💡 *Please run the CMIP6 extraction and future drought analysis pipeline.*")
        st.stop()

    future_df = pd.read_csv(future_events_csv)
    future_df['onset'] = pd.to_datetime(future_df['onset'])
    future_df['termination'] = pd.to_datetime(future_df['termination'])
    future_df['station_id'] = future_df['station_id'].astype(str)
    
    df_full = pd.read_parquet(data_csv)
    df_full['station_id'] = df_full['station_id'].astype(str)
    
    hist_df = pd.read_csv(project_root / "outputs" / "processed" / "drought_events_summary.csv")
    hist_df['onset'] = pd.to_datetime(hist_df['onset'])
    hist_df['termination'] = pd.to_datetime(hist_df['termination'])
    hist_df['station_id'] = hist_df['station_id'].astype(str)

    st.markdown("#### 🎛️ Projection Filters & Station Info")
    col_f1, col_f2 = st.columns([1, 3])
    
    with col_f1:
        stations_with_events = future_df[['station_id', 'station_name']].drop_duplicates()
        station_map = dict(zip(stations_with_events['station_id'], stations_with_events['station_name']))
        display_options = [f"{name} ({sid})" for sid, name in station_map.items()]
        display_options.sort()
        
        selected_display = st.selectbox("Select Station:", options=display_options)
        selected_station_id = selected_display.split('(')[-1].strip(')')
        
        stn_meta = df_full[df_full['station_id'] == selected_station_id].iloc[0]
        stn_lulc_name = stn_meta.get('Land_Use')
        stn_lulc_code = stn_meta.get('lulc_class')
        stn_lulc = LULC_MAP.get(int(stn_lulc_code), 'Unknown') if pd.notna(stn_lulc_code) and pd.isna(stn_lulc_name) else (stn_lulc_name if pd.notna(stn_lulc_name) else 'Unknown')

        st.markdown("##### 📍 Station Info")
        st.metric("Elevation", f"{stn_meta['elev_m']:.1f} m")
        st.metric("Land Cover", stn_lulc)

    with col_f2:
        map_df = pd.DataFrame({'lat': [stn_meta['lat']], 'lon': [stn_meta['lon']], 'station': [selected_display]})
        view_state = pdk.ViewState(latitude=stn_meta['lat'], longitude=stn_meta['lon'], zoom=10, pitch=0)
        layers = [pdk.Layer("ScatterplotLayer", data=map_df, get_position="[lon, lat]", get_color="[255, 50, 50, 255]", get_radius=2000, pickable=True)]
        st.pydeck_chart(pdk.Deck(map_style='road', initial_view_state=view_state, layers=layers, tooltip={"html": f"<b>Station:</b> {selected_display}"}, height=250))

    st.markdown("---")
    
    col_f3, col_f4, col_f5 = st.columns(3)
    with col_f3:
        available_models = sorted(future_df['model'].dropna().unique())
        selected_model = st.selectbox("Select GCM Model:", options=available_models)
    with col_f4:
        available_scenarios = sorted(future_df['scenario'].dropna().unique())
        selected_scenario = st.selectbox("Select SSP Scenario:", options=available_scenarios)
    with col_f5:
        available_timeframes = sorted(future_df['timeframe'].dropna().unique())
        
        if selected_scenario == 'historical':
            valid_timeframes = ['historical']
            default_idx = 0
        else:
            valid_timeframes = [tf for tf in available_timeframes if tf in ['near', 'far']]
            default_idx = valid_timeframes.index('far') if 'far' in valid_timeframes else 0
        
        selected_timeframe = st.selectbox(
            "Select Timeframe:", 
            options=valid_timeframes, 
            index=default_idx
        )

    future_stn = future_df[
        (future_df['station_id'] == selected_station_id) & 
        (future_df['model'] == selected_model) & 
        (future_df['scenario'] == selected_scenario) & 
        (future_df['timeframe'] == selected_timeframe)
    ].copy()
    
    hist_stn = hist_df[hist_df['station_id'] == selected_station_id].copy()

    if future_stn.empty:
        st.warning(f"⚠️ No drought events projected for **{selected_display}** under {selected_model} ({selected_scenario}) for the {selected_timeframe} period. Try a different model or scenario.")
    else:
        st.markdown("#### 📈 1. Projected Timeline")
        
        index_type = st.radio("Select Drought Index:", ["SPI-12", "SPEI-12"], horizontal=True, help="SPI uses only precipitation. SPEI includes Potential Evapotranspiration (PET) to account for temperature-driven aridification.")
        
        st.caption(f"Future {index_type} trajectory based on ML-corrected CMIP6 data. Red bars indicate drought months (≤ -1.0). Markers show projected onset (▼) and termination (▲).")
        
        cmip6_file = cmip6_dir / f"sarawak_cmip6_{selected_model}_{selected_scenario}_{selected_timeframe}.parquet"
        
        if cmip6_file.exists():
            future_cmip6 = pd.read_parquet(cmip6_file)
            future_cmip6 = future_cmip6[future_cmip6['station_id'] == selected_station_id].sort_values('date').set_index('date')
            
            from src.drought_indices import calculate_spi, calculate_spei
            
            with st.spinner(f"Calculating future {index_type} for {selected_timeframe}..."):
                if index_type == "SPI-12":
                    future_cmip6['INDEX_12'] = calculate_spi(future_cmip6['P_future'], scale=12)
                else:
                    future_cmip6['INDEX_12'] = calculate_spei(future_cmip6['P_future'], future_cmip6['PET_future'], scale=12)
            
            fig_future_idx = go.Figure()
            colors = ['red' if v <= -1.0 else '#1f77b4' for v in future_cmip6['INDEX_12']]
            
            fig_future_idx.add_trace(go.Bar(
                x=future_cmip6.index, y=future_cmip6['INDEX_12'], name=f'Future {index_type}',
                marker_color=colors, opacity=0.8
            ))
            
            fig_future_idx.add_hline(y=-1.0, line_dash="dash", line_color="black", line_width=2)
            fig_future_idx.add_hline(y=-1.5, line_dash="dot", line_color="darkred", line_width=1.5)
            fig_future_idx.add_hline(y=-2.0, line_dash="dot", line_color="red", line_width=1.5)
            fig_future_idx.add_hline(y=0, line_dash="dot", line_color="gray")

            for i, event in future_stn.iterrows():
                try:
                    onset_idx = future_cmip6.loc[event['onset'], 'INDEX_12']
                    termination_idx = future_cmip6.loc[event['termination'], 'INDEX_12']
                    
                    fig_future_idx.add_vrect(x0=event['onset'], x1=event['termination'], fillcolor="red", opacity=0.15, line_width=0)
                    
                    fig_future_idx.add_trace(go.Scatter(
                        x=[event['onset']], y=[onset_idx], mode='markers',
                        marker=dict(symbol='triangle-down', size=12, color='darkred', line=dict(width=2, color='black')),
                        name=f'Onset: {event["onset"].strftime("%Y-%m")}',
                        hovertemplate=f'Onset: {event["onset"].strftime("%Y-%m-%d")}<br>{index_type}: {onset_idx:.2f}<br>Severity: {event["severity"]:.1f}<br>Duration: {int(event["duration"])} months<extra></extra>',
                        showlegend=False
                    ))
                    
                    fig_future_idx.add_trace(go.Scatter(
                        x=[event['termination']], y=[termination_idx], mode='markers',
                        marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=2, color='black')),
                        name=f'Termination: {event["termination"].strftime("%Y-%m")}',
                        hovertemplate=f'Termination: {event["termination"].strftime("%Y-%m-%d")}<br>{index_type}: {termination_idx:.2f}<br>Severity: {event["severity"]:.1f}<br>Duration: {int(event["duration"])} months<extra></extra>',
                        showlegend=False
                    ))
                except Exception:
                    continue

            fig_future_idx.update_layout(
                height=400, yaxis_title=f"{index_type} Value", xaxis_title="Date",
                template="plotly_white", hovermode="x unified", yaxis=dict(range=[-3.5, 3.5])
            )
            st.plotly_chart(fig_future_idx, width="stretch")
        else:
            st.warning(f"⚠️ CMIP6 data file not found for this combination.")

        st.markdown("---")

        st.markdown("#### 📉 2. Return Period Shift: Historical vs. Future")
        st.caption("Compares the median return period of historical gauge droughts to future CMIP6 droughts. Bars **lower in the future** (red) indicate droughts are becoming **more frequent**.")
        
        cat_order = ['Moderate', 'Severe', 'Extreme']
        hist_agg = hist_stn[hist_stn['category'].isin(cat_order)].groupby('category')['return_period_years'].median().reset_index()
        hist_agg['period'] = 'Historical Baseline'
        
        fut_agg = future_stn[future_stn['category'].isin(cat_order)].groupby('category')['return_period_years'].median().reset_index()
        fut_agg['period'] = f'Future ({selected_timeframe})'
        
        comp_df = pd.concat([hist_agg, fut_agg], ignore_index=True)
        for cat in cat_order:
            if cat not in comp_df['category'].values:
                comp_df = pd.concat([comp_df, pd.DataFrame({'category': [cat], 'return_period_years': [np.nan], 'period': ['Historical Baseline']})])
                comp_df = pd.concat([comp_df, pd.DataFrame({'category': [cat], 'return_period_years': [np.nan], 'period': [f'Future ({selected_timeframe})']})])
        
        comp_df = comp_df.dropna(subset=['return_period_years'])
        
        if not comp_df.empty:
            fig_rp = px.bar(
                comp_df, x='category', y='return_period_years', color='period',
                barmode='group', text='return_period_years',
                category_orders={'category': cat_order},
                labels={'return_period_years': 'Median Return Period (Years)', 'category': 'Drought Category'},
                color_discrete_map={'Historical Baseline': '#1f77b4', f'Future ({selected_timeframe})': '#d62728'},
                height=400
            )
            fig_rp.update_traces(texttemplate='%{text:.1f} yrs', textposition='outside')
            fig_rp.update_layout(xaxis_title="Drought Severity Category", yaxis_title="Median Return Period (Years)")
            st.plotly_chart(fig_rp, width="stretch")
        else:
            st.info("Insufficient data to plot return period shift for the selected categories.")

        st.markdown("---")

        st.markdown(f"#### 📅 3. Projected Drought Risk Timeline: {selected_timeframe}")
        st.caption("Gantt-style visualization of projected drought events. Hover over bars to see the projected Return Period for each specific event.")
        
        future_stn_sorted = future_stn.sort_values('onset')
        future_stn_sorted['start_date'] = future_stn_sorted['onset'].dt.strftime('%Y-%m')
        future_stn_sorted['end_date'] = future_stn_sorted['termination'].dt.strftime('%Y-%m')
        
        color_map = {'Moderate': '#fdae61', 'Severe': '#d7191c', 'Extreme': '#7f0000'}
        future_stn_sorted['color'] = future_stn_sorted['category'].map(color_map)
        
        fig_gantt = go.Figure()
        added_categories = set()
        
        for i, row in future_stn_sorted.iterrows():
            show_legend = row['category'] not in added_categories
            if show_legend:
                added_categories.add(row['category'])
                
            fig_gantt.add_trace(go.Scatter(
                x=[row['start_date'], row['end_date']],
                y=[row['category'], row['category']],
                mode='lines',
                line=dict(color=row['color'], width=25),
                name=row['category'],
                hovertemplate=f"<b>{row['category']} Drought</b><br>" +
                              f"Start: {row['start_date']}<br>" +
                              f"End: {row['end_date']}<br>" +
                              f"Duration: {row['duration']} months<br>" +
                              f"Severity: {row['severity']:.1f}<br>" +
                              f"<b>Return Period: 1-in-{row['return_period_years']:.1f} years</b><extra></extra>",
                showlegend=show_legend
            ))
            
        fig_gantt.update_layout(
            height=350, xaxis_title="Time", yaxis_title="Drought Category",
            yaxis={'categoryorder': 'array', 'categoryarray': ['Extreme', 'Severe', 'Moderate']},
            template="plotly_white", showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_gantt, width="stretch")
        
        st.markdown("#### 📋 Projected Drought Events Table")
        display_fut = future_stn_sorted[['onset', 'termination', 'duration', 'severity', 'category', 'return_period_years']].copy()
        display_fut['return_period_years'] = display_fut['return_period_years'].round(1)
        display_fut['onset'] = display_fut['onset'].dt.strftime('%Y-%m')
        display_fut['termination'] = display_fut['termination'].dt.strftime('%Y-%m')
        
        st.dataframe(
            display_fut, hide_index=True, width="stretch",
            column_config={
                "onset": st.column_config.TextColumn("Onset"),
                "termination": st.column_config.TextColumn("Termination"),
                "duration": st.column_config.NumberColumn("Duration (Months)"),
                "severity": st.column_config.NumberColumn("Severity"),
                "category": st.column_config.TextColumn("Category"),
                "return_period_years": st.column_config.NumberColumn("Return Period (Years)", help="Estimated average recurrence interval for a drought of this severity or worse in the future climate.")
            }
        )
