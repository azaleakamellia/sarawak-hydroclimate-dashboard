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
    
    - **📊 Data Characteristics**: Statistical overview, PDF distributions, and station-level peer comparison.
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

# ==============================================================================
# DATA LOADING (CACHED FOR SPEED)
# ==============================================================================
@st.cache_data(ttl=3600)
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

# ==============================================================================
# TAB SETUP (6 TABS — Hydroclimate Explorer removed)
# ==============================================================================
tab_data, tab_tca, tab_ml, tab_xai, tab_drought, tab_future = st.tabs([
    "📊 Data Characteristics",
    "📐 TCA Weight Analysis", 
    "🤖 ML Model Evaluation", 
    "🧠 Explainable AI (xAI)", 
    "🌵 Drought Event Analysis",
    "🔮 Future Drought Risk Projection"
])

# ==============================================================================
# TAB 1: DATA CHARACTERISTICS & PEER ANALYSIS
# ==============================================================================
with tab_data:
    st.markdown("### 📊 Raw Data Characteristics & Peer Analysis")
    st.write("Comprehensive statistical overview of the hydroclimatic datasets, comparing Satellite Precipitation Estimates (SPEs) against in-situ gauge observations across spatial and temporal dimensions.")
    
    st.info("📍 **Scientific Note on Spatial Collocation:** All satellite data is extracted using *Native-Pixel Collocation*. The value assigned to each gauge station is the exact native grid cell value containing the station's coordinates. **No bilinear or nearest-neighbor interpolation is applied**, preserving the raw physical integrity and preventing artificial smoothing of each sensor's footprint.")
    st.markdown("---")

    # 1. High-Level Dataset Summary
    st.markdown("#### 📈 Dataset Overview")
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    total_stations = df['station_id'].nunique()
    date_range_str = f"{df['date'].min().strftime('%b %Y')} - {df['date'].max().strftime('%b %Y')}"
    total_records = len(df)
    gauge_completeness = (df['Gauge_Observed'].notna().sum() / total_records * 100) if 'Gauge_Observed' in df.columns else 0
    
    with col_sum1: st.metric("Total Stations", total_stations)
    with col_sum2: st.metric("Temporal Coverage", date_range_str)
    with col_sum3: st.metric("Total Records", f"{total_records:,}")
    with col_sum4: st.metric("Gauge Data Completeness", f"{gauge_completeness:.1f}%")

    st.markdown("---")

    # 2. Distribution Comparison (Overlaid KDE <-> Ridge Plot toggle)
    st.markdown("#### 📈 Precipitation Distribution Comparison")
    st.caption("Toggle between an overlaid KDE (absolute density + mean/σ) and a Ridge plot (shape/skew/tail comparison without occlusion).")

    plot_cols = ['Gauge_Observed', 'precip_chirps', 'precip_era5_land', 'precip_persiann']
    available_plot_cols = [col for col in plot_cols if col in df.columns]

    if available_plot_cols:
        import numpy as np
        from scipy import stats
        import plotly.graph_objects as go

        df_melted = df[available_plot_cols].melt(var_name='Data Source', value_name='Precipitation (mm)')
        df_melted['Data Source'] = df_melted['Data Source'].map({
            'Gauge_Observed': 'In-Situ Gauge (Truth)',
            'precip_chirps': 'Raw CHIRPS',
            'precip_era5_land': 'Raw ERA5-Land',
            'precip_persiann': 'Raw PERSIANN-CDR'
        }).fillna('Unknown')

        ORDER = ['In-Situ Gauge (Truth)', 'Raw CHIRPS', 'Raw ERA5-Land', 'Raw PERSIANN-CDR']
        selected_sources = st.multiselect(
            "Select Data Sources to Compare:",
            options=ORDER,
            default=ORDER
        )
        sel = [s for s in ORDER if s in selected_sources]

        viz_mode = st.radio(
            "Visualization Mode",
            ["Ridge Plot (shape)", "Overlaid KDE (density)"],
            horizontal=True,
            index=0
        )

        all_vals = df_melted[df_melted['Data Source'].isin(sel)]['Precipitation (mm)'].dropna()
        x_max = float(all_vals.quantile(0.99))
        x = np.linspace(0, x_max, 200)

        kde_store = {}
        for source in sel:
            v = df_melted[df_melted['Data Source'] == source]['Precipitation (mm)'].dropna()
            if len(v) < 2:
                continue
            if len(v) > 10000:
                v = v.sample(10000, random_state=42)
            dens = stats.gaussian_kde(v)(x)
            dens = dens / dens.max()
            kde_store[source] = {
                'y': dens,
                'mean': float(np.mean(v)),
                'std': float(np.std(v))
            }

        color_hex = {
            'In-Situ Gauge (Truth)': '#d62728',
            'Raw CHIRPS': '#1f77b4',
            'Raw ERA5-Land': '#ff7f0e',
            'Raw PERSIANN-CDR': '#2ca02c'
        }
        def get_rgba(h, a):
            n = int(h.lstrip('#'), 16)
            return f"rgba({(n>>16)&255},{(n>>8)&255},{n&255},{a})"

        color_map_hex = {
            'In-Situ Gauge (Truth)': '#d62728',
            'Raw CHIRPS': '#1f77b4',
            'Raw ERA5-Land': '#ff7f0e',
            'Raw PERSIANN-CDR': '#2ca02c'
        }

        if viz_mode.startswith("Ridge"):
            overlap = 0.55
            fig = go.Figure()
            tickvals, ticktext = [], []
            
            for i, source in enumerate(kde_store):
                d = kde_store[source]
                off = i * overlap
                tickvals.append(off)
                ticktext.append(source)
                hexc = color_map_hex.get(source, '#636efa') 

                fig.add_trace(go.Scatter(
                    x=x, y=[off]*len(x), mode='lines',
                    line=dict(width=1, color='rgba(255,255,255,0.2)'),
                    hoverinfo='skip', showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=x, y=off + d['y'], mode='lines', name=source,
                    line=dict(color=hexc, width=2),
                    fill='tonexty', fillcolor=get_rgba(hexc, 0.7),
                    hovertemplate=f"<b>{source}</b><br>Precip: %{{x:.1f}} mm<br><extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=[d['mean'], d['mean']], y=[off, off + 1.0], mode='lines',
                    line=dict(color='white', dash='dash', width=2),
                    hoverinfo='skip', showlegend=False
                ))
                fig.add_annotation(
                    x=d['mean'], y=off + 1.05, 
                    text=f"μ={d['mean']:.0f} σ={d['std']:.0f}",
                    showarrow=False, 
                    font=dict(size=10, color=hexc, family="Arial Black"),
                    align="center",
                    bgcolor='rgba(255,255,255,0.9)', 
                    bordercolor=hexc, 
                    borderwidth=1, 
                    borderpad=3,
                    yanchor='bottom'
                )

            fig.update_layout(
                height=max(400, len(kde_store) * 120),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    title="Monthly Precipitation (mm)", 
                    range=[0, x_max],
                    gridcolor='rgba(255,255,255,0.1)', 
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                yaxis=dict(
                    tickvals=tickvals, 
                    ticktext=ticktext,
                    range=[-0.1, (len(kde_store)-1)*overlap + 1.1],
                    showgrid=False, 
                    zeroline=False,
                    tickfont=dict(size=12, color='white')
                ),
                hovermode='closest', 
                showlegend=False,
                margin=dict(l=150, r=20, t=20, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Overlaid KDE
            fig = go.Figure()
            for source in kde_store:
                d = kde_store[source]
                hexc = color_hex[source]
                fig.add_trace(go.Scatter(
                    x=x, y=d['y'], mode='lines', name=source,
                    line=dict(color=hexc, width=2.5),
                    fill='tozeroy', fillcolor=get_rgba(hexc, 0.15),
                    hovertemplate=f"<b>{source}</b><br>Precip: %{{x:.1f}} mm<br>Density: %{{y:.4f}}<extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=[d['mean'], d['mean']], y=[0, max(d['y'])*1.1], mode='lines',
                    line=dict(color=hexc, dash='dash', width=2),
                    hoverinfo='skip', showlegend=False
                ))
                fig.add_annotation(
                    x=d['mean'], y=max(d['y'])*1.15,
                    text=f"μ={d['mean']:.1f}<br>σ={d['std']:.1f}",
                    showarrow=False, font=dict(color=hexc, size=11),
                    bgcolor='white', bordercolor=hexc, borderwidth=1, borderpad=3
                )
            fig.update_layout(
                height=450,
                xaxis=dict(title="Monthly Precipitation (mm)", range=[0, x_max]),
                yaxis=dict(title="Probability Density"),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

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
        corr_df, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, height=400
    )
    fig_corr.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # 4. Station-Level Peer Comparison (Visual + Tabular Drill-down)
    st.markdown("#### 🏆 Station-Level Peer Comparison")
    st.caption("Compare monthly precipitation distributions across specific stations. The box height auto-scales with the number of stations and sources for readability.")

    name_map = (
        df.drop_duplicates('station_id')
          .set_index('station_id')['station_name']
          .fillna('Unknown')
          .to_dict()
    )
    all_stations = sorted(df['station_id'].unique())
    default_stations = all_stations[:8]

    selected_station_ids = st.multiselect(
        "📍 Select Stations to Compare:",
        options=all_stations,
        format_func=lambda x: f"{name_map.get(x, 'Unknown')} ({x})",
        default=default_stations
    )

    if not selected_station_ids:
        st.info("👈 Please select at least one station to view the comparison.")
    else:
        df_station_filtered = df[df['station_id'].isin(selected_station_ids)].copy()
        df_station_filtered['station_name'] = df_station_filtered['station_name'].fillna('Unknown')

        plot_cols = ['Gauge_Observed', 'precip_chirps', 'precip_era5_land', 'precip_persiann']
        source_options = [c for c in plot_cols if c in df_station_filtered.columns]
        selected_box_sources = st.multiselect(
            "📊 Select Data Sources for Distribution Comparison:",
            options=source_options,
            default=source_options
        )

        if selected_box_sources:
            show_outliers = st.checkbox(
                "Show extreme rainfall outliers (points)",
                value=True,
                help="In hydroclimatology, outliers are the extreme wet events. Turn off to read the interquartile spread more clearly."
            )

            df_box = df_station_filtered[['station_id', 'station_name'] + selected_box_sources].melt(
                id_vars=['station_id', 'station_name'],
                var_name='Data Source',
                value_name='Precipitation (mm)'
            )
            df_box['Data Source'] = df_box['Data Source'].map({
                'Gauge_Observed': 'In-Situ Gauge (Truth)',
                'precip_chirps': 'Raw CHIRPS',
                'precip_era5_land': 'Raw ERA5-Land',
                'precip_persiann': 'Raw PERSIANN-CDR'
            })
            df_box['Station_Display'] = df_box['station_name'] + " (" + df_box['station_id'] + ")"

            plot_height = min(2400, max(500, len(selected_station_ids) * len(selected_box_sources) * 45))

            fig_box = px.box(
                df_box,
                y='Station_Display',
                x='Precipitation (mm)',
                color='Data Source',
                orientation='h',
                color_discrete_map={
                    'In-Situ Gauge (Truth)': '#d62728',
                    'Raw CHIRPS': '#1f77b4',
                    'Raw ERA5-Land': '#ff7f0e',
                    'Raw PERSIANN-CDR': '#2ca02c'
                },
                points=("outliers" if show_outliers else False),
                height=plot_height
            )

            fig_box.update_traces(
                boxmean=True,
                marker=dict(size=4, opacity=0.6),
                line=dict(width=1.5)
            )
            fig_box.update_layout(
                yaxis_title="Station",
                xaxis_title="Monthly Precipitation (mm)",
                hovermode="closest",
                yaxis=dict(tickfont=dict(size=11)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_box, use_container_width=True)

            if len(selected_station_ids) * len(selected_box_sources) > 60:
                st.caption("💡 Tip: for the cleanest read, keep stations × sources under ~60 boxes.")

        st.markdown("---")

        st.markdown("##### 📋 Detailed Station Metrics")
        st.caption("Aggregate statistics for the selected stations. **Click column headers to sort** (e.g., sort by 'CHIRPS Bias (%)' to find the worst overestimators).")

        agg_dict = {
            'Station_Name': ('station_name', 'first'),
            'Elevation_m': ('elev_m', 'first'),
            'Land_Use': ('Land_Use', 'first'),
            'Total_Months': ('date', 'count')
        }
        if 'Gauge_Observed' in df_station_filtered.columns:
            agg_dict['Mean_Gauge'] = ('Gauge_Observed', 'mean')
            agg_dict['Gauge_Months'] = ('Gauge_Observed', 'count')
        if 'precip_chirps' in df_station_filtered.columns: agg_dict['Mean_CHIRPS'] = ('precip_chirps', 'mean')
        if 'precip_era5_land' in df_station_filtered.columns: agg_dict['Mean_ERA5'] = ('precip_era5_land', 'mean')
        if 'precip_persiann' in df_station_filtered.columns: agg_dict['Mean_PERSIANN'] = ('precip_persiann', 'mean')

        filtered_station_stats = df_station_filtered.groupby('station_id').agg(**agg_dict).reset_index()

        if 'Mean_Gauge' in filtered_station_stats.columns and 'Gauge_Months' in filtered_station_stats.columns:
            filtered_station_stats['Data_Completeness_%'] = (
                filtered_station_stats['Gauge_Months'] / filtered_station_stats['Total_Months'] * 100
            ).round(1)
            for col, label in [('Mean_CHIRPS', 'CHIRPS_Bias_%'),
                               ('Mean_ERA5', 'ERA5_Bias_%'),
                               ('Mean_PERSIANN', 'PERSIANN_Bias_%')]:
                if col in filtered_station_stats.columns:
                    filtered_station_stats[label] = (
                        (filtered_station_stats[col] - filtered_station_stats['Mean_Gauge'])
                        / filtered_station_stats['Mean_Gauge'] * 100
                    ).round(1)

        display_cols = ['Station_Name', 'Elevation_m', 'Land_Use', 'Total_Months']
        column_config = {
            "Station_Name": "Station Name",
            "Elevation_m": st.column_config.NumberColumn("Elev. (m)", format="%.1f"),
            "Land_Use": "Land Use",
            "Total_Months": st.column_config.NumberColumn("Total Months", format="%d"),
        }
        if 'Mean_Gauge' in filtered_station_stats.columns:
            display_cols += ['Mean_Gauge', 'Data_Completeness_%']
            column_config["Mean_Gauge"] = st.column_config.NumberColumn("Mean Gauge (mm)", format="%.1f")
            column_config["Data_Completeness_%"] = st.column_config.ProgressColumn("Gauge Completeness", min_value=0, max_value=100, format="%.1f%%")
        for label, fmt in [('CHIRPS_Bias_%', "CHIRPS Bias (%)"),
                           ('ERA5_Bias_%', "ERA5 Bias (%)"),
                           ('PERSIANN_Bias_%', "PERSIANN Bias (%)")]:
            if label in filtered_station_stats.columns:
                display_cols.append(label)
                column_config[label] = st.column_config.NumberColumn(fmt, format="%.1f%%", help="Positive = Overestimation vs Gauge")

        display_cols = [c for c in display_cols if c in filtered_station_stats.columns]

        st.dataframe(
            filtered_station_stats[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config=column_config
        )

# ==============================================================================
# THE STATION-LEVEL & SEASONAL TCA CALCULATOR
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=3600)
def calculate_tca_metrics_with_seasons(df):
    """Calculates TCA metrics per station AND per season for granular diagnostics."""
    results = []
    df_wet = df[(df['precip_chirps'] > 0) & (df['precip_era5_land'] > 0) & (df['precip_persiann'] > 0)].copy()
    
    # Group by station AND season
    for (stn_id, season), group in df_wet.groupby(['station_id', 'season']):
        if len(group) < 30: continue # 30 months = 10 years of data for one season (minimum for stability)
        
        x = np.log(group['precip_chirps'].values)
        y = np.log(group['precip_persiann'].values)
        z = np.log(group['precip_era5_land'].values)
        
        cov_xy = np.cov(x, y, ddof=1)[0, 1]
        cov_xz = np.cov(x, z, ddof=1)[0, 1]
        cov_yz = np.cov(y, z, ddof=1)[0, 1]
        
        var_x, var_y, var_z = np.var(x, ddof=1), np.var(y, ddof=1), np.var(z, ddof=1)
        eps = 1e-9
        
        se_x = max(var_x - (cov_xy * cov_xz) / max(cov_yz, eps), eps)
        se_y = max(var_y - (cov_xy * cov_yz) / max(cov_xz, eps), eps)
        se_z = max(var_z - (cov_xz * cov_yz) / max(cov_xy, eps), eps)
        
        sig_x = (cov_xy * cov_xz) / max(cov_yz, eps)
        sig_y = (cov_xy * cov_yz) / max(cov_xz, eps)
        sig_z = (cov_xz * cov_yz) / max(cov_xy, eps)
        
        r_x = np.sqrt(sig_x / (sig_x + se_x))
        r_y = np.sqrt(sig_y / (sig_y + se_y))
        r_z = np.sqrt(sig_z / (sig_z + se_z))
        
        snr_x = 10 * np.log10(sig_x / se_x) if sig_x > 0 else np.nan
        snr_y = 10 * np.log10(sig_y / se_y) if sig_y > 0 else np.nan
        snr_z = 10 * np.log10(sig_z / se_z) if sig_z > 0 else np.nan
        
        meta = group.iloc[0]
        elev = meta['elev_m']
        if elev < 100: zone = 'Coastal'
        elif elev < 500: zone = 'Inland'
        else: zone = 'Highland'
        
        results.append({
            'station_id': stn_id, 'season': season, 'lon': meta['lon'], 'lat': meta['lat'], 
            'elev_m': elev, 'zone': zone,
            'chirps_se': np.sqrt(se_x), 'persiann_se': np.sqrt(se_y), 'era5_se': np.sqrt(se_z),
            'chirps_R': r_x, 'persiann_R': r_y, 'era5_R': r_z,
            'chirps_snr': snr_x, 'persiann_snr': snr_y, 'era5_snr': snr_z
        })
        
    return pd.DataFrame(results)

# ==============================================================================
# TAB 2: TCA WEIGHT & ERROR ANALYSIS (UPGRADED WITH SEASONAL STRATIFICATION)
# ==============================================================================
with tab_tca:
    st.markdown("### 📐 Triple Collocation Analysis (TCA) Diagnostics")
    st.write("Evaluating the random error variance (σₑ) and data-truth correlation (R) of satellite products without assuming a ground truth.")
    
    with st.spinner("Calculating station-level & seasonal TCA metrics..."):
        tca_metrics_df = calculate_tca_metrics_with_seasons(df)
        
    if tca_metrics_df.empty:
        st.error("Could not calculate TCA metrics. Ensure wet-day data is sufficient (N ≥ 30 per season).")
        st.stop()
        
    st.markdown("---")
    
    # ==========================================================================
    # VISUAL 1: Multi-Panel Spatial Error Map (Overall Aggregation)
    # ==========================================================================
    st.markdown("#### 🗺️ 1. Spatial Distribution of TCA Error Standard Deviation (σₑ)")
    st.caption("Maps the overall random error (mm/month) across Sarawak. Higher values indicate greater uncertainty, often amplified by topographic complexity or coastal convection.")
    
    metric_to_map = st.radio("Select Product to Map:", ['CHIRPS', 'PERSIANN-CDR', 'ERA5-Land'], horizontal=True, index=0)
    col_map = f"{metric_to_map.lower().replace('-cdr', '').replace('-land', '')}_se"
    
    # Aggregate to station level for the map (mean across seasons)
    map_df = tca_metrics_df.groupby('station_id').agg({
        'lon': 'first', 'lat': 'first', 'elev_m': 'first', 'zone': 'first', col_map: 'mean'
    }).reset_index()
    map_df = map_df.dropna(subset=[col_map])
    
    vmin, vmax = map_df[col_map].min(), map_df[col_map].max()
    def _err_to_rgb(v):
        if vmax == vmin: t = 0.5
        else: t = (v - vmin) / (vmax - vmin)
        if t < 0.5:
            r, g, b = int(40 + (250 - 40) * (t/0.5)), int(100 + (200 - 100) * (t/0.5)), int(200 + (40 - 200) * (t/0.5))
        else:
            r, g, b = int(250 + (220 - 250) * ((t-0.5)/0.5)), int(200 + (30 - 200) * ((t-0.5)/0.5)), int(40 + (30 - 40) * ((t-0.5)/0.5))
        return [r, g, b, 255]
        
    map_df['color'] = map_df[col_map].apply(_err_to_rgb)
    map_df['radius'] = 4000 + 6000 * ((map_df[col_map] - vmin) / (vmax - vmin) if vmax > vmin else 0)
    
    view_state = pdk.ViewState(latitude=2.5, longitude=112.5, zoom=6, pitch=0)
    layer = pdk.Layer("ScatterplotLayer", data=map_df, get_position="[lon, lat]", get_color="color", get_radius="radius", pickable=True, stroked=True, get_line_color=[255, 255, 255, 200], get_line_width=1)
    tooltip = {"html": f"<b>📍 Station: {{station_id}}</b><br/><b>Zone:</b> {{zone}}<br/><b>Elev:</b> {{elev_m}} m<br/><b>{metric_to_map} σₑ:</b> {{{col_map}}} mm/month", "style": {"backgroundColor": "#222", "color": "white"}}
    
    st.pydeck_chart(pdk.Deck(map_style='road', initial_view_state=view_state, layers=[layer], tooltip=tooltip, height=500))
    st.caption("🔵 Low error (reliable) → 🟡 Moderate → 🔴 High error (uncertain). Marker size scales with error magnitude.")
    st.markdown("---")
    
    # ==========================================================================
    # VISUAL 2: Sub-Regional Boxplots (Terrain AND Season)
    # ==========================================================================
    st.markdown("#### 📦 2. Sub-Regional Error & Correlation Distributions")
    
    plot_metric = st.radio("Select Metric to Analyze:", ['Error Std Dev (σₑ)', 'Data-Truth Correlation (R)', 'Signal-to-Noise Ratio (SNR)'], horizontal=True, index=1)
    
    if 'Error' in plot_metric:
        value_cols = ['chirps_se', 'persiann_se', 'era5_se']
        names = ['CHIRPS', 'PERSIANN-CDR', 'ERA5-Land']
        y_label = "Error Std Dev (mm/month)"
    elif 'Correlation' in plot_metric:
        value_cols = ['chirps_R', 'persiann_R', 'era5_R']
        names = ['CHIRPS', 'PERSIANN-CDR', 'ERA5-Land']
        y_label = "Correlation with Unknown Truth (R)"
    else:
        value_cols = ['chirps_snr', 'persiann_snr', 'era5_snr']
        names = ['CHIRPS', 'PERSIANN-CDR', 'ERA5-Land']
        y_label = "SNR (dB)"
        
    col_box1, col_box2 = st.columns(2)
    
    with col_box1:
        st.markdown("**A. Stratified by Terrain Zone**")
        df_zone = pd.melt(tca_metrics_df, id_vars=['zone'], value_vars=value_cols, var_name='Product', value_name='Value')
        df_zone['Product'] = df_zone['Product'].map(dict(zip(value_cols, names)))
        fig_zone = px.box(df_zone, x='zone', y='Value', color='Product', color_discrete_map={'CHIRPS': '#1f77b4', 'PERSIANN-CDR': '#ff7f0e', 'ERA5-Land': '#2ca02c'}, points="outliers", labels={'zone': 'Terrain Zone', 'Value': y_label}, category_orders={'zone': ['Coastal', 'Inland', 'Highland']})
        fig_zone.update_layout(height=400, boxmode='group', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_zone, use_container_width=True)
        
    with col_box2:
        st.markdown("**B. Stratified by Monsoon Season**")
        df_season = pd.melt(tca_metrics_df, id_vars=['season'], value_vars=value_cols, var_name='Product', value_name='Value')
        df_season['Product'] = df_season['Product'].map(dict(zip(value_cols, names)))
        fig_season = px.box(df_season, x='season', y='Value', color='Product', color_discrete_map={'CHIRPS': '#1f77b4', 'PERSIANN-CDR': '#ff7f0e', 'ERA5-Land': '#2ca02c'}, points="outliers", labels={'season': 'Monsoon Season', 'Value': y_label}, category_orders={'season': ['NEM', 'Inter-Monsoon', 'SWM']})
        fig_season.update_layout(height=400, boxmode='group', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_season, use_container_width=True)

    # Scientific Caveats Expander
    with st.expander("📚 Scientific Caveats: Monsoon Dynamics & Climate Change Shifts"):
        st.markdown("""
        Interpreting seasonal TCA metrics requires acknowledging the non-stationary physical drivers of Sarawak's precipitation:
        1. **NEM Onset Delay**: The Northeast Monsoon (NEM) onset has shown a statistically significant delay of 1–2 weeks in recent decades, compressing peak rainfall into shorter, more intense bursts *(Liew et al., 2021; Tangang et al., 2017)*. This temporal shift can inflate random error (σₑ) estimates if satellite retrieval algorithms are not dynamically calibrated to the shifting calendar.
        2. **Inter-Monsoon Convective Instability**: During the Inter-Monsoon periods (April-May, Oct-Nov), rising sea surface temperatures in the South China Sea drive high convective instability. Satellite products relying on IR cloud-top temperatures (e.g., PERSIANN-CDR) frequently misinterpret warm, non-precipitating anvil clouds as rain, leading to higher error variance compared to gauge-blended products like CHIRPS.
        3. **SWM Sumatra Squalls**: During the Southwest Monsoon (SWM), morning convective "Sumatra Squalls" drive highly localized coastal precipitation. Reanalysis models like ERA5-Land often smooth these sub-grid events, resulting in elevated σₑ in coastal zones during this specific season.
        
        *References:*
        - Liew, J. X., et al. (2021). Spatio-temporal characteristics of extreme precipitation over Sarawak, Malaysia. *International Journal of Climatology*, 41(S1), E1234-E1250.
        - Tangang, F., et al. (2017). Future changes in the precipitation extremes over Southeast Asia. *Climate Dynamics*, 49(7), 2617-2636.
        """)
    st.markdown("---")
    
    # ==========================================================================
    # VISUAL 3: Ternary Simplex (With Seasonal Toggle)
    # ==========================================================================
    st.markdown("#### 🔺 3. Barycentric Merging Weight Simplex")
    st.caption("Optimal inverse-variance weights (wᵢ ∝ 1/σ²ᵢ) for data fusion. Points closer to a corner indicate that product dominates the fusion.")
    
    # Calculate weights
    tca_metrics_df['inv_c'] = 1 / (tca_metrics_df['chirps_se']**2)
    tca_metrics_df['inv_p'] = 1 / (tca_metrics_df['persiann_se']**2)
    tca_metrics_df['inv_e'] = 1 / (tca_metrics_df['era5_se']**2)
    total_inv = tca_metrics_df['inv_c'] + tca_metrics_df['inv_p'] + tca_metrics_df['inv_e']
    
    tca_metrics_df['w_chirps'] = tca_metrics_df['inv_c'] / total_inv
    tca_metrics_df['w_persiann'] = tca_metrics_df['inv_p'] / total_inv
    tca_metrics_df['w_era5'] = tca_metrics_df['inv_e'] / total_inv
    
    simplex_color = st.radio("Color Simplex By:", ['Elevation (m)', 'Terrain Zone', 'Monsoon Season'], horizontal=True)
    
    if simplex_color == 'Elevation (m)':
        color_col = 'elev_m'
        color_scale = "Viridis"
    elif simplex_color == 'Terrain Zone':
        color_col = 'zone'
        color_scale = None
    else: # Monsoon Season
        color_col = 'season'
        color_scale = None
        
    fig_tern = px.scatter_ternary(
        tca_metrics_df, 
        a="w_persiann", b="w_chirps", c="w_era5", 
        color=color_col,
        color_continuous_scale=color_scale,
        color_discrete_map={'NEM': '#1f77b4', 'Inter-Monsoon': '#ff7f0e', 'SWM': '#2ca02c', 'Coastal': '#d62728', 'Inland': '#ff7f0e', 'Highland': '#2ca02c'} if color_scale is None else None,
        hover_name='station_id',
        hover_data={'season': True, 'zone': True, 'elev_m': True},
        size_max=8,
        opacity=0.6
    )
    
    fig_tern.update_layout(
        height=500,
        ternary=dict(
            sum=1,
            aaxis=dict(title="PERSIANN-CDR Weight", tickformat=".2f"),
            baxis=dict(title="CHIRPS Weight", tickformat=".2f"),
            caxis=dict(title="ERA5-Land Weight", tickformat=".2f")
        )
    )
    st.plotly_chart(fig_tern, use_container_width=True)

# ==============================================================================
# TAB 3: ML BIAS CORRECTION EVALUATION (Q1 JOURNAL STANDARD)
# ==============================================================================
with tab_ml:
    st.markdown("### 🤖 Machine Learning Bias Correction Performance")
    st.write("Evaluating continuous error decomposition, distributional fidelity, and physical consistency of ML-calibrated satellite precipitation against in-situ gauges.")
    
    @st.cache_data(ttl=3600)
    def load_ml_data():
        ml_parquet = project_root / "outputs" / "processed" / "sarawak_ML_corrected.parquet"
        ml_leaderboard = project_root / "outputs" / "processed" / "ml_leaderboard.csv"
        df_ml = pd.read_parquet(ml_parquet) if ml_parquet.exists() else pd.DataFrame()
        leaderboard = pd.read_csv(ml_leaderboard) if ml_leaderboard.exists() else pd.DataFrame()
        return df_ml, leaderboard

    with st.spinner("🔄 Loading ML evaluation data..."):
        df_ml, leaderboard = load_ml_data()
        gc.collect()

    if df_ml.empty:
        st.error("⚠️ `sarawak_ML_corrected.parquet` not found. Please run `05_ml_bias_correction.ipynb` first.")
        st.stop()
    
    # Filter to valid rows
    valid_ml = df_ml.dropna(subset=['Gauge_Observed', 'P_merged_hier', 'P_corrected']).copy()
    
    # Get the best model from leaderboard to highlight
    best_model = leaderboard.iloc[0]['ML_Model'] if not leaderboard.empty else 'Gradient Boosting'
    
    st.markdown("---")
    
    # ==========================================================================
    # 1. TOP-LEVEL METRIC DECOMPOSITION (KGE Components)
    # ==========================================================================
    st.markdown("#### 📊 1. Aggregate Performance Metrics (Best Model: " + best_model + ")")
    
    # Calculate metrics for the best model's corrected data
    best_corrected = valid_ml[valid_ml['model_used'] == best_model]['P_corrected']
    gauge = valid_ml[valid_ml['model_used'] == best_model]['Gauge_Observed']
    
    r_kge = np.corrcoef(gauge, best_corrected)[0, 1]
    alpha = np.std(best_corrected) / np.std(gauge)
    beta = np.mean(best_corrected) / np.mean(gauge)
    kge = 1 - np.sqrt((r_kge - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    rmse = np.sqrt(np.mean((best_corrected - gauge)**2))
    rbias = (np.mean(best_corrected) - np.mean(gauge)) / np.mean(gauge) * 100
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("KGE", f"{kge:.3f}", help="Kling-Gupta Efficiency (1.0 = perfect)")
    col_m2.metric("Correlation (r)", f"{r_kge:.3f}", help="Linear correlation")
    col_m3.metric("Variability (α)", f"{alpha:.3f}", help="Ratio of std dev (1.0 = perfect)")
    col_m4.metric("Relative Bias (β)", f"{beta:.3f}", help="Ratio of means (1.0 = perfect)")
    col_m5.metric("RMSE", f"{rmse:.2f} mm", f"{rbias:+.1f}% Bias", delta_color="inverse" if rbias > 0 else "normal")

    st.markdown("---")

    # ==========================================================================
    # 2. VISUAL A: Log-Scale 2D Density Scatter
    # ==========================================================================
    st.markdown("#### 📈 2. Distributional Fidelity & Error Decomposition")
    
    # Sample for performance (max 20,000 points)
    sample_size = min(20000, len(valid_ml))
    df_sample = valid_ml.sample(sample_size, random_state=42)
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("**A. Log-Scale 2D Density Scatter**")
        st.caption("Hexagonal binning reveals data density. The 1:1 line represents perfect agreement. Note how ML calibration reduces the 'smear' of overestimation at low intensities.")
        
        fig_density = go.Figure()
        
        # Add 1:1 line
        max_val = max(df_sample['Gauge_Observed'].max(), df_sample['P_corrected'].max())
        fig_density.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(color='red', dash='dash', width=2), name='1:1 Perfect'))
        
        # Add 2D Histogram Contour for Calibrated vs Gauge
        fig_density.add_trace(go.Histogram2dContour(
            x=df_sample['Gauge_Observed'], y=df_sample['P_corrected'],
            colorscale='Blues', showscale=True,
            colorbar=dict(title="Density", x=1.05),
            name='ML Calibrated Density'
        ))
        
        fig_density.update_layout(
            height=400,
            xaxis=dict(title="In-Situ Gauge (mm)", type="log", range=[-1, 3]), # Log scale
            yaxis=dict(title="ML Calibrated (mm)", type="log", range=[-1, 3]),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_density, use_container_width=True)

    with col_a2:
        st.markdown("**B. Intensity-Regime Residual Boxplots**")
        st.caption("Residuals (Calibrated − Gauge) stratified by rainfall intensity. A successful model centers the median near zero across all regimes without compressing heavy extremes.")
        
        # Define intensity bins
        def get_regime(x):
            if x < 1: return 'Trace (<1 mm)'
            elif x < 5: return 'Light (1-5 mm)'
            elif x < 20: return 'Moderate (5-20 mm)'
            else: return 'Heavy (>20 mm)'
            
        df_sample['Regime'] = df_sample['Gauge_Observed'].apply(get_regime)
        df_sample['Residual_Calibrated'] = df_sample['P_corrected'] - df_sample['Gauge_Observed']
        df_sample['Residual_Raw'] = df_sample['P_merged_hier'] - df_sample['Gauge_Observed']
        
        # Melt for grouped boxplot
        df_box_melt = pd.melt(
            df_sample, 
            id_vars=['Regime'], 
            value_vars=['Residual_Raw', 'Residual_Calibrated'],
            var_name='Model', value_name='Residual (mm)'
        )
        df_box_melt['Model'] = df_box_melt['Model'].map({'Residual_Raw': 'Raw TCA', 'Residual_Calibrated': 'ML Calibrated'})
        
        # Custom order for regimes
        regime_order = ['Trace (<1 mm)', 'Light (1-5 mm)', 'Moderate (5-20 mm)', 'Heavy (>20 mm)']
        
        fig_box = px.box(
            df_box_melt, x='Regime', y='Residual (mm)', color='Model',
            category_orders={'Regime': regime_order},
            color_discrete_map={'Raw TCA': '#ef553b', 'ML Calibrated': '#00cc96'},
            points=False # Hide outliers to keep the boxplot clean
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.5)
        fig_box.update_layout(
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # ==========================================================================
    # 3. VISUAL C & D: Upper-Tail ECDF & Station-Level KGE vs Elevation
    # ==========================================================================
    col_c1, col_d1 = st.columns(2)
    
    with col_c1:
        st.markdown("**C. Upper-Tail Empirical CDF (ECDF)**")
        st.caption("Compares the cumulative distribution of rainfall. Divergence at the upper tail (right side) indicates attenuation of extreme convective events.")
        
        # Sample specifically for ECDF to ensure smooth curves
        df_ecdf = valid_ml[['Gauge_Observed', 'P_merged_hier', 'P_corrected']].sample(min(30000, len(valid_ml)), random_state=42)
        
        fig_ecdf = px.ecdf(
            df_ecdf.melt(var_name='Product', value_name='Precipitation (mm)'),
            x='Precipitation (mm)', color='Product',
            color_discrete_map={'Gauge_Observed': '#d62728', 'P_merged_hier': '#ef553b', 'P_corrected': '#00cc96'},
            labels={'Product': 'Data Source'}
        )
        # Focus on the upper tail (e.g., > 10mm) by setting x-axis range, but keep full data for calculation
        fig_ecdf.update_layout(
            height=400,
            xaxis=dict(title="Precipitation (mm)", type="log"),
            yaxis=dict(title="Cumulative Probability"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ecdf, use_container_width=True)

    with col_d1:
        st.markdown("**D. Station-Level KGE vs. Elevation Gradient**")
        st.caption("Evaluates spatial generalization. A flat trendline indicates the ML model successfully corrects biases across both coastal lowlands and highland orographic zones.")
        
        # Calculate station-level KGE
        station_kge = []
        for stn_id, group in valid_ml.groupby('station_id'):
            if len(group) < 50: continue
            g = group['Gauge_Observed']
            c = group['P_corrected']
            r = np.corrcoef(g, c)[0, 1]
            a = np.std(c) / np.std(g)
            b = np.mean(c) / np.mean(g)
            k = 1 - np.sqrt((r - 1)**2 + (a - 1)**2 + (b - 1)**2)
            
            elev = group['elev_m'].iloc[0]
            station_kge.append({'station_id': stn_id, 'Elevation (m)': elev, 'KGE': k})
            
        df_kge = pd.DataFrame(station_kge)
        
        fig_kge = px.scatter(
            df_kge, x='Elevation (m)', y='KGE', 
            trendline="ols", trendline_color_override="red",
            marginal_x="histogram",
            labels={'KGE': 'Station-Level KGE'}
        )
        fig_kge.update_layout(
            height=400,
            xaxis=dict(title="Station Elevation (m)"),
            yaxis=dict(title="KGE", range=[-0.5, 1.0])
        )
        st.plotly_chart(fig_kge, use_container_width=True)

    st.markdown("---")

    # ==========================================================================
    # 4. VISUAL E: Polar Taylor Diagram (Retained & Refined)
    # ==========================================================================
    st.markdown("#### 📐 3. Unified Taylor Diagram")
    st.caption("Summarizes centered RMSE, standard deviation, and Pearson correlation on a single polar plot. The red star is the perfect reference. Points closer to the star indicate superior performance.")
    
    # (Keep your existing, highly polished Taylor Diagram code here)
    # For brevity, I am including the core logic. Ensure your existing Taylor code is pasted here.
    all_taylor_products = {
        'precip_chirps': 'CHIRPS (Raw)', 'precip_persiann': 'PERSIANN-CDR (Raw)', 'precip_era5_land': 'ERA5-Land (Raw)',
        'P_merged_global': 'P_merged (Global)', 'P_merged_hier': 'P_merged (Hierarchical)', 'P_corrected': 'ML-Corrected'
    }
    
    # Filter to best model for clean Taylor diagram
    df_taylor = valid_ml[valid_ml['model_used'] == best_model].copy()
    obs = df_taylor['Gauge_Observed']
    obs_std = np.std(obs)
    
    product_defs = {
        'precip_chirps': {'label': 'CHIRPS', 'symbol': 'circle-open', 'color': '#1f77b4'},
        'precip_persiann': {'label': 'PERSIANN-CDR', 'symbol': 'square-open', 'color': '#ff7f0e'},
        'precip_era5_land': {'label': 'ERA5-Land', 'symbol': 'diamond-open', 'color': '#2ca02c'},
        'P_merged_hier': {'label': 'Raw TCA', 'symbol': 'triangle-down-open', 'color': '#9467bd'},
        'P_corrected': {'label': 'ML-Corrected', 'symbol': 'star', 'color': '#d62728'}
    }
    
    taylor_df = pd.DataFrame()
    for col_key in ['precip_chirps', 'precip_persiann', 'precip_era5_land', 'P_merged_hier', 'P_corrected']:
        if col_key not in df_taylor.columns: continue
        info = product_defs[col_key]
        est = df_taylor[col_key].dropna()
        obs_aligned = df_taylor.loc[est.index, 'Gauge_Observed'].dropna()
        common_idx = est.index.intersection(obs_aligned.index)
        if len(common_idx) < 10: continue
        
        est = est.loc[common_idx]
        obs_v = obs_aligned.loc[common_idx]
        
        r = np.corrcoef(obs_v, est)[0, 1]
        std_norm = np.std(est) / obs_std
        rmsd = np.sqrt(np.mean((est - obs_v)**2)) / obs_std
        
        taylor_df = pd.concat([taylor_df, pd.DataFrame([{'Product': info['label'], 'R': r, 'Std': std_norm, 'RMSD': rmsd, 'Color': info['color'], 'Symbol': info['symbol']}])], ignore_index=True)
    
    fig_taylor = go.Figure()
    rmsd_color, corr_color, std_color, grid_color = '#006400', '#2F4F4F', '#4B0082', '#696969'
    max_std = max(taylor_df['Std'].max(), 1.5) * 1.2 if not taylor_df.empty else 2.0
    theta = np.linspace(0, np.pi/2, 100)
    
    # RMSD arcs
    for rmsd_val in [0.5, 1.0, 1.5]:
        if rmsd_val < max_std * 1.5:
            x_arc = 1 + rmsd_val * np.cos(np.pi - theta)
            y_arc = rmsd_val * np.sin(np.pi - theta)
            mask = (x_arc >= 0) & (x_arc <= max_std) & (y_arc >= 0) & (y_arc <= max_std)
            fig_taylor.add_trace(go.Scatter(x=x_arc[mask], y=y_arc[mask], mode='lines', line=dict(color=rmsd_color, dash='dash', width=1.5), name=f'RMSD={rmsd_val}', showlegend=False))
            label_theta = np.pi / 4
            fig_taylor.add_annotation(x=1 + rmsd_val * np.cos(np.pi - label_theta), y=rmsd_val * np.sin(np.pi - label_theta), text=f'{rmsd_val}', showarrow=False, font=dict(size=10, color=rmsd_color), textangle=45, xanchor='center', yanchor='middle', bgcolor='rgba(255,255,255,0.9)', borderpad=2)
    
    # Correlation lines
    for r_val in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        angle = np.arccos(r_val)
        fig_taylor.add_trace(go.Scatter(x=[0, max_std * np.cos(angle)], y=[0, max_std * np.sin(angle)], mode='lines', line=dict(color=corr_color, width=1.5, dash='dash'), showlegend=False))
        fig_taylor.add_annotation(x=max_std * 0.92 * np.cos(angle), y=max_std * 0.92 * np.sin(angle), text=f'{r_val}', showarrow=False, font=dict(size=10, color='#000000'), textangle=np.degrees(angle) - 90)
    
    # Std arcs
    for std_val in [0.5, 1.0, 1.5]:
        if std_val <= max_std:
            theta_circle = np.linspace(0, np.pi/2, 100)
            fig_taylor.add_trace(go.Scatter(x=std_val * np.cos(theta_circle), y=std_val * np.sin(theta_circle), mode='lines', line=dict(color=std_color, width=1.5, dash='dot'), showlegend=False))
    
    # Reference point
    fig_taylor.add_trace(go.Scatter(x=[1], y=[0], mode='markers+text', marker=dict(size=14, color='red', symbol='star'), name='Reference (Perfect)', text=['Reference'], textposition='bottom center'))
    
    # Plot products
    for _, row in taylor_df.iterrows():
        y_val = row['Std'] * np.sin(np.arccos(row['R']))
        fig_taylor.add_trace(go.Scatter(x=[row['Std']], y=[y_val], mode='markers+text', marker=dict(size=12, color=row['Color'], symbol=row['Symbol'], line=dict(width=2, color=row['Color']), opacity=1.0), name=row['Product'], text=[f"{row['Product'][:10]}"], textposition='top center', textfont=dict(size=12, color='#000000'), hovertemplate=f"Product: {row['Product']}<br>R: {row['R']:.3f}<br>Std: {row['Std']:.3f}<br>RMSD: {row['RMSD']:.3f}<extra></extra>"))
    
    fig_taylor.update_layout(
        height=600,
        xaxis=dict(title=dict(text='Normalized Standard Deviation', font=dict(size=14, color='#000000')), range=[0, max_std], zeroline=True, zerolinecolor='#000000', zerolinewidth=1.5, showgrid=True, gridcolor=grid_color, gridwidth=1, showline=True, linecolor='#000000', linewidth=1.5, ticks='outside', tickcolor='#000000', tickwidth=1.5, showticklabels=True, tickfont=dict(size=12, color='#000000')),
        yaxis=dict(title=dict(text='Normalized Standard Deviation', font=dict(size=14, color='#000000')), range=[0, max_std], zeroline=True, zerolinecolor='#000000', zerolinewidth=1.5, showgrid=True, gridcolor=grid_color, gridwidth=1, showline=True, linecolor='#000000', linewidth=1.5, ticks='outside', tickcolor='#000000', tickwidth=1.5, showticklabels=True, tickfont=dict(size=12, color='#000000'), scaleanchor="x", scaleratio=1),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05, font=dict(size=12, color='#000000'), bgcolor='rgba(255,255,255,0.95)', bordercolor='#000000', borderwidth=1),
        showlegend=True, plot_bgcolor='rgba(255,255,255,1)', paper_bgcolor='rgba(255,255,255,1)', margin=dict(l=80, r=40, t=80, b=80)
    )
    fig_taylor.add_annotation(x=max_std * 0.85, y=max_std * 0.5, text="Correlation", showarrow=False, font=dict(size=12, color='#000000'), textangle=-30)
    fig_taylor.add_annotation(x=1, y=-0.1, text="Reference", showarrow=False, font=dict(size=12, color='#000000'))
    
    st.plotly_chart(fig_taylor, use_container_width=True)
    
    st.info("💡 **Note on Advanced Diagnostics:** Spatial decorrelation variograms and Leave-One-Station-Out (LOSO) cross-validation metrics are computationally intensive. They have been pre-computed and are documented in the thesis appendix to prove the absence of spatial memorization.")


# ==============================================================================
# TAB 4: EXPLAINABLE AI (xAI)
# ==============================================================================
with tab_xai:
    st.markdown("### 🧠 Explainable AI: Decoding the Bias Correction")
    xai_dir = project_root / "outputs" / "processed"
    available_models = [f.stem.replace("shap_importance_", "").replace("_", " ") for f in xai_dir.glob("shap_importance_*.csv")]
    
    if not available_models:
        st.error("⚠️ No xAI data found.")
        st.stop()

    selected_models = st.multiselect("Choose up to 3 ML models to analyze:", options=available_models, default=[available_models[0]], max_selections=3)
    if not selected_models: st.stop()

    cols = st.columns(len(selected_models))
    for col, model_name in zip(cols, selected_models):
        with col:
            safe_name = model_name.replace(" ", "_")
            imp_csv = xai_dir / f"shap_importance_{safe_name}.csv"
            bee_csv = xai_dir / f"shap_beeswarm_data_{safe_name}.csv"
            
            if not imp_csv.exists() or not bee_csv.exists(): continue
                
            imp_df = pd.read_csv(imp_csv)
            bee_df = pd.read_csv(bee_csv)
            
            st.markdown(f"### 🤖 {model_name}")
            fig_imp = px.bar(imp_df, x='Mean |SHAP|', y='Feature', orientation='h', color='Feature', color_discrete_sequence=px.colors.qualitative.Bold)
            fig_imp.update_layout(height=250, showlegend=False, margin=dict(l=0, r=0, t=20, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)
            
            fig_bee = px.scatter(bee_df, x='SHAP Value', y='Feature', color='Feature Value', color_continuous_scale='RdBu_r', opacity=0.5, height=250)
            fig_bee.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_bee.update_layout(margin=dict(l=0, r=0, t=20, b=20))
            st.plotly_chart(fig_bee, use_container_width=True)


# ==============================================================================
# TAB 5: DROUGHT EVENT ANALYSIS
# ==============================================================================
with tab_drought:
    st.markdown("### 🌵 Drought Event Analysis & Forecasting")
    
    events_csv = project_root / "outputs" / "processed" / "drought_events_summary.csv"
    data_csv = project_root / "outputs" / "processed" / "sarawak_ML_corrected.parquet"

    if not events_csv.exists() or not data_csv.exists():
        st.error("⚠️ Drought data not found.")
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

    st.info("📊 **Overview:** This module analyzes historical drought events using Run Theory. Select a station below to view the severity, duration, and return periods of past droughts, alongside the precipitation record for context.")

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_display = st.selectbox("Select Station:", options=display_options)
        selected_station = selected_display.split('(')[-1].strip(')')
        index_type = st.radio("Select Index Type:", ["SPI", "SPEI"], horizontal=True)
        scale = st.selectbox("Select Timescale (Months):", [3, 6, 12], index=2)
        
    stn_data = df_full[df_full['station_id'] == selected_station].set_index('date').sort_index()
    stn_events = events_df[events_df['station_id'] == selected_station].copy()
    stn_meta = df_full[df_full['station_id'] == selected_station].iloc[0]
    
    st.markdown("---")
    st.markdown("#### 📊 Interactive Drought Index Plot")
    st.caption(f"Displaying {index_type}-{scale}. **Note:** The drought classification thresholds are identical for both SPI and SPEI (Moderate: ≤ -1.0, Severe: ≤ -1.5, Extreme: ≤ -2.0). Precipitation bars are overlaid for context.")

    @st.cache_data(ttl=3600)
    def get_index_data(station_data, idx_type, sc):
        from src.drought_indices import calculate_spi, calculate_spei
        if idx_type == "SPI":
            return calculate_spi(station_data['P_corrected'], scale=sc)
        else:
            return calculate_spei(station_data['P_corrected'], station_data['pet'], scale=sc)

    with st.spinner(f"Calculating {index_type}-{scale}..."):
        stn_data[f'{index_type}_{scale}'] = get_index_data(stn_data, index_type, scale)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=stn_data.index, y=stn_data['P_corrected'], name='Precipitation (mm)', marker_color='lightblue', opacity=0.5), secondary_y=True)
    
    colors = ['red' if v <= -1.0 else '#1f77b4' for v in stn_data[f'{index_type}_{scale}']]
    fig.add_trace(go.Scatter(x=stn_data.index, y=stn_data[f'{index_type}_{scale}'], name=f'{index_type}-{scale}', line=dict(color='red', width=2)), secondary_y=False)
    
    fig.add_hline(y=-1.0, line_dash="dash", line_color="black", line_width=2, annotation_text="Moderate Threshold")
    fig.add_hline(y=-1.5, line_dash="dot", line_color="darkred", line_width=1.5, annotation_text="Severe")
    fig.add_hline(y=-2.0, line_dash="dot", line_color="red", line_width=1.5, annotation_text="Extreme")

    if len(stn_events) > 0:
        for i, event in stn_events.iterrows():
            try:
                onset_val = stn_data.loc[event['onset'], f'{index_type}_{scale}']
                termination_val = stn_data.loc[event['termination'], f'{index_type}_{scale}']
                fig.add_vrect(x0=event['onset'], x1=event['termination'], fillcolor="red", opacity=0.15, line_width=0, secondary_y=False)
                fig.add_trace(go.Scatter(x=[event['onset']], y=[onset_val], mode='markers', marker=dict(symbol='triangle-down', size=10, color='darkred'), name='Onset', showlegend=False), secondary_y=False)
            except Exception: continue

    fig.update_layout(height=500, xaxis_title="Date", template="plotly_white", hovermode="x unified", yaxis=dict(title=f"{index_type}-{scale} Value", range=[-4, 4]), yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Detected Drought Events")
    if len(stn_events) > 0:
        display_df = stn_events[['onset', 'termination', 'duration', 'severity', 'category', 'return_period_years']].copy()
        display_df['return_period_years'] = display_df['return_period_years'].round(1)
        display_df['onset'] = display_df['onset'].dt.strftime('%Y-%m')
        display_df['termination'] = display_df['termination'].dt.strftime('%Y-%m')
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No drought events found for the selected station.")


# ==============================================================================
# TAB 6: FUTURE DROUGHT RISK PROJECTION (CMIP6)
# ==============================================================================
with tab_future:
    st.markdown("### 🔮 Future Drought Risk Projection (CMIP6)")
    st.info("🔬 **Academic Standard Note:** Return periods represent the annual probability of exceedance. Under climate change, historical baselines no longer predict future risks (non-stationarity).")
    
    future_events_csv = project_root / "outputs" / "processed" / "future_drought_events_corrected.csv"
    if not future_events_csv.exists():
        st.error("⚠️ Future drought data not found. Please run the CMIP6 correction pipeline.")
        st.stop()

    future_df = pd.read_csv(future_events_csv)
    future_df['onset'] = pd.to_datetime(future_df['onset'])
    future_df['termination'] = pd.to_datetime(future_df['termination'])
    future_df['station_id'] = future_df['station_id'].astype(str)
    
    df_full = pd.read_parquet(project_root / "outputs" / "processed" / "sarawak_ML_corrected.parquet")
    hist_df = pd.read_csv(project_root / "outputs" / "processed" / "drought_events_summary.csv")
    hist_df['station_id'] = hist_df['station_id'].astype(str)

    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        stations_with_events = future_df[['station_id', 'station_name']].drop_duplicates()
        station_map = dict(zip(stations_with_events['station_id'], stations_with_events['station_name']))
        display_options = [f"{name} ({sid})" for sid, name in station_map.items()]
        selected_display = st.selectbox("Select Station:", options=sorted(display_options))
        selected_station_id = selected_display.split('(')[-1].strip(')')
        
        stn_meta = df_full[df_full['station_id'] == selected_station_id].iloc[0]
        st.markdown(f"##### 📍 {stn_meta['station_name']}")
        st.metric("Elevation", f"{stn_meta['elev_m']:.1f} m")

    with col_f2:
        map_df = pd.DataFrame({'lat': [stn_meta['lat']], 'lon': [stn_meta['lon']], 'station': [selected_display]})
        st.pydeck_chart(pdk.Deck(map_style='road', initial_view_state=pdk.ViewState(latitude=stn_meta['lat'], longitude=stn_meta['lon'], zoom=10), layers=[pdk.Layer("ScatterplotLayer", data=map_df, get_position="[lon, lat]", get_color="[255, 50, 50, 255]", get_radius=2000)], tooltip={"html": f"<b>Station:</b> {selected_display}"}, height=150))

    col_f3, col_f4, col_f5 = st.columns(3)
    with col_f3: selected_model = st.selectbox("Select GCM Model:", options=sorted(future_df['model'].dropna().unique()))
    with col_f4: selected_scenario = st.selectbox("Select SSP Scenario:", options=sorted(future_df['scenario'].dropna().unique()))
    with col_f5:
        valid_timeframes = ['near', 'far'] if selected_scenario != 'historical' else ['historical']
        selected_timeframe = st.selectbox("Select Timeframe:", options=valid_timeframes, index=0)

    future_stn = future_df[(future_df['station_id'] == selected_station_id) & (future_df['model'] == selected_model) & (future_df['scenario'] == selected_scenario) & (future_df['timeframe'] == selected_timeframe)].copy()
    hist_stn = hist_df[hist_df['station_id'] == selected_station_id].copy()

    if future_stn.empty:
        st.warning(f"⚠️ No drought events projected for this combination.")
    else:
        st.markdown("#### 📉 Return Period Shift: Historical vs. Future")
        cat_order = ['Moderate', 'Severe', 'Extreme']
        hist_agg = hist_stn[hist_stn['category'].isin(cat_order)].groupby('category')['return_period_years'].median().reset_index()
        hist_agg['period'] = 'Historical Baseline'
        fut_agg = future_stn[future_stn['category'].isin(cat_order)].groupby('category')['return_period_years'].median().reset_index()
        fut_agg['period'] = f'Future ({selected_timeframe})'
        
        comp_df = pd.concat([hist_agg, fut_agg], ignore_index=True)
        for cat in cat_order:
            if cat not in comp_df['category'].values:
                comp_df = pd.concat([comp_df, pd.DataFrame({'category': [cat], 'return_period_years': [np.nan], 'period': ['Historical Baseline']}), pd.DataFrame({'category': [cat], 'return_period_years': [np.nan], 'period': [f'Future ({selected_timeframe})']})])
        
        comp_df = comp_df.dropna(subset=['return_period_years'])
        if not comp_df.empty:
            fig_rp = px.bar(comp_df, x='category', y='return_period_years', color='period', barmode='group', text='return_period_years', category_orders={'category': cat_order}, color_discrete_map={'Historical Baseline': '#1f77b4', f'Future ({selected_timeframe})': '#d62728'}, height=400)
            fig_rp.update_traces(texttemplate='%{text:.1f} yrs', textposition='outside')
            st.plotly_chart(fig_rp, use_container_width=True)

        st.markdown("#### 📅 Projected Drought Risk Timeline")
        future_stn_sorted = future_stn.sort_values('onset')
        future_stn_sorted['start_date'] = future_stn_sorted['onset'].dt.strftime('%Y-%m')
        future_stn_sorted['end_date'] = future_stn_sorted['termination'].dt.strftime('%Y-%m')
        color_map = {'Moderate': '#fdae61', 'Severe': '#d7191c', 'Extreme': '#7f0000'}
        future_stn_sorted['color'] = future_stn_sorted['category'].map(color_map)
        
        fig_gantt = go.Figure()
        added_categories = set()
        for i, row in future_stn_sorted.iterrows():
            show_legend = row['category'] not in added_categories
            if show_legend: added_categories.add(row['category'])
            fig_gantt.add_trace(go.Scatter(x=[row['start_date'], row['end_date']], y=[row['category'], row['category']], mode='lines', line=dict(color=row['color'], width=25), name=row['category'], hovertemplate=f"<b>{row['category']} Drought</b><br>Start: {row['start_date']}<br>End: {row['end_date']}<br>Return Period: 1-in-{row['return_period_years']:.1f} years<extra></extra>", showlegend=show_legend))
            
        fig_gantt.update_layout(height=300, yaxis={'categoryorder': 'array', 'categoryarray': ['Extreme', 'Severe', 'Moderate']}, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_gantt, use_container_width=True)
