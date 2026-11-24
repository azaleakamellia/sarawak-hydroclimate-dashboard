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
    'precip_chirps': 'Raw CHIRPS', 
    'precip_era5_land': 'Raw ERA5-Land',
    'precip_persiann': 'Raw PERSIANN-CDR'
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

if 'Gauge_Observed' in df.columns:
    PRECIP_SOURCES['Gauge_Observed'] = 'In-Situ Gauge (Truth)'

# ==============================================================================
# TAB SETUP
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

        # Preferred order: Truth at the base, SPEs stacked above it
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
            index=0  # set to 1 to default to overlaid KDE
        )

        # Shared, honest x-range across all selected sources (trim the extreme tail)
        all_vals = df_melted[df_melted['Data Source'].isin(sel)]['Precipitation (mm)'].dropna()
        x_max = float(all_vals.quantile(0.99))
        x = np.linspace(0, x_max, 200)

        # Pre-compute normalized KDE + stats per source (subsampled for speed)
        kde_store = {}
        for source in sel:
            v = df_melted[df_melted['Data Source'] == source]['Precipitation (mm)'].dropna()
            if len(v) < 2:
                continue
            if len(v) > 10000:
                v = v.sample(10000, random_state=42)
            dens = stats.gaussian_kde(v)(x)
            dens = dens / dens.max()  # normalize each ridge to unit peak
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
        # Helper function to convert hex to rgba for fills
        def get_rgba(h, a):
            n = int(h.lstrip('#'), 16)
            return f"rgba({(n>>16)&255},{(n>>8)&255},{n&255},{a})"

        # Define colors
        color_map_hex = {
            'In-Situ Gauge (Truth)': '#d62728',
            'Raw CHIRPS': '#1f77b4',
            'Raw ERA5-Land': '#ff7f0e',
            'Raw PERSIANN-CDR': '#2ca02c'
        }

        if viz_mode.startswith("Ridge"):
            overlap = 0.55  # Controls the vertical spacing between ridges
            fig = go.Figure()
            tickvals, ticktext = [], []
            
            for i, source in enumerate(kde_store):
                d = kde_store[source]
                off = i * overlap
                tickvals.append(off)
                ticktext.append(source)
                
                # ✅ FIX: Look up the specific hex string for this source
                hexc = color_map_hex.get(source, '#636efa') 

                # 1. Baseline separator (thin light line for dark mode)
                fig.add_trace(go.Scatter(
                    x=x, y=[off]*len(x), mode='lines',
                    line=dict(width=1, color='rgba(255,255,255,0.2)'),
                    hoverinfo='skip', showlegend=False
                ))
                
                # 2. Filled Ridge
                fig.add_trace(go.Scatter(
                    x=x, y=off + d['y'], mode='lines', name=source,
                    line=dict(color=hexc, width=2),
                    fill='tonexty', fillcolor=get_rgba(hexc, 0.7),
                    hovertemplate=f"<b>{source}</b><br>Precip: %{{x:.1f}} mm<br><extra></extra>"
                ))
                
                # 3. Mean Line (White for contrast against dark fills)
                fig.add_trace(go.Scatter(
                    x=[d['mean'], d['mean']], y=[off, off + 1.0], mode='lines',
                    line=dict(color='white', dash='dash', width=2),
                    hoverinfo='skip', showlegend=False
                ))
                
                # 4. Annotation (Standardized with Overlaid KDE style)
                fig.add_annotation(
                    x=d['mean'], y=off + 1.05, 
                    text=f"μ={d['mean']:.0f} σ={d['std']:.0f}",
                    showarrow=False, 
                    # ✅ FIX: Use the specific hex string, not the dictionary
                    font=dict(size=10, color=hexc, family="Arial Black"),
                    align="center",
                    bgcolor='rgba(255,255,255,0.9)', 
                    bordercolor=hexc, 
                    borderwidth=1, 
                    borderpad=3,
                    yanchor='bottom'
                )

            # Layout Fixes for Dark Mode & Readability
            fig.update_layout(
                height=max(400, len(kde_store) * 120),
                plot_bgcolor='rgba(0,0,0,0)',   # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
                font=dict(color='white'),       # White text for axis labels
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
                margin=dict(l=150, r=20, t=20, b=50) # Extra left margin for station names
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
                    fill='tozeroy', fillcolor=rgba(hexc, 0.15),
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

    # A. Clean station names (fixes the 'nan (ID)' tags)
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

        # B. Data source selection
        plot_cols = ['Gauge_Observed', 'precip_chirps', 'precip_era5_land', 'precip_persiann']
        source_options = [c for c in plot_cols if c in df_station_filtered.columns]
        selected_box_sources = st.multiselect(
            "📊 Select Data Sources for Distribution Comparison:",
            options=source_options,
            default=source_options
        )

        if selected_box_sources:
            # Optional UX toggle: hide outlier dots for a cleaner read
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

            # ✅ FIX #3: height scales with stations × sources (capped so the page stays scrollable)
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

            # ✅ Styling: mean line + thinner boxes + smaller outlier dots = readable clusters
            fig_box.update_traces(
                boxmean=True,                 # dashed line = mean (great for bias spotting)
                marker=dict(size=4, opacity=0.6),
                line=dict(width=1.5)
            )
            fig_box.update_layout(
                yaxis_title="Station",
                xaxis_title="Monthly Precipitation (mm)",
                hovermode="closest",          # 'x unified' is messy on boxplots
                yaxis=dict(tickfont=dict(size=11)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_box, use_container_width=True)

            if len(selected_station_ids) * len(selected_box_sources) > 60:
                st.caption("💡 Tip: for the cleanest read, keep stations × sources under ~60 boxes.")

        st.markdown("---")

        # C. Synchronized Tabular Data (Drill-down)
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

        # ✅ FIX #1: gate on the columns that actually EXIST after aggregation
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

        # ✅ FIX #1 (belt-and-braces): only render columns that truly exist
        display_cols = [c for c in display_cols if c in filtered_station_stats.columns]

        st.dataframe(
            filtered_station_stats[display_cols],
            hide_index=True,
            use_container_width=True,
            column_config=column_config
        )
        
# ==============================================================================
# TAB 2: HYDROCLIMATE EXPLORER
# ==============================================================================
with tab_explore:
    st.sidebar.markdown("### 🎛️ Explorer Controls")
    
    min_date, max_date = df['date'].min().date(), df['date'].max().date()
    date_range = st.sidebar.slider("📅 Timeline", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")
    plot_grouping = st.sidebar.radio("📊 Chart Display Mode", options=["By Station", "By Dataset"], index=0)
    temporal_agg = st.sidebar.radio("⏱️ Temporal Resolution", options=["Monthly", "Annual Totals"], index=0)
    
    selected_precip = st.sidebar.multiselect("🌧️ Precipitation Sources", options=list(PRECIP_SOURCES.keys()), format_func=lambda x: PRECIP_SOURCES[x], default=['precip_chirps'])
    
    station_dict = df.drop_duplicates('station_id').set_index('station_id').get('station_name', pd.Series(index=df['station_id'].unique(), data="Unknown")).to_dict()
    all_stations = sorted(df['station_id'].unique())
    selected_stations = st.sidebar.multiselect("📍 Select Stations (Max 5)", options=all_stations, format_func=lambda x: f"{x} - {station_dict.get(x, 'Unknown')}", default=[all_stations[0]] if all_stations else [], max_selections=5)
    
    show_et = st.sidebar.selectbox("☀️ Evapotranspiration Data", options=["None", "ERA5-Land PET (mm)"], index=1)

    st.markdown("---")

    df_filtered = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])].copy()

    if temporal_agg == "Annual Totals":
        df_filtered['year'] = df_filtered['date'].dt.year
        agg_dict = {src: 'sum' for src in PRECIP_SOURCES.keys() if src in df_filtered.columns}
        if 'pet' in df_filtered.columns: agg_dict['pet'] = 'sum'
        meta_cols = ['lon', 'lat', 'elev_m', 'Land_Use']
        for col in meta_cols:
            if col in df_filtered.columns: agg_dict[col] = 'first'
        df_filtered = df_filtered.groupby(['station_id', 'year']).agg(agg_dict).reset_index()
        df_filtered['date'] = pd.to_datetime(df_filtered['year'].astype(str) + '-07-01')

    def calculate_metrics(data, sources, truth_col='Gauge_Observed'):
        results = []
        if truth_col not in data.columns: return pd.DataFrame()
        for src in sources:
            if src == truth_col or src not in data.columns: continue
            valid_mask = data[src].notna() & data[truth_col].notna()
            sim, obs = data.loc[valid_mask, src], data.loc[valid_mask, truth_col]
            if len(sim) > 1:
                results.append({"Product": PRECIP_SOURCES[src], "RMSE (mm)": round(np.sqrt(((sim - obs) ** 2).mean()), 2), "MAE (mm)": round(np.abs(sim - obs).mean(), 2), "CC (r)": round(np.corrcoef(sim, obs)[0, 1], 3)})
        return pd.DataFrame(results)

    col_plots, col_map = st.columns([2.5, 1])

    with col_map:
        st.markdown("### 🗺️ Spatial Collocation")
        map_agg = {src: 'sum' for src in PRECIP_SOURCES.keys() if src in df_filtered.columns}
        if 'pet' in df_filtered.columns: map_agg['pet'] = 'sum'
        summary_df = df_filtered.groupby(['station_id', 'lon', 'lat', 'elev_m', 'Land_Use'], dropna=False).agg(map_agg).reset_index()
        
        for col in map_agg.keys(): summary_df[f"{col}_str"] = summary_df[col].apply(lambda x: f"{x:,.1f} mm" if pd.notnull(x) else "N/A")
        
        center_lat = summary_df['lat'].mean() if not summary_df.empty else 0
        center_lon = summary_df['lon'].mean() if not summary_df.empty else 0
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6, pitch=0)
        
        unselected_df = summary_df[~summary_df['station_id'].isin(selected_stations)]
        focus_df = summary_df[summary_df['station_id'].isin(selected_stations)]
        
        layers = []
        if not unselected_df.empty: layers.append(pdk.Layer("ScatterplotLayer", data=unselected_df, get_position="[lon, lat]", get_color="[120, 120, 120, 140]", get_radius=4000, pickable=True))
        if not focus_df.empty: layers.append(pdk.Layer("ScatterplotLayer", data=focus_df, get_position="[lon, lat]", get_color="[255, 50, 50, 255]", get_radius=8000, pickable=True))

        tooltip_html = "<b>📍 Station: {station_id}</b><br/><b>Land Use:</b> {Land_Use}<br/><b>Elev:</b> {elev_m} m<hr style='margin: 8px 0px;'/>"
        for src in selected_precip:
            if src in summary_df.columns: tooltip_html += f"<b>{PRECIP_SOURCES[src]}:</b> {{{src}_str}}<br/>"
                
        st.pydeck_chart(pdk.Deck(map_style='road', initial_view_state=view_state, layers=layers, tooltip={"html": tooltip_html}, height=300))
        
        if 'Gauge_Observed' in df.columns and selected_stations:
            metrics_df = calculate_metrics(df_filtered[df_filtered['station_id'].isin(selected_stations)], selected_precip)
            if not metrics_df.empty: 
                st.markdown("### 📈 Performance Metrics")
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with col_plots:
        st.markdown("### 📊 Hydroclimatic Time Series")
        if not selected_stations or not selected_precip:
            st.warning("👈 Please select at least one station and one precipitation source from the sidebar.")
        else:
            color_map = {'precip_chirps': '#1f77b4', 'precip_era5_land': '#ff7f0e', 'precip_persiann': '#2ca02c', 'Gauge_Observed': '#d62728'}
            if plot_grouping == "By Station":
                for station in selected_stations:
                    station_data = df_filtered[df_filtered['station_id'] == station].sort_values('date')
                    fig = make_subplots(rows=2 if show_et != "None" else 1, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3] if show_et != "None" else [1.0])
                    
                    for src in selected_precip:
                        fig.add_trace(go.Scatter(x=station_data['date'], y=station_data[src], name=PRECIP_SOURCES[src], mode='lines', line=dict(color=color_map.get(src, None), width=2)), row=1, col=1)
                    
                    if show_et != "None" and 'pet' in station_data.columns:
                        fig.add_trace(go.Scatter(x=station_data['date'], y=station_data['pet'], name="PET", fill='tozeroy', line=dict(color='rgba(255, 165, 0, 0.8)', width=1)), row=2, col=1)
                    
                    if temporal_agg == "Annual Totals":
                        fig.update_xaxes(tickformat="%Y")
                        
                    fig.update_layout(title=dict(text=f"<b>Station: {station}</b>"), height=400, margin={"r":20,"t":40,"l":20,"b":20}, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)), hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 3: TCA WEIGHT ANALYSIS (OPTIMIZED & PDF-ALIGNED)
# ==============================================================================
with tab_tca:
    st.markdown("### 📐 Triple Collocation Analysis (TCA) Weight Allocation")
    st.write("This module evaluates the inverse-variance weights assigned to each satellite product. Higher weights indicate lower random error variance (higher reliability) in that specific regime.")
    
    # Reference the PDF's core concept
    st.info("📖 **Scientific Context:** Following TCA methodology (McColl et al., 2014), weights are derived from mutual log-space covariance. As noted in the presentation guide, these weights are *regime-dependent*—shifting significantly across monsoon seasons and elevation quartiles.")

    # 1. Stratification Selector
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
            if len(data) < 5: return [np.nan, np.nan, np.nan]
            try:
                X = np.log(data[['precip_chirps', 'precip_era5_land', 'precip_persiann']].values)
                y = np.log(data[target_col].values)
                w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                return w
            except Exception: return [np.nan, np.nan, np.nan]
        
        w_g = get_w(valid, 'P_merged_global')
        df_g = pd.DataFrame([{"Stratum": "Global", "CHIRPS": w_g[0], "ERA5-Land": w_g[1], "PERSIANN-CDR": w_g[2]}])
        
        records_s = [{"Stratum": s, "CHIRPS": w[0], "ERA5-Land": w[1], "PERSIANN-CDR": w[2]} 
                     for s in sorted(valid['season'].dropna().unique()) 
                     for w in [get_w(valid[valid['season'] == s], 'P_merged_season')]]
        df_s = pd.DataFrame(records_s)
        
        records_e = [{"Stratum": e, "CHIRPS": w[0], "ERA5-Land": w[1], "PERSIANN-CDR": w[2]} 
                     for e in sorted(valid['elev_bin'].dropna().unique(), reverse=True) 
                     for w in [get_w(valid[valid['elev_bin'] == e], 'P_merged_elev')]]
        df_e = pd.DataFrame(records_e)
        
        records_h = [{"Stratum": h, "CHIRPS": w[0], "ERA5-Land": w[1], "PERSIANN-CDR": w[2]} 
                     for h in sorted(valid['stratum_hier'].dropna().unique()) 
                     for w in [get_w(valid[valid['stratum_hier'] == h], 'P_merged_hier')]]
        df_h = pd.DataFrame(records_h)
        
        return df_g, df_s, df_e, df_h

    file_path = str(project_root / "outputs" / "processed" / "sarawak_TCA_fused.parquet")
    with st.spinner("Calculating TCA weights..."):
        df_g, df_s, df_e, df_h = compute_all_tca_weights(file_path)
        gc.collect()
    
    weights_map = {"Global": df_g, "Seasonal": df_s, "Elevation": df_e, "Hierarchical": df_h}
    weights_df = weights_map.get(strat_level, df_g)
    
    if weights_df.empty or weights_df['CHIRPS'].isna().all():
        st.warning("⚠️ No valid TCA weights could be calculated for this stratum.")
    else:
        # --- VISUAL 1: THE TERNARY SIMPLEX (PDF Figure 7) ---
        st.markdown("#### 🔺 1. Ternary Merging Weight Simplex")
        st.caption("Maps optimal TCA inverse-variance weights ($w_i \propto 1/\sigma_{\epsilon,i}^2$). Points closer to a vertex indicate that product dominates the fusion in that regime.")
        
        fig_tern = px.scatter_ternary(
            weights_df, 
            a="PERSIANN-CDR", b="CHIRPS", c="ERA5-Land", 
            hover_name="Stratum", 
            color="Stratum" if strat_level != "Global" else None, 
            color_discrete_sequence=px.colors.qualitative.Bold if strat_level == "Global" else None,
            size_max=15,
            opacity=0.8
        )
        fig_tern.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        fig_tern.update_layout(
            height=500, margin={"r":20,"t":40,"l":20,"b":20},
            ternary=dict(
                sum=1, 
                aaxis=dict(title="PERSIANN-CDR Weight", tickformat=".2f", tickfont=dict(size=10)),
                baxis=dict(title="CHIRPS Weight", tickformat=".2f", tickfont=dict(size=10)),
                caxis=dict(title="ERA5-Land Weight", tickformat=".2f", tickfont=dict(size=10))
            )
        )
        st.plotly_chart(fig_tern, use_container_width=True)

        st.markdown("---")

        # --- VISUAL 2: REGIME-DEPENDENT HEATMAPS (PDF Figure 1/2 Adaptation) ---
        st.markdown("#### 🌡️ 2. Regime-Dependent Reliability Heatmaps")
        st.caption("Visualizes how satellite reliability shifts across Monsoon Seasons (Left) and Elevation Quartiles (Right). Yellow = High Weight (Low Error).")
        
        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            st.markdown("**Seasonal Weights**")
            # Ensure correct order for seasons
            season_order = ['NEM', 'Inter', 'SWM']
            df_s_sorted = df_s[df_s['Stratum'].isin(season_order)].copy()
            df_s_sorted['Stratum'] = pd.Categorical(df_s_sorted['Stratum'], categories=season_order, ordered=True)
            df_s_sorted = df_s_sorted.sort_values('Stratum')
            
            if not df_s_sorted.empty:
                heat_data_s = df_s_sorted[['CHIRPS', 'ERA5-Land', 'PERSIANN-CDR']].values
                fig_heat_s = go.Figure(data=go.Heatmap(
                    z=heat_data_s,
                    x=['CHIRPS', 'ERA5-Land', 'PERSIANN-CDR'],
                    y=df_s_sorted['Stratum'],
                    colorscale='YlOrRd', # Yellow to Red is better for "Heat/Reliability"
                    text=np.round(heat_data_s, 3),
                    texttemplate="%{text}",
                    hoverongaps=False,
                    colorbar=dict(title="Weight")
                ))
                fig_heat_s.update_layout(height=300, margin={"r":20,"t":20,"l":20,"b":20}, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_heat_s, use_container_width=True)

        with col_h2:
            st.markdown("**Elevation Weights (Q1=Low, Q4=High)**")
            if not df_e.empty:
                heat_data_e = df_e[['CHIRPS', 'ERA5-Land', 'PERSIANN-CDR']].values
                fig_heat_e = go.Figure(data=go.Heatmap(
                    z=heat_data_e,
                    x=['CHIRPS', 'ERA5-Land', 'PERSIANN-CDR'],
                    y=df_e['Stratum'],
                    colorscale='YlOrRd',
                    text=np.round(heat_data_e, 3),
                    texttemplate="%{text}",
                    hoverongaps=False,
                    colorbar=dict(title="Weight")
                ))
                fig_heat_e.update_layout(height=300, margin={"r":20,"t":20,"l":20,"b":20}, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_heat_e, use_container_width=True)

        st.markdown("---")

        # --- VISUAL 3: RELIABILITY SHIFT BAR CHART (PDF Figure 2 Concept) ---
        st.markdown("#### 📊 3. Reliability Shift Across Elevation Quartiles")
        st.caption("Grouped bar chart showing the evolution of satellite weights from Lowland (Q1) to Highland (Q4) regimes. Highlights the impact of orographic lifting on satellite performance.")
        
        if not df_e.empty:
            df_e_melt = df_e.melt(id_vars='Stratum', var_name='Product', value_name='Weight')
            
            fig_bar = px.bar(
                df_e_melt, 
                x='Stratum', 
                y='Weight', 
                color='Product',
                barmode='group',
                text='Weight',
                color_discrete_map={
                    'CHIRPS': '#1f77b4', 
                    'ERA5-Land': '#ff7f0e', 
                    'PERSIANN-CDR': '#2ca02c'
                },
                height=400
            )
            fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_bar.update_layout(
                xaxis_title="Elevation Quartile (Q1=Lowest, Q4=Highest)",
                yaxis_title="TCA Weight (Reliability)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- RAW DATA EXPANDER ---
        with st.expander("🔍 View Raw TCA Weight Data"):
            st.dataframe(weights_df.style.format({"CHIRPS": "{:.3f}", "ERA5-Land": "{:.3f}", "PERSIANN-CDR": "{:.3f}"}), use_container_width=True)


# ==============================================================================
# TAB 4: ML BIAS CORRECTION EVALUATION
# ==============================================================================
with tab_ml:
    st.markdown("### 🤖 Machine Learning Bias Correction Performance")
    
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
        st.error("⚠️ `sarawak_ML_corrected.parquet` not found.")
    else:
        valid_ml = df_ml.dropna(subset=['Gauge_Observed', 'P_merged_hier', 'P_corrected']).copy()
        available_models = sorted(leaderboard['ML_Model'].unique()) if not leaderboard.empty else []
        selected_models = st.multiselect("🔍 Select ML Models to Compare", options=available_models, default=[available_models[0]] if available_models else [])
        
        if not selected_models: st.stop()
        
        st.markdown("#### 🏆 Model Performance Leaderboard")
        display_lb = leaderboard[leaderboard['ML_Model'].isin(selected_models)].copy()
        display_lb = display_lb.sort_values('Composite_Score', ascending=False)
        st.dataframe(display_lb.style.highlight_max(axis=0, subset=['Composite_Score'], color='#d4edda'), hide_index=True, use_container_width=True)

        st.markdown("#### 📊 Unified Performance Comparison")
        marker_shapes = {'Linear Regression': 'square', 'Random Forest': 'square', 'Gradient Boosting': 'square', 'Neural Network (MLP)': 'square', 'SVR': 'square'}
        
        fig_scatter = go.Figure()
        MAX_POINTS = 15000 
        
        for model in selected_models:
            model_data = valid_ml[valid_ml['model_used'] == model] if 'model_used' in valid_ml.columns else valid_ml
            for variant, col_name in [('Raw (Hierarchical)', 'P_merged_hier'), (f'Corrected ({model})', 'P_corrected')]:
                subset = model_data[['Gauge_Observed', col_name]].dropna()
                if len(subset) > MAX_POINTS: subset = subset.sample(MAX_POINTS, random_state=42)
                
                is_raw = 'Raw' in variant
                color = '#ef553b' if is_raw else '#00cc96'
                
                fig_scatter.add_trace(go.Scattergl(
                    x=subset['Gauge_Observed'], y=subset[col_name], mode='markers', name=f"{model} - {variant}",
                    marker=dict(symbol='square', size=6, color=color, opacity=0.4, line=dict(color='white', width=0.5))
                ))
        
        max_val = max(valid_ml['Gauge_Observed'].max(), valid_ml['P_merged_hier'].max(), valid_ml['P_corrected'].max())
        fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='1:1 Perfect Agreement', line=dict(color='red', dash='dash', width=3)))
        
        fig_scatter.update_layout(height=500, xaxis_title="In-Situ Gauge (mm)", yaxis_title="Estimated Precipitation (mm)", hovermode='closest', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("#### 📐 Unified Taylor Diagram")
        all_taylor_products = {
            'precip_chirps': 'CHIRPS (Raw)', 'precip_persiann': 'PERSIANN-CDR (Raw)', 'precip_era5_land': 'ERA5-Land (Raw)',
            'P_merged_global': 'P_merged (Global)', 'P_merged_hier': 'P_merged (Hierarchical)', 'P_corrected': 'ML-Corrected P_merged'
        }
        
        selected_taylor_products = st.multiselect("🔍 Select Products to Display on Taylor Diagram", options=list(all_taylor_products.keys()), default=['precip_chirps', 'precip_persiann', 'precip_era5_land', 'P_merged_hier', 'P_corrected'], format_func=lambda x: all_taylor_products[x])
        
        obs = valid_ml['Gauge_Observed']
        obs_std = np.std(obs)
        
        product_defs = {
            'precip_chirps': {'label': 'CHIRPS', 'symbol': 'circle-open', 'color': '#1f77b4'},
            'precip_persiann': {'label': 'PERSIANN-CDR', 'symbol': 'square-open', 'color': '#ff7f0e'},
            'precip_era5_land': {'label': 'ERA5-Land', 'symbol': 'diamond-open', 'color': '#2ca02c'},
            'P_merged_global': {'label': 'P_merged (Global)', 'symbol': 'triangle-up-open', 'color': '#8c564b'},
            'P_merged_hier': {'label': 'P_merged (Hier)', 'symbol': 'triangle-down-open', 'color': '#9467bd'},
            'P_corrected': {'label': 'ML-Corrected', 'symbol': 'star', 'color': '#d62728'}
        }
        
        taylor_df = pd.DataFrame()
        for col_key in selected_taylor_products:
            if col_key not in valid_ml.columns: continue
            info = product_defs[col_key]
            est = valid_ml[col_key].dropna()
            obs_aligned = valid_ml.loc[est.index, 'Gauge_Observed'].dropna()
            common_idx = est.index.intersection(obs_aligned.index)
            est = est.loc[common_idx]
            obs_v = obs_aligned.loc[common_idx]
            
            if len(est) < 10 or np.std(est) == 0: continue
                
            r = np.corrcoef(obs_v, est)[0, 1]
            std_norm = np.std(est) / obs_std
            rmsd = np.sqrt(np.mean((est - obs_v)**2)) / obs_std
            
            taylor_df = pd.concat([taylor_df, pd.DataFrame([{'Product': info['label'], 'R': r, 'Std': std_norm, 'RMSD': rmsd, 'Color': info['color'], 'Symbol': info['symbol']}])], ignore_index=True)
        
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
                fig_taylor.add_trace(go.Scatter(x=x_arc[mask], y=y_arc[mask], mode='lines', line=dict(color=rmsd_color, dash='dash', width=1.5), name=f'RMSD={rmsd_val}', showlegend=False))
                label_theta = np.pi / 4
                label_x = 1 + rmsd_val * np.cos(np.pi - label_theta)
                label_y = rmsd_val * np.sin(np.pi - label_theta)
                fig_taylor.add_annotation(x=label_x, y=label_y, text=f'{rmsd_val}', showarrow=False, font=dict(size=10, color=rmsd_color), textangle=45, xanchor='center', yanchor='middle', bgcolor='rgba(255,255,255,0.9)', borderpad=2)
        
        for r_val in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
            angle = np.arccos(r_val)
            fig_taylor.add_trace(go.Scatter(x=[0, max_std * np.cos(angle)], y=[0, max_std * np.sin(angle)], mode='lines', line=dict(color=corr_color, width=1.5, dash='dash'), showlegend=False))
            fig_taylor.add_annotation(x=max_std * 0.92 * np.cos(angle), y=max_std * 0.92 * np.sin(angle), text=f'{r_val}', showarrow=False, font=dict(size=10, color='#000000'), textangle=np.degrees(angle) - 90)
        
        for std_val in [0.5, 1.0, 1.5]:
            if std_val <= max_std:
                theta_circle = np.linspace(0, np.pi/2, 100)
                fig_taylor.add_trace(go.Scatter(x=std_val * np.cos(theta_circle), y=std_val * np.sin(theta_circle), mode='lines', line=dict(color=std_color, width=1.5, dash='dot'), showlegend=False))
        
        fig_taylor.add_trace(go.Scatter(x=[1], y=[0], mode='markers+text', marker=dict(size=14, color='red', symbol='star'), name='Reference (Perfect)', text=['Reference'], textposition='bottom center'))
        
        for _, row in taylor_df.iterrows():
            y_val = row['Std'] * np.sin(np.arccos(row['R']))
            fig_taylor.add_trace(go.Scatter(x=[row['Std']], y=[y_val], mode='markers+text', marker=dict(size=12, color=row['Color'], symbol=row['Symbol'], line=dict(width=2, color=row['Color']), opacity=1.0), name=row['Product'], text=[f"{row['Product'][:10]}"], textposition='top center', textfont=dict(size=12, color='#000000'), hovertemplate=f"Product: {row['Product']}<br>R: {row['R']:.3f}<br>Std: {row['Std']:.3f}<br>RMSD: {row['RMSD']:.3f}<extra></extra>"))
        
        fig_taylor.update_layout(height=700, xaxis=dict(title=dict(text='Normalized Standard Deviation', font=dict(size=14, color='#000000')), range=[0, max_std], zeroline=True, zerolinecolor='#000000', zerolinewidth=1.5, showgrid=True, gridcolor=grid_color, gridwidth=1, showline=True, linecolor='#000000', linewidth=1.5, ticks='outside', tickcolor='#000000', tickwidth=1.5, showticklabels=True, tickfont=dict(size=12, color='#000000')), yaxis=dict(title=dict(text='Normalized Standard Deviation', font=dict(size=14, color='#000000')), range=[0, max_std], zeroline=True, zerolinecolor='#000000', zerolinewidth=1.5, showgrid=True, gridcolor=grid_color, gridwidth=1, showline=True, linecolor='#000000', linewidth=1.5, ticks='outside', tickcolor='#000000', tickwidth=1.5, showticklabels=True, tickfont=dict(size=12, color='#000000'), scaleanchor="x", scaleratio=1), legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05, font=dict(size=12, color='#000000'), bgcolor='rgba(255,255,255,0.95)', bordercolor='#000000', borderwidth=1), showlegend=True, plot_bgcolor='rgba(255,255,255,1)', paper_bgcolor='rgba(255,255,255,1)', margin=dict(l=80, r=40, t=80, b=80))
        fig_taylor.add_annotation(x=max_std * 0.85, y=max_std * 0.5, text="Correlation", showarrow=False, font=dict(size=12, color='#000000'), textangle=-30)
        fig_taylor.add_annotation(x=1, y=-0.1, text="Reference", showarrow=False, font=dict(size=12, color='#000000'))
        st.plotly_chart(fig_taylor, use_container_width=True)
        
        try:
            st.download_button(label="📥 Download Taylor Diagram PNG", data=fig_taylor.to_image(format="png", scale=2), file_name='taylor_diagram.png', mime='image/png')
        except Exception:
            st.caption("💡 *Note: To enable PNG download, install the Kaleido package by running `pip install kaleido` in your terminal.*")
        
        st.markdown("#### 📊 Performance Metrics Summary")
        metrics_rows = []
        for model in selected_models:
            model_data = valid_ml[valid_ml['model_used'] == model] if 'model_used' in valid_ml.columns else valid_ml
            for variant_name, est_col in [('Raw P_merged', 'P_merged_hier'), ('ML-Corrected', 'P_corrected')]:
                est = model_data[est_col]
                obs_v = model_data['Gauge_Observed']
                r = np.corrcoef(obs_v, est)[0, 1]
                r2 = r ** 2
                rmse = np.sqrt(np.mean((est - obs_v)**2))
                bias = np.mean(est - obs_v)
                nse = 1 - (np.sum((obs_v - est)**2) / np.sum((obs_v - np.mean(obs_v))**2))
                metrics_rows.append({'Model': model, 'Variant': variant_name, 'R': round(r, 3), 'R²': round(r2, 3), 'NSE': round(nse, 3), 'RMSE (mm)': round(rmse, 2), 'Bias (mm)': round(bias, 2)})
        st.dataframe(pd.DataFrame(metrics_rows), hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 5: EXPLAINABLE AI (xAI)
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
# TAB 6: DROUGHT EVENT ANALYSIS
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
# TAB 7: FUTURE DROUGHT RISK PROJECTION (CMIP6)
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
