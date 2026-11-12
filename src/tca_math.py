# src/tca_math.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

class HierarchicalTCAFuser:
    """
    Hierarchical Triple Collocation Analysis (TCA) Engine.
    
    Scientific Rationale:
    ---------------------
    1. Multiplicative TCA: Precipitation is strictly positive and highly skewed.
       We apply natural log transformation (ln(X)) to stabilize variance before
       calculating error covariances (McColl et al., 2014).
    2. Hierarchical Stratification: Satellite error is non-stationary. We calculate
       local error variances stratified by Season (NEM, SWM, Inter) and Elevation
       Quartiles (Q1-Q4) to capture spatiotemporal error dynamics.
    3. Inverse Variance Weighting (IVW): Products with lower error variance receive
       higher weight in the fused product.
    4. Fallback Logic: If a specific stratum (e.g., NEM x Q4) has < 30 data points,
       the engine automatically falls back to the broader Season stratum, and 
       finally to the Global stratum to ensure mathematical stability.
    """

    def __init__(self, input_parquet_path):
        print("🔄 Loading master dataset for TCA...")
        self.df = pd.read_parquet(input_parquet_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['month'] = self.df['date'].dt.month
        
        # 1. Define Seasons (Sarawak Monsoon Context)
        def get_season(m):
            if m in [11, 12, 1, 2, 3]: return 'NEM'
            elif m in [5, 6, 7, 8, 9]: return 'SWM'
            else: return 'Inter-Monsoon'
        self.df['season'] = self.df['month'].apply(get_season)
        
        # 2. Define Elevation Quartiles (Q1=Lowest, Q4=Highest)
        # We use qcut to ensure equal number of stations per quartile
        self.df['elev_bin'], bins = pd.qcut(self.df['elev_m'], q=4, 
                                            labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                                            retbins=True, duplicates='drop')
        self.df['elev_bin'] = self.df['elev_bin'].astype(str)
        
        # 3. Create Hierarchical Stratum Key
        self.df['stratum_hier'] = self.df['season'] + '_' + self.df['elev_bin']
        
        print(f"✅ Data loaded: {len(self.df)} rows, {self.df['station_id'].nunique()} stations.")
        print(f"   Seasons: {self.df['season'].unique()}")
        print(f"   Elevation Bins: {self.df['elev_bin'].unique()}")

    @staticmethod
    def _calculate_tca_weights(x, y, z):
        """
        Core TCA Math (Multiplicative/Log-Space).
        Returns weights and error variances for X, Y, Z.
        """
        # Log transform
        x_l, y_l, z_l = np.log(x), np.log(y), np.log(z)
        
        # Covariances
        cov_xy = np.cov(x_l, y_l, ddof=1)[0, 1]
        cov_xz = np.cov(x_l, z_l, ddof=1)[0, 1]
        cov_yz = np.cov(y_l, z_l, ddof=1)[0, 1]
        
        # Variances
        var_x = np.var(x_l, ddof=1)
        var_y = np.var(y_l, ddof=1)
        var_z = np.var(z_l, ddof=1)
        
        # Error Variances (McColl et al., 2014 Eq 3)
        # Add small epsilon to prevent division by zero if covariance is 0
        eps = 1e-9
        ev_x = var_x - (cov_xy * cov_xz) / max(cov_yz, eps)
        ev_y = var_y - (cov_xy * cov_yz) / max(cov_xz, eps)
        ev_z = var_z - (cov_xz * cov_yz) / max(cov_xy, eps)
        
        # Handle negative error variances (violates TCA assumptions)
        # Fallback: set to small positive value
        if ev_x <= 0: ev_x = eps
        if ev_y <= 0: ev_y = eps
        if ev_z <= 0: ev_z = eps
        
        # Inverse Variance Weights (Normalized to sum to 1)
        inv_x, inv_y, inv_z = 1/ev_x, 1/ev_y, 1/ev_z
        total_inv = inv_x + inv_y + inv_z
        
        w_x = inv_x / total_inv
        w_y = inv_y / total_inv
        w_z = inv_z / total_inv
        
        return w_x, w_y, w_z, ev_x, ev_y, ev_z

    def _get_weights_for_stratum(self, data_subset, min_samples=30):
        """Calculate weights for a subset, returning None if too few samples."""
        # Strict wet days filter (all 3 SPEs > 0) for log transform
        valid = data_subset[
            (data_subset['precip_chirps'] > 0) & 
            (data_subset['precip_era5_land'] > 0) & 
            (data_subset['precip_persiann'] > 0)
        ]
        
        if len(valid) < min_samples:
            return None
            
        x = valid['precip_chirps'].values
        y = valid['precip_persiann'].values
        z = valid['precip_era5_land'].values
        
        return self._calculate_tca_weights(x, y, z)

    def run_fusion(self, output_parquet_path):
        """
        Execute Hierarchical TCA and generate 4 P_merged columns.
        """
        print("\n Starting Hierarchical TCA Fusion...")
        
        # 1. Calculate Global Weights
        print("   -> Calculating Global weights...")
        global_weights = self._get_weights_for_stratum(self.df, min_samples=100)
        if global_weights is None:
            raise ValueError("Not enough global data for TCA.")
        gw_c, gw_p, gw_e = global_weights[0], global_weights[1], global_weights[2]
        
        # 2. Calculate Seasonal Weights
        print("   -> Calculating Seasonal weights...")
        season_weights = {}
        for season in self.df['season'].unique():
            sub = self.df[self.df['season'] == season]
            w = self._get_weights_for_stratum(sub)
            if w: season_weights[season] = w
        
        # 3. Calculate Elevation Weights
        print("   -> Calculating Elevation weights...")
        elev_weights = {}
        for eb in self.df['elev_bin'].unique():
            sub = self.df[self.df['elev_bin'] == eb]
            w = self._get_weights_for_stratum(sub)
            if w: elev_weights[eb] = w
            
        # 4. Calculate Hierarchical Weights (Season x Elevation)
        print("   -> Calculating Hierarchical weights (Season x Elevation)...")
        hier_weights = {}
        for sh in self.df['stratum_hier'].unique():
            sub = self.df[self.df['stratum_hier'] == sh]
            w = self._get_weights_for_stratum(sub, min_samples=30)
            if w: hier_weights[sh] = w
            
        # 5. Map Weights and Calculate P_merged
        print("   -> Mapping weights and calculating P_merged products...")
        
        # Initialize columns
        self.df['P_merged_global'] = np.nan
        self.df['P_merged_season'] = np.nan
        self.df['P_merged_elev'] = np.nan
        self.df['P_merged_hier'] = np.nan
        
        # Log transform main data (handle zeros by adding tiny epsilon for log, but mask later)
        # Actually, we only calculate P_merged where all 3 > 0 to avoid log(0)
        valid_mask = (
            (self.df['precip_chirps'] > 0) & 
            (self.df['precip_era5_land'] > 0) & 
            (self.df['precip_persiann'] > 0)
        )
        
        log_c = np.log(self.df.loc[valid_mask, 'precip_chirps'])
        log_p = np.log(self.df.loc[valid_mask, 'precip_persiann'])
        log_e = np.log(self.df.loc[valid_mask, 'precip_era5_land'])
        
        # Global
        self.df.loc[valid_mask, 'P_merged_global'] = np.exp(
            gw_c * log_c + gw_p * log_p + gw_e * log_e
        )
        
        # Season, Elev, Hier (with fallback logic)
        for idx in self.df[valid_mask].index:
            row = self.df.loc[idx]
            s, eb, sh = row['season'], row['elev_bin'], row['stratum_hier']
            
            # Season
            w_s = season_weights.get(s, global_weights)
            self.df.loc[idx, 'P_merged_season'] = np.exp(w_s[0]*log_c[idx] + w_s[1]*log_p[idx] + w_s[2]*log_e[idx])
            
            # Elevation
            w_eb = elev_weights.get(eb, global_weights)
            self.df.loc[idx, 'P_merged_elev'] = np.exp(w_eb[0]*log_c[idx] + w_eb[1]*log_p[idx] + w_eb[2]*log_e[idx])
            
            # Hierarchical (Fallback: Hier -> Season -> Global)
            w_h = hier_weights.get(sh, season_weights.get(s, global_weights))
            self.df.loc[idx, 'P_merged_hier'] = np.exp(w_h[0]*log_c[idx] + w_h[1]*log_p[idx] + w_h[2]*log_e[idx])
            
        # 6. Save Output
        out_path = Path(output_parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(out_path, compression='snappy', index=False)
        
        print(f"\n✅ SUCCESS! Saved TCA-fused dataset to: {out_path}")
        print(f"   Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        # Print Summary of Weights
        print("\n📊 TCA Weight Summary (Multiplicative Model):")
        print(f"   Global: CHIRPS={gw_c:.3f}, PERSIANN={gw_p:.3f}, ERA5={gw_e:.3f}")
        for s, w in season_weights.items():
            print(f"   {s}: CHIRPS={w[0]:.3f}, PERSIANN={w[1]:.3f}, ERA5={w[2]:.3f}")
            
        return self.df