import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SHAPExplainer:
    def __init__(self, parquet_path, model_dir):
        self.parquet_path = Path(parquet_path)
        self.model_dir = Path(model_dir)
        
        # Load the default best model
        model_path = self.model_dir / "best_ml_model.pkl"
        self.model = joblib.load(model_path)
        self.model_type = type(self.model).__name__
        
        print(f" Loaded model type: {self.model_type}")
        
        # Load the data
        df = pd.read_parquet(self.parquet_path)
        
        # CRITICAL FIX: Recreate lulc_code
        if 'lulc_class' in df.columns:
            df['lulc_code'] = pd.Categorical(df['lulc_class']).codes
        elif 'lulc_code' not in df.columns:
            raise ValueError("Neither 'lulc_class' nor 'lulc_code' found in data.")
            
        self.feature_cols = [
            'elev_m', 'lulc_code', 'month', 
            'precip_chirps', 'precip_era5_land', 'precip_persiann', 'P_merged_hier'
        ]
        
        required_cols = self.feature_cols + ['Gauge_Observed']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        self.df_valid = df.dropna(subset=required_cols).copy()
        print(f"✅ Data loaded. {len(self.df_valid)} valid samples available.")

    def compute_shap_values(self, sample_size=2000):
        if len(self.df_valid) > sample_size:
            X_sample = self.df_valid.sample(sample_size, random_state=42)[self.feature_cols]
            original_sample_df = self.df_valid.sample(sample_size, random_state=42).copy()
        else:
            X_sample = self.df_valid[self.feature_cols].copy()
            original_sample_df = self.df_valid.copy()
            
        print(f"🧮 Computing SHAP for {self.model_type} (n={len(X_sample)})...")
        
        try:
            if self.model_type in ['RandomForestRegressor', 'GradientBoostingRegressor', 'ExtraTreesRegressor', 'HistGradientBoostingRegressor']:
                print("   → Using TreeExplainer (optimized for tree-based models)")
                self.explainer = shap.TreeExplainer(self.model)
                shap_values = self.explainer.shap_values(X_sample)
            elif self.model_type == 'MLPRegressor':
                print("   → Using PermutationExplainer (for MLP neural network)")
                print("   ️ This may take a few minutes...")
                try:
                    self.explainer = shap.PermutationExplainer(self.model, X_sample, nsamples=50)
                    shap_values = self.explainer.shap_values(X_sample)
                except (AttributeError, TypeError) as e:
                    print(f"   → PermutationExplainer failed: {e}")
                    print("   → Falling back to KernelExplainer")
                    background = X_sample.sample(min(100, len(X_sample)))
                    self.explainer = shap.KernelExplainer(self.model.predict, background)
                    shap_values = self.explainer.shap_values(X_sample)
            else:
                print(f"   → Using KernelExplainer (fallback for {self.model_type})")
                background = X_sample.sample(min(100, len(X_sample)))
                self.explainer = shap.KernelExplainer(self.model.predict, background)
                shap_values = self.explainer.shap_values(X_sample)
        except Exception as e:
            print(f"❌ Error computing SHAP values: {e}")
            raise
            
        print("✅ SHAP values computed successfully.")
        return shap_values, X_sample, original_sample_df

    def get_global_importance(self, shap_values):
        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[0])
        elif hasattr(shap_values, 'values'):
            shap_vals = np.array(shap_values.values)
        else:
            shap_vals = np.array(shap_values)
        
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        display_names = [col.replace('lulc_code', 'LULC Class') for col in self.feature_cols]
        
        return pd.DataFrame({
            'Feature': display_names,
            'Mean |SHAP|': mean_abs_shap
        }).sort_values('Mean |SHAP|', ascending=True)

    def get_beeswarm_data(self, shap_values, X_sample, original_sample_df):
        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[0])
        elif hasattr(shap_values, 'values'):
            shap_vals = np.array(shap_values.values)
        else:
            shap_vals = np.array(shap_values)
        
        n_samples, n_features = shap_vals.shape
        display_names = [col.replace('lulc_code', 'LULC Class') for col in self.feature_cols]
        
        features_flat = []
        shap_vals_flat = []
        feature_vals_flat = []
        months_flat = []
        
        for i in range(n_features):
            features_flat.extend([display_names[i]] * n_samples)
            shap_vals_flat.extend(shap_vals[:, i])
            feature_vals_flat.extend(X_sample.iloc[:, i])
            months_flat.extend(original_sample_df['month'].values)
            
        beeswarm_df = pd.DataFrame({
            'Feature': features_flat,
            'SHAP Value': shap_vals_flat,
            'Feature Value': feature_vals_flat,
            'Month': months_flat
        })
        
        def get_season(m):
            if m in [11, 12, 1, 2]: return 'NEM'
            if m in [5, 6, 7, 8, 9]: return 'SWM'
            return 'Inter'
            
        beeswarm_df['Season'] = beeswarm_df['Month'].apply(get_season)
        return beeswarm_df