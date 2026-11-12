# src/ml_bias_correction.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLBiasCorrector:
    """
    Multi-Model Machine Learning Bias Correction Engine.
    Evaluates multiple algorithms across all P_merged variants to find the 
    optimal fusion-correction pipeline.
    """

    def __init__(self, input_parquet_path):
        print("🔄 Loading TCA-fused dataset for ML Bias Correction...")
        self.df = pd.read_parquet(input_parquet_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        
        # Define the 4 P_merged variants to test
        self.p_merged_cols = [
            'P_merged_global', 'P_merged_season', 
            'P_merged_elev', 'P_merged_hier'
        ]
        
        print(f"✅ Loaded {len(self.df)} rows. Testing {len(self.p_merged_cols)} fusion variants.")

    def _prepare_features(self, p_col):
        """Engineers features and calculates residual bias for a specific P_merged."""
        df = self.df.copy()
        
        # Target: Residual Bias (Gauge - P_merged). 
        # If positive, satellite underestimated. If negative, satellite overestimated.
        df['residual_bias'] = df['Gauge_Observed'] - df[p_col]
        
        # Features: Physical covariates + Satellite signals
        # We include the raw SPEs so the ML model can learn magnitude-dependent biases
        feature_cols = [
            'elev_m', 'lulc_class', 'month',
            'precip_chirps', 'precip_era5_land', 'precip_persiann', p_col
        ]
        
        # Encode LULC
        df['lulc_code'] = pd.Categorical(df['lulc_class']).codes
        feature_cols[1] = 'lulc_code'
        
        # Drop NaNs
        valid_mask = df[feature_cols + ['residual_bias']].notna().all(axis=1)
        df_valid = df[valid_mask].copy()
        
        return df_valid, feature_cols

    def train_and_evaluate_all(self, output_parquet_path):
        """Trains multiple models on all P_merged variants and selects the best."""
        
        # Define the models to test (The "MATLAB" equivalents)
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(50, 20), max_iter=1000, random_state=42)
        }
        
        results = []
        best_overall_score = -999
        best_config = None
        
        # 1. Loop through Fusion Variants
        for p_col in self.p_merged_cols:
            print(f"\n🚀 Evaluating Fusion Variant: {p_col}")
            df_valid, feature_cols = self._prepare_features(p_col)
            
            X = df_valid[feature_cols].values
            y = df_valid['residual_bias'].values
            
            # 2. Chronological Split (80% Train / 20% Test)
            # Sort by date first to ensure no future data leaks into training
            df_valid = df_valid.sort_values('date')
            split_idx = int(len(df_valid) * 0.8)
            
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features (Crucial for SVR, MLP, and Linear Regression)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 3. Loop through ML Models
            for model_name, model in models.items():
                print(f"   -> Training {model_name}...")
                
                # Train
                if model_name in ['Linear Regression', 'Random Forest', 'Gradient Boosting']:
                    model.fit(X_train, y_train) # Tree/Linear models don't strictly need scaling, but it's safe
                    y_pred = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train) # NN/SVR need scaling
                    y_pred = model.predict(X_test_scaled)
                
                # Evaluate
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Calculate KGE (Kling-Gupta Efficiency)
                r_kge = np.corrcoef(y_test, y_pred)[0, 1]
                alpha = np.std(y_pred) / np.std(y_test)
                beta = np.mean(y_pred) / np.mean(y_test)
                kge = 1 - np.sqrt((r_kge - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                
                # Composite Score (Higher is better. We want high KGE/R2, low RMSE)
                # Normalized roughly for comparison
                score = (kge + r2) / 2 - (rmse / 1000) 
                
                results.append({
                    'Fusion_Method': p_col.replace('P_merged_', '').title(),
                    'ML_Model': model_name,
                    'RMSE': rmse,
                    'R2': r2,
                    'KGE': kge,
                    'Composite_Score': score
                })
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_config = {
                        'fusion': p_col,
                        'model_name': model_name,
                        'model_obj': model,
                        'scaler': scaler,
                        'feature_cols': feature_cols
                    }

        # 4. Generate Leaderboard
        results_df = pd.DataFrame(results).sort_values('Composite_Score', ascending=False)
        print("\n" + "="*60)
        print("🏆 ML MODEL LEADERBOARD (Ranked by Composite Score)")
        print("="*60)
        print(results_df[['Fusion_Method', 'ML_Model', 'RMSE', 'R2', 'KGE']].to_string(index=False))
        print("="*60)
        
        # 5. Apply the BEST model to the full dataset
        print(f"\n🌟 WINNING CONFIG: {best_config['fusion']} + {best_config['model_name']}")
        print("🔄 Applying best model to full historical dataset...")
        
        df_full = self.df.copy()
        df_full, feat_cols = self._prepare_features(best_config['fusion'])
        
        X_full = df_full[feat_cols].values
        if best_config['model_name'] in ['Neural Network (MLP)']:
            X_full_scaled = best_config['scaler'].transform(X_full)
            predicted_bias = best_config['model_obj'].predict(X_full_scaled)
        else:
            predicted_bias = best_config['model_obj'].predict(X_full)
            
        # Correct the P_merged
        df_full['P_corrected'] = df_full[best_config['fusion']] + predicted_bias
        df_full['P_corrected'] = df_full['P_corrected'].clip(lower=0) # Rain can't be negative
        
        # Save
        out_path = Path(output_parquet_path)
        df_full.to_parquet(out_path, compression='snappy', index=False)
        
        # Save the winning model and scaler for future use (e.g., CMIP6, XAI)
        model_path = out_path.parent / "best_ml_model.pkl"
        joblib.dump(best_config['model_obj'], model_path)
        joblib.dump(best_config['scaler'], out_path.parent / "scaler.pkl")
        
        print(f"💾 Saved corrected dataset to: {out_path}")
        print(f"💾 Saved winning model to: {model_path}")
        
        return results_df, df_full