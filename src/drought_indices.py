import pandas as pd
import numpy as np
from scipy import stats

def calculate_spi(precip_series, scale=12):
    """Calculates SPI using the Gamma distribution (McKee et al., 1993)."""
    if precip_series.isna().any():
        precip_series = precip_series.fillna(0)
    
    spi = pd.Series(np.nan, index=precip_series.index)
    p = precip_series.reset_index(drop=True)
    
    if scale > 1:
        precip_accumulated = p.rolling(window=scale, min_periods=scale).sum()
    else:
        precip_accumulated = p.copy()
        
    precip_accumulated.index = precip_series.index
        
    for month in range(1, 13):
        month_mask = precip_accumulated.index.month == month
        month_data = precip_accumulated[month_mask].dropna()
        
        if len(month_data) < 5:
            continue
            
        try:
            # FIX: Use .values to force NumPy array and avoid Pandas index alignment bugs
            alpha, loc, beta = stats.gamma.fit(month_data.values[month_data.values > 0] + 1e-6)
            prob = stats.gamma.cdf(month_data.values, alpha, loc, beta)
            prob = np.clip(prob, 0.0001, 0.9999)
            
            # FIX: Use .update() which is safe for partial index assignment
            spi.update(pd.Series(stats.norm.ppf(prob), index=month_data.index))
        except Exception as e:
            print(f"  ⚠️ SPI calculation failed for month {month}: {e}")
            continue
            
    return spi

def calculate_spei(precip_series, pet_series, scale=12):
    """Calculates SPEI using the Log-Logistic distribution (Vicente-Serrano et al., 2010)."""
    if precip_series.isna().any():
        precip_series = precip_series.fillna(0)
    
    if pet_series.isna().any():
        pet_series = pet_series.fillna(pet_series.mean())
    
    p = precip_series.reset_index(drop=True)
    pet = pet_series.reset_index(drop=True)
    
    water_balance = (p - pet).rolling(window=scale, min_periods=scale).sum()
    water_balance.index = precip_series.index
    
    spei = pd.Series(np.nan, index=water_balance.index)
    
    for month in range(1, 13):
        month_mask = water_balance.index.month == month
        month_data = water_balance[month_mask].dropna()
        
        if len(month_data) < 5:
            continue
            
        try:
            min_val = month_data.min()
            shift = abs(min_val) + 1e-6 if min_val <= 0 else 1e-6
            
            # FIX: Use .values to force NumPy array
            shifted_data = month_data.values + shift
            
            c, loc, scale_param = stats.fisk.fit(shifted_data)
            prob = stats.fisk.cdf(month_data.values + shift, c, loc, scale_param)
            prob = np.clip(prob, 0.0001, 0.9999)
            
            # FIX: Use .update() to safely assign back to the specific dates
            spei.update(pd.Series(stats.norm.ppf(prob), index=month_data.index))
        except Exception as e:
            print(f"  ⚠️ SPEI calculation failed for month {month}: {e}")
            continue
            
    return spei

def extract_drought_events(index_series, threshold=-1.0, min_duration=2, inter_event_time=1):
    """Implements Run Theory with tropical modifications to extract drought events."""
    events = []
    in_drought = False
    start_date = None
    current_severity = 0.0
    min_index = 0.0
    
    for date, value in index_series.items():
        if pd.isna(value):
            if in_drought:
                events.append({
                    'onset': start_date, 'termination': date, 
                    'duration': 0, 'severity': current_severity, 'min_index': min_index
                })
                in_drought = False
            continue
            
        if value <= threshold:
            if not in_drought:
                in_drought = True
                start_date = date
                current_severity = abs(value)
                min_index = value
            else:
                current_severity += abs(value)
                if value < min_index:
                    min_index = value
        else:
            if in_drought:
                in_drought = False
                duration_months = (date.year - start_date.year) * 12 + (date.month - start_date.month)
                events.append({
                    'onset': start_date, 
                    'termination': date, 
                    'duration': duration_months, 
                    'severity': current_severity, 
                    'min_index': min_index
                })
                
    if in_drought:
        last_date = index_series.index[-1]
        duration_months = (last_date.year - start_date.year) * 12 + (last_date.month - start_date.month) + 1
        events.append({
            'onset': start_date, 
            'termination': last_date, 
            'duration': duration_months, 
            'severity': current_severity, 
            'min_index': min_index
        })
        
    df_events = pd.DataFrame(events)
    if df_events.empty:
        return df_events
        
    df_events = df_events[df_events['duration'] >= min_duration]
    
    merged_events = []
    for i, row in df_events.iterrows():
        if merged_events:
            last_event = merged_events[-1]
            gap_months = (row['onset'].year - last_event['termination'].year) * 12 + (row['onset'].month - last_event['termination'].month)
            if gap_months <= inter_event_time:
                last_event['termination'] = row['termination']
                last_event['duration'] += row['duration']
                last_event['severity'] += row['severity']
                last_event['min_index'] = min(last_event['min_index'], row['min_index'])
                continue
        merged_events.append(row.to_dict())
        
    final_df = pd.DataFrame(merged_events)
    if not final_df.empty:
        final_df['intensity'] = final_df['severity'] / final_df['duration']
        
    return final_df