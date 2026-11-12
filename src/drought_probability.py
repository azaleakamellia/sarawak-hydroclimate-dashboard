import pandas as pd
import numpy as np

def calculate_return_periods(events_df, total_months):
    """
    Calculates the Return Period (Tr) for each drought event based on Severity.
    Tr = (N + 1) / m  (Weibull plotting position)
    Where N = total years, m = rank of the event (1 = most severe).
    """
    if events_df.empty:
        return events_df
        
    df = events_df.copy()
    
    # Rank events by severity (1 = most severe)
    # ascending=False means highest severity gets rank 1
    df['rank'] = df['severity'].rank(ascending=False, method='min')
    
    # Calculate Return Period in Years
    total_years = total_months / 12.0
    df['return_period_years'] = (total_years + 1) / df['rank']
    
    # Calculate Probability of Exceedance (Confidence Level)
    # This is the probability that an event of this severity or worse occurs in any given year
    df['probability_exceedance'] = 1.0 / df['return_period_years']
    
    # Categorize Drought (McKee et al. 1993)
    # FIX: Changed 'min_spi' to 'min_index' to support both SPI and SPEI
    conditions = [
        (df['min_index'] <= -2.0),
        (df['min_index'] <= -1.5),
        (df['min_index'] <= -1.0)
    ]
    choices = ['Extreme', 'Severe', 'Moderate']
    df['category'] = np.select(conditions, choices, default='Normal/Wet')
    
    return df.sort_values('onset')