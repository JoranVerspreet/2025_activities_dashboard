import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

def f_metrics_daily(df_test_final, y_true_col='unit_sales', y_pred_cols=['pred_xgb', 'baseline_sales']):
    """
    Compute weekly metrics for multiple prediction columns in df_test_final.
    Metrics are calculated at store-item-day level (no aggregation).
    Returns a summary table with weekly metrics for each prediction column.
    """
    # --- 1. Input validation ---
    required_columns = ['date', y_true_col, 'store_nbr', 'item_nbr'] + y_pred_cols
    for col in required_columns:
        if col not in df_test_final.columns:
            raise ValueError(f"Missing required column: '{col}'")
    
    if not np.issubdtype(df_test_final['date'].dtype, np.datetime64):
        raise TypeError("Column 'date' must be in datetime format")
    
    df = df_test_final.copy()
    
    # --- 2. Add ISO week/year columns ---
    df['iso_year'] = df['date'].dt.isocalendar().year
    df['iso_week'] = df['date'].dt.isocalendar().week
    
    # --- 3. Determine starting week (min_week + 8) ---
    min_date = df['date'].min()
    min_year, min_week, _ = min_date.isocalendar()
    start_week = min_week + 8
    start_year = min_year
    max_week = datetime.date(min_year, 12, 28).isocalendar()[1]

    if start_week > max_week:
        start_week -= max_week
        start_year += 1
    
    # --- 4. Get all unique (year, week) pairs after start_week ---
    df_weeks = df[['iso_year', 'iso_week']].drop_duplicates()
    df_weeks = df_weeks[
        (df_weeks['iso_year'] > start_year) |
        ((df_weeks['iso_year'] == start_year) & (df_weeks['iso_week'] >= start_week))
    ].sort_values(['iso_year', 'iso_week'])
    
    # --- 5. Compute metrics per week for each prediction column ---
    all_metrics = []

    for _, row in df_weeks.iterrows():
        year = row['iso_year']
        week = row['iso_week']
        df_week = df[(df['iso_year'] == year) & (df['iso_week'] == week)]
        if df_week.empty:
            continue

        y_true = df_week[y_true_col].values

        for y_pred_col in y_pred_cols:
            y_pred = df_week[y_pred_col].values
            errors = y_true - y_pred
            metrics = {
                'week': f"{year}-W{week:02d}",
                'prediction': y_pred_col,
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mape': np.mean(np.abs(errors / np.maximum(y_true, 1e-6))) * 100,
                'rae': np.sum(np.abs(errors)) / np.sum(y_true) * 100,
                'bias': np.sum(errors) / np.sum(y_true) * 100,
                'over_delivery': np.sum(np.where(errors < 0, -errors, 0)),
                'under_delivery': np.sum(np.where(errors > 0, errors, 0)),
                'total_sales': np.sum(y_true)
            }
            all_metrics.append(metrics)
    
    # --- 6. Create summary table ---
    df_summary = pd.DataFrame(all_metrics)

    # --- 7. Add average row per prediction column ---
    avg_rows = df_summary.groupby('prediction').mean(numeric_only=True).reset_index()
    avg_rows['week'] = 'Average'
    
    return avg_rows



