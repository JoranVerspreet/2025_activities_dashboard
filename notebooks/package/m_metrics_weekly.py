import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

def f_metrics_weekly(df_test_final, y_true_col='unit_sales', y_pred_cols=['pred_xgb', 'baseline_sales']):
    """
    Compute rolling 2-week prediction metrics (aggregated per store) in df_test_final.
    - Skips the first 8 weeks
    - Evaluates each 2-week prediction window until the end of the test period
    - Aggregates at the store level
    - Returns average metrics per prediction column across all windows
    """
    # --- 1. Input validation ---
    required_columns = ['date', y_true_col, 'store_nbr'] + y_pred_cols
    for col in required_columns:
        if col not in df_test_final.columns:
            raise ValueError(f"Missing required column: '{col}'")

    if not np.issubdtype(df_test_final['date'].dtype, np.datetime64):
        raise TypeError("Column 'date' must be datetime")

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

    # --- 4. Identify all valid (year, week) pairs after start_week ---
    df_weeks = df[['iso_year', 'iso_week']].drop_duplicates()
    df_weeks = df_weeks[
        (df_weeks['iso_year'] > start_year) |
        ((df_weeks['iso_year'] == start_year) & (df_weeks['iso_week'] >= start_week))
    ].sort_values(['iso_year', 'iso_week']).reset_index(drop=True)

    # --- Exclude the last 2 weeks of the test period ---
    max_year, max_week, _ = df['date'].max().isocalendar()
    df_weeks = df_weeks[~((df_weeks['iso_year'] == max_year) & (df_weeks['iso_week'] >= max_week-2))]


    # --- 5. Rolling 2-week windows evaluation ---
    all_metrics = []

    for i in range(0, len(df_weeks), 2):  # step through 2-week windows
        window = df_weeks.iloc[i:i+2]
        if len(window) == 0:
            continue

        # Filter for these 2 weeks
        df_window = df.merge(window, on=['iso_year', 'iso_week'])
        if df_window.empty:
            continue

        # Aggregate weekly sales per store
        weekly_df = (
            df_window.groupby(['store_nbr', 'iso_year', 'iso_week'])[[y_true_col] + y_pred_cols]
            .sum()
            .reset_index()
        )

        y_true = weekly_df[y_true_col].values

        for y_pred_col in y_pred_cols:
            y_pred = weekly_df[y_pred_col].values
            errors = y_true - y_pred
            if np.sum(y_true) == 0:
                continue  # skip this window


            metrics = {
                'weeks': ', '.join([f"{int(y)}-W{int(w):02d}" for y, w in zip(window['iso_year'], window['iso_week'])]),
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

    # --- 7. Compute overall averages per prediction type ---
    avg_rows = df_summary.groupby('prediction').mean(numeric_only=True).reset_index()
    avg_rows['weeks'] = 'Average (2-week rolling)'
    df_summary = pd.concat([df_summary, avg_rows], ignore_index=True)

    # --- 8. Format for readability ---
    for col in ['rmse', 'mae', 'mape', 'rae', 'bias']:
        df_summary[col] = df_summary[col].round(2)
    for col in ['over_delivery', 'under_delivery', 'total_sales']:
        df_summary[col] = df_summary[col].astype(int)

    return df_summary
