import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def f_metrics_monthly(df_test_final, y_true_col='unit_sales', y_pred_cols=['pred_xgb', 'baseline_sales']):
    """
    Compute RMSE, MAE, MAPE, RAE, bias, total sales, over and under delivery 
    for multiple prediction columns during each full month in the test set.
    Aggregates to month-item level before computing metrics.
    Returns per-month metrics and overall averages.
    Excludes the first 4 months and last 3 months of the test set from evaluation.
    """

    # --- 1. Input validation ---
    required_columns = ['date', y_true_col, 'item_nbr'] + y_pred_cols
    for col in required_columns:
        if col not in df_test_final.columns:
            raise ValueError(f"Missing required column: '{col}'")

    if not np.issubdtype(df_test_final['date'].dtype, np.datetime64):
        raise TypeError("Column 'date' must be in datetime format")

    df = df_test_final.copy()

    # --- 2. Add year and month columns ---
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # --- 3. Determine valid months (exclude first 4 and last 3 months) ---
    df_months = df[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).reset_index(drop=True)
    if len(df_months) <= 7:
        raise ValueError("Test set is too short to exclude first 4 and last 3 months")
    valid_months = df_months.iloc[4:-3]

    # --- 4. Filter df to only valid months ---
    df = df.merge(valid_months, on=['year', 'month'], how='inner')

    # --- 4. Loop through each month and calculate metrics ---
    monthly_metrics = []

    for _, row in df_months.iterrows():
        year = row['year']
        month = row['month']
        df_month = df[(df['year'] == year) & (df['month'] == month)]

        if df_month.empty:
            continue

        # Aggregate at month-item level
        df_item_month = df_month.groupby(['item_nbr', 'year', 'month'])[[y_true_col] + y_pred_cols].sum().reset_index()
        y_true = df_item_month[y_true_col].values

        # Compute metrics for each prediction column
        for y_pred_col in y_pred_cols:
            y_pred = df_item_month[y_pred_col].values
            errors = y_true - y_pred

            metrics = {
                'month': f"{year}-{month:02d}",
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
            monthly_metrics.append(metrics)

    # --- 5. Create summary table ---
    df_summary = pd.DataFrame(monthly_metrics)

    # --- 6. Add average row per prediction column ---
    avg_rows = df_summary.groupby('prediction').mean(numeric_only=True).reset_index()
    avg_rows['month'] = 'Average'
    df_summary = pd.concat([df_summary, avg_rows], ignore_index=True)

    # --- 7. Format for readability ---
    for col in ['rmse', 'mae', 'mape', 'rae', 'bias']:
        df_summary[col] = df_summary[col].round(2)
    for col in ['over_delivery', 'under_delivery', 'total_sales']:
        df_summary[col] = df_summary[col].astype(int)

    return df_summary

