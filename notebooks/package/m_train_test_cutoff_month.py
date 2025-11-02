import pandas as pd

def f_split_time_series_month(df, date_col="date", test_months=5, buffer_months=4):
    """
    Splits time series into training/validation and test sets.

    Parameters:
    - df: DataFrame with time series data
    - date_col: column containing dates
    - test_months: number of months for testing/evaluation, take into account that the last 3 months
    can not be used for evaluation due to unkown 3 months ahead targets
    - buffer_months: number of buffer months before test period (for lags)

    Returns:
    - df_trainval: DataFrame for training/validation
    - df_test_final: DataFrame for testing
    - cutoff_date: last date included in training/validation
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Total months to exclude from training/validation
    total_exclude = pd.DateOffset(months=test_months + buffer_months)

    cutoff_date = df.index.max() - total_exclude  # last date for training/validation
    test_start_date = cutoff_date + pd.DateOffset(months=buffer_months)  # start of test period

    df_trainval = df[df.index <= cutoff_date].copy()
    df_test_final = df[df.index >= test_start_date].copy()

    # Make sure date_col is also a column
    df_trainval[date_col] = df_trainval.index
    df_test_final[date_col] = df_test_final.index

    return df_trainval, df_test_final, cutoff_date

