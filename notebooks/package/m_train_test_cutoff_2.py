import pandas as pd

def f_split_time_series_2(df, date_col="date", test_days=90, buffer_days=3):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    test_buffer_range = pd.Timedelta(days=test_days) + pd.Timedelta(days=buffer_days)

    cutoff_date = df.index.max() - test_buffer_range
    test_start_date = cutoff_date + pd.Timedelta(days=buffer_days)

    df_trainval = df[df.index <= cutoff_date].copy()
    df_test_final = df[df.index > test_start_date].copy()

    # Make sure date_col is also a column
    df_trainval[date_col] = df_trainval.index
    df_test_final[date_col] = df_test_final.index

    return df_trainval, df_test_final, cutoff_date
