

import pandas as pd

def f_split_time_series(df, date_col="date", test_days=90):
    """
    Splits a time series DataFrame into train/validation and final test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a datetime column.
    date_col : str, default="date"
        The name of the datetime column to use as index.
    test_days : int, default=90
        Number of days to hold out for the final test set.

    Returns
    -------
    df_trainval : pd.DataFrame
        Training + validation dataset (before cutoff_date).
    df_test_final : pd.DataFrame
        Final test dataset (after cutoff_date).
    cutoff_date : pd.Timestamp
        The cutoff date separating train/validation from test.
    """

    # Ensure datetime index
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Define hold-out test set
    cutoff_date = df.index.max() - pd.Timedelta(days=test_days)

    df_trainval = df[df.index <= cutoff_date]
    df_test_final = df[df.index > cutoff_date]

    return df_trainval, df_test_final, cutoff_date

