import pandas as pd

def f_baseline_model(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline voorspellingsmodel op basis van gemiddelde verkopen per weekdag.

    Voor elke combinatie van store en item berekent de functie het gemiddelde 
    aantal verkopen per weekdag over de afgelopen `lookback_days` dagen en 
    gebruikt dit als voorspelling voor toekomstige data.

    Parameters
    ----------
    test_df : pd.DataFrame
        DataFrame met in ieder geval de kolommen:
        - 'date' (datetime64)
        - 'store_nbr' (int/str)
        - 'item_nbr' (int/str)
        - 'unit_sales' (float/int)

    Returns
    -------
    pd.DataFrame
        DataFrame met dezelfde rijen als `test_df`, maar uitgebreid met een kolom:
        - 'predicted_sales' : float
          de voorspelde verkopen (0 indien geen voorspelling mogelijk).
    """
    
    lookback_days = 28
    group_cols = ['store_nbr', 'item_nbr']
    
    start_date = test_df['date'].min() + pd.Timedelta(days=lookback_days)
    end_date = test_df['date'].max()
    prediction_dates = pd.date_range(start=start_date, end=end_date)

    all_preds = []

    for date in prediction_dates:
        cutoff = date - pd.Timedelta(days=1)
        lookback_start = cutoff - pd.Timedelta(days=lookback_days)

        history = test_df[(test_df['date'] > lookback_start) & (test_df['date'] <= cutoff)]

        for (store, item), group in history.groupby(group_cols):
            weekday_avg = group.groupby(group['date'].dt.dayofweek)['unit_sales'].mean()
            weekday_avg = weekday_avg.reindex(range(7), fill_value=0)
            predicted = weekday_avg[date.dayofweek]

            all_preds.append({
                'date': date,
                'store_nbr': store,
                'item_nbr': item,
                'predicted_sales': predicted
            })

    # Create predictions DataFrame
    baseline_df = pd.DataFrame(all_preds)

    # Ensure matching dtypes
    test_df['item_nbr'] = test_df['item_nbr'].astype(str)
    baseline_df['item_nbr'] = baseline_df['item_nbr'].astype(str)

    # Merge actuals with predictions
    merged_all = pd.merge(
        baseline_df,
        test_df,
        on=['date', 'store_nbr', 'item_nbr'],
        how='left'
    )

    # Fill missing predictions
    merged_all['predicted_sales'] = merged_all['predicted_sales'].fillna(0)

    return merged_all
