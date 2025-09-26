from typing import Optional, Union
import pandas as pd
import numpy as np
from prophet import Prophet
from tqdm import tqdm

def f_prophet_model(
    sales_history: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    prophet_kwargs: Optional[dict] = None
) -> pd.DataFrame:
    """
    Maak een forecast per winkel (store) met Prophet en verdeel deze over items
    met benchmarkgewichten gebaseerd op historische verkoop.

    Parameters
    ----------
    sales_history : pd.DataFrame
        Volledige historische verkoopdata, gebruikt voor het berekenen van weights en
        om echte sales toe te voegen aan het resultaat.
        Vereiste kolommen:
        - 'store_nbr'   : int of str, winkel-ID
        - 'item_nbr'    : int of str, product-ID
        - 'date'        : datetime of str
        - 'unit_sales'  : float, aantal verkochte eenheden
        - 'perishable'  : int (0/1), indicator voor bederfelijkheid
    train : pd.DataFrame
        Trainingssubset (data vóór split). Vereiste kolommen:
        - 'store_nbr', 'date', 'unit_sales'
    test : pd.DataFrame
        Testsubset (data ná split, waarvoor voorspellingen nodig zijn).
        Vereiste kolommen:
        - 'store_nbr', 'date'
    prophet_kwargs : dict, optional
        Optionele parameters om door te geven aan Prophet().
        Default instellingen zijn:
        {
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": True,
            "interval_width": 0.8,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0
        }

    Returns
    -------
    pd.DataFrame
        DataFrame met per store × date × item voorspellingen en aanvullende kolommen:
        - 'store_nbr', 'date', 'item_nbr'
        - 'weight_benchmark' : berekend gewicht voor het item
        - 'forecast_total'   : totale voorspelling van de winkel
        - 'yhat_item'        : voorspelde verkoop voor het item
        - 'unit_sales'       : daadwerkelijke verkoop (indien beschikbaar)
        - 'weight'           : gewicht (1.25 voor perishable, anders 1.0)
    """
    # --- Preprocessing ---
    sales_history = sales_history.copy()
    sales_history['date'] = pd.to_datetime(sales_history['date'])
    sales_history['weight'] = np.where(sales_history.get('perishable', 0) == 1, 1.25, 1.0)
    sales_history['quarter'] = sales_history['date'].dt.quarter
    sales_history['weekday'] = sales_history['date'].dt.weekday

    train = train.copy()
    test = test.copy()
    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])

    # --- Prophet default settings ---
    default_kwargs = {
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": True,
        "interval_width": 0.8,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0
    }
    if prophet_kwargs is None:
        prophet_kwargs = default_kwargs
    else:
        # override defaults met user-input
        prophet_kwargs = {**default_kwargs, **prophet_kwargs}

    # --- Stap 1: Forecast per store × dag ---
    all_forecasts_list: list[pd.DataFrame] = []
    stores_to_forecast = test['store_nbr'].unique()

    for store in tqdm(stores_to_forecast, desc="Forecast per store"):
        train_store = train[train['store_nbr'] == store].copy()
        if train_store.empty:
            continue

        train_daily = (
            train_store.groupby('date')['unit_sales']
            .sum()
            .reset_index()
            .rename(columns={'unit_sales': 'y'})
        )
        train_daily['ds'] = train_daily['date']

        test_dates = sorted(test[test['store_nbr'] == store]['date'].unique())
        if len(test_dates) == 0 or len(train_daily) < 2:
            continue
        test_df = pd.DataFrame({'ds': pd.to_datetime(test_dates)})

        model = Prophet(**prophet_kwargs)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5, mode='multiplicative')
        model.add_seasonality(name='semi_annual', period=182.5, fourier_order=3, mode='multiplicative')

        model.fit(train_daily[['ds', 'y']])
        forecast = model.predict(test_df[['ds']])

        daily_forecast = pd.DataFrame({
            'store_nbr': store,
            'date': forecast['ds'],
            'forecast_total': forecast['yhat']
        })
        all_forecasts_list.append(daily_forecast)

    if not all_forecasts_list:
        return pd.DataFrame(columns=[
            'store_nbr','date','item_nbr','weight_benchmark',
            'forecast_total','yhat_item','unit_sales','weight'
        ])

    all_store_forecasts = pd.concat(all_forecasts_list, ignore_index=True)

    # --- Stap 2: Bereken benchmarkgewichten per item ---
    def get_weights(store: Union[int, str], date: pd.Timestamp, sales_df: pd.DataFrame) -> Optional[pd.Series]:
        weekday = pd.to_datetime(date).weekday()
        past_dates = [pd.to_datetime(date) - pd.Timedelta(weeks=w) for w in range(1, 5)]
        mask = (
            (sales_df['store_nbr'] == store) &
            (sales_df['date'].isin(past_dates)) &
            (sales_df['weekday'] == weekday)
        )
        past_data = sales_df[mask]
        if past_data.empty:
            return None
        total_sales = past_data.groupby('item_nbr')['unit_sales'].sum()
        if total_sales.sum() == 0:
            return None
        return total_sales / total_sales.sum()

    weights_list: list[dict] = []
    for _, row in tqdm(all_store_forecasts.iterrows(), total=len(all_store_forecasts), desc="Benchmarkgewichten"):
        store = row['store_nbr']
        date = row['date']
        weights = get_weights(store, date, sales_history)
        if weights is not None:
            for item_nbr, w in weights.items():
                weights_list.append({
                    'store_nbr': store,
                    'date': pd.to_datetime(date),
                    'item_nbr': item_nbr,
                    'weight_benchmark': float(w)
                })

    if not weights_list:
        return pd.DataFrame(columns=[
            'store_nbr','date','item_nbr','weight_benchmark',
            'forecast_total','yhat_item','unit_sales','weight'
        ])

    weights_df = pd.DataFrame(weights_list)

    # --- Stap 3: Verdeel forecast over items ---
    merged = weights_df.merge(all_store_forecasts, on=['store_nbr', 'date'], how='left')
    merged['yhat_item'] = merged['forecast_total'] * merged['weight_benchmark']

    merged = merged.merge(
        sales_history[['store_nbr', 'date', 'item_nbr', 'unit_sales', 'weight']],
        on=['store_nbr', 'date', 'item_nbr'],
        how='left'
    )

    return merged.sort_values(['store_nbr', 'date', 'item_nbr']).reset_index(drop=True)
