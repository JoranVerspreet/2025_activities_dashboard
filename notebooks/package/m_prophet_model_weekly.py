from typing import Optional
import pandas as pd
import numpy as np
from prophet import Prophet
from tqdm import tqdm

def f_prophet_model_weekly(
    sales_history: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    prophet_kwargs: Optional[dict] = None,
    rolling_weeks: int = 4
) -> pd.DataFrame:
    """
    Forecast per store op weekniveau met Prophet, uitsmeren over items en weekdagen,
    veilige aanpak voor eerste/laatste week: rolling average voor incomplete weken.

    Parameters
    ----------
    sales_history : pd.DataFrame
        Vereiste kolommen: 'store_nbr', 'item_nbr', 'date', 'unit_sales'
    train : pd.DataFrame
        Trainingsdata vóór split. Kolommen: 'store_nbr', 'date', 'unit_sales'
    test : pd.DataFrame
        Testdata na split. Kolommen: 'store_nbr', 'date'
    prophet_kwargs : dict, optional
        Parameters voor Prophet.
    rolling_weeks : int
        Aantal weken voor rolling average fallback.

    Returns
    -------
    pd.DataFrame
        Columns: ['store_nbr','date','item_nbr','weight_item','weight_weekday',
                  'forecast_total','yhat_item','unit_sales']
    """
    # --- Preprocessing ---
    sales_history = sales_history.copy()
    sales_history['date'] = pd.to_datetime(sales_history['date'])
    sales_history['week'] = sales_history['date'].dt.to_period('W').apply(lambda x: x.start_time)
    sales_history['weekday'] = sales_history['date'].dt.weekday

    train = train.copy()
    train['date'] = pd.to_datetime(train['date'])
    train['week'] = train['date'].dt.to_period('W').apply(lambda x: x.start_time)

    test = test.copy()
    test['date'] = pd.to_datetime(test['date'])
    test['week'] = test['date'].dt.to_period('W').apply(lambda x: x.start_time)

    # --- Prophet default settings ---
    default_kwargs = {
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "interval_width": 0.8,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0
    }
    if prophet_kwargs is None:
        prophet_kwargs = default_kwargs
    else:
        prophet_kwargs = {**default_kwargs, **prophet_kwargs}

    # --- Forecast per store (volle weken) ---
    all_forecasts_list = []
    stores_to_forecast = test['store_nbr'].unique()

    for store in tqdm(stores_to_forecast, desc="Forecast per store (week)"):
        train_store = train[train['store_nbr']==store].copy()
        if train_store.empty:
            continue

        train_weekly = (
            train_store.groupby('week')['unit_sales'].sum()
            .reset_index()
            .rename(columns={'week':'ds','unit_sales':'y'})
        )

        test_weeks = sorted(test[test['store_nbr']==store]['week'].unique())
        if len(test_weeks)==0 or len(train_weekly)<2:
            continue

        # Alleen volle weken die in training bestaan
        valid_weeks = [w for w in test_weeks if w in train_weekly['ds'].values]
        forecast_list = []

        if valid_weeks:
            test_df = pd.DataFrame({'ds': pd.to_datetime(valid_weeks)})

            model = Prophet(**prophet_kwargs)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='semi_annual', period=182.5, fourier_order=3)

            model.fit(train_weekly[['ds','y']])
            forecast = model.predict(test_df[['ds']])

            weekly_forecast = pd.DataFrame({
                'store_nbr': store,
                'week': forecast['ds'],
                'forecast_total': forecast['yhat']
            })
            forecast_list.append(weekly_forecast)

        # --- Rolling average voor eerste/laatste week indien ontbrekend ---
        missing_weeks = set(test_weeks) - set(valid_weeks)
        for mw in missing_weeks:
            # Rolling average van laatste N weken vóór mw
            past_weeks = train_weekly[train_weekly['ds'] < mw].sort_values('ds').tail(rolling_weeks)
            if not past_weeks.empty:
                yhat = past_weeks['y'].mean()
            else:
                yhat = train_weekly['y'].mean()  # fallback
            forecast_list.append(pd.DataFrame({
                'store_nbr': store,
                'week': [mw],
                'forecast_total': [yhat]
            }))

        if forecast_list:
            all_forecasts_list.append(pd.concat(forecast_list, ignore_index=True))

    if not all_forecasts_list:
        return pd.DataFrame(columns=[
            'store_nbr','date','item_nbr','weight_item','weight_weekday',
            'forecast_total','yhat_item','unit_sales'
        ])

    forecast_week_all = pd.concat(all_forecasts_list, ignore_index=True)

    # --- Bereken gecombineerde gewichten item × weekdag ---
    recent_sales = sales_history.copy()

    item_week_totals = (
        recent_sales.groupby(['store_nbr','week','item_nbr'])['unit_sales'].sum().reset_index()
    )
    store_week_totals = (
        item_week_totals.groupby(['store_nbr','week'])['unit_sales'].sum().reset_index()
        .rename(columns={'unit_sales':'store_week_sales'})
    )
    item_week_totals = item_week_totals.merge(store_week_totals, on=['store_nbr','week'], how='left')
    item_week_totals['weight_item'] = item_week_totals['unit_sales'] / item_week_totals['store_week_sales']

    weekday_totals = (
        recent_sales.groupby(['store_nbr','week','weekday'])['unit_sales'].sum().reset_index()
    )
    weekday_totals['weight_weekday'] = weekday_totals.groupby(['store_nbr','week'])['unit_sales'].transform(lambda x: x/x.sum())

    combined_weights = item_week_totals.merge(weekday_totals, on=['store_nbr','week'], how='outer')
    combined_weights['combined_weight'] = combined_weights['weight_item'] * combined_weights['weight_weekday']

    # --- Verdeel forecast over items × weekdagen ---
    df_forecast = forecast_week_all.merge(combined_weights, on=['store_nbr','week'], how='left')
    df_forecast['date'] = df_forecast['week'] + pd.to_timedelta(df_forecast['weekday'], unit='d')

    # --- Filter op testsetdagen ---
    start_date = test['date'].min()
    end_date = test['date'].max()
    df_forecast = df_forecast[(df_forecast['date'] >= start_date) & (df_forecast['date'] <= end_date)].copy()

    # --- Her-normaliseer gewichten per week ---
    df_forecast['combined_weight_norm'] = (
        df_forecast.groupby(['store_nbr','week'])['combined_weight'].transform(lambda x: x / x.sum())
    )
    df_forecast['yhat_item'] = df_forecast['forecast_total'] * df_forecast['combined_weight_norm']

    # Voeg daadwerkelijke sales toe
    df_forecast = df_forecast.merge(
        sales_history[['store_nbr','date','item_nbr','unit_sales']],
        on=['store_nbr','date','item_nbr'],
        how='left'
    )

    return df_forecast[[
        'store_nbr','date','item_nbr','weight_item','weight_weekday',
        'forecast_total','yhat_item','unit_sales'
    ]].sort_values(['store_nbr','date','item_nbr']).reset_index(drop=True)
