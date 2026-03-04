import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from pandas.api.types import CategoricalDtype


st.title("2025 Weekly Activity Dashboard")



# Load data
df = pd.read_csv('../data/activities.csv')

# Month translations
months = {
    'januari': 'January','februari': 'February','maart': 'March','april': 'April','mei': 'May',
    'juni': 'June','juli': 'July','augustus': 'August','september': 'September','oktober': 'October',
    'november': 'November','december': 'December','jan': 'January','feb': 'February','mrt': 'March',
    'apr': 'April','mei': 'May','jun': 'June','jul': 'July','aug': 'August','sep': 'September',
    'okt': 'October','nov': 'November','dec': 'December',
}

df['Datum van activiteit_en'] = df['Datum van activiteit'].str.lower()
for nl, en in months.items():
    df['Datum van activiteit_en'] = df['Datum van activiteit_en'].str.replace(nl, en, regex=False)

df['Datum van activiteit'] = pd.to_datetime(
    df['Datum van activiteit_en'],
    format='%d %B %Y, %H:%M:%S'
)
df.drop(columns='Datum van activiteit_en', inplace=True)
df = df[df['Afstand'] != 0]
df =df[df['Activiteitstype'] != 'Training']
df = df[df['Activiteitstype'] != 'Wandeling']

df['Date'] = df['Datum van activiteit']
df['Year'] = df['Datum van activiteit'].dt.year
df['Month'] = df['Datum van activiteit'].dt.month
df['Quarter'] = df['Datum van activiteit'].dt.quarter
df['Week'] = df['Datum van activiteit'].dt.isocalendar().week



# Filter 2025
df_2025 = df[df['Year'] == 2025]
df_2025['Month'] = df_2025['Date'].dt.month_name()

month_order = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
]
month_cat = CategoricalDtype(categories=month_order, ordered=True)
df_2025['Month'] = df_2025['Month'].astype(month_cat)

df_weekly = (
    df_2025
    .groupby(['Month', 'Week', 'Activiteitstype'], observed=True, as_index=False)
    ['Beweegtijd']
    .sum()
)

df_weekly['Hours'] = df_weekly['Beweegtijd'] / 3600
df_weekly['Month_Week'] = df_weekly['Month'].astype(str) + " W" + df_weekly['Week'].astype(str)
from pandas.api.types import CategoricalDtype

# Create the correct order
month_order = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
]

# Ensure weeks within months are ordered
ordered_weeks = []
for month in month_order:
    weeks_in_month = df_weekly[df_weekly['Month'] == month]['Week'].sort_values().unique()
    for week in weeks_in_month:
        ordered_weeks.append(f"{month} W{week}")

# Set as categorical
month_week_cat = CategoricalDtype(categories=ordered_weeks, ordered=True)
df_weekly['Month_Week'] = df_weekly['Month_Week'].astype(month_week_cat)


# Plotly chart
fig = px.bar(df_weekly, x="Month_Week", y="Hours", color="Activiteitstype")

fig.update_xaxes(
    categoryorder="array",
    categoryarray=df_weekly['Month_Week'].cat.categories.tolist()
)

fig.update_layout(
    xaxis=dict(
        rangeslider=dict(visible=True),
        tickangle=-45
    )
)

st.plotly_chart(fig, use_container_width=True)

