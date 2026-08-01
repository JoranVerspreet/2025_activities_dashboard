import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.api.types import CategoricalDtype


st.title("2025 Activities Overview")



# Load data
df = pd.read_csv('data/activities.csv')

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
df_2025 = df[df['Year'] == 2025].copy()
df_2025['Month'] = df_2025['Date'].dt.month_name()

month_order = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
]
month_cat = CategoricalDtype(categories=month_order, ordered=True)
df_2025['Month'] = df_2025['Month'].astype(month_cat)

df_monthly = (
    df_2025
    .groupby(['Month', 'Activiteitstype'], observed=True, as_index=False)
    ['Beweegtijd']
    .sum()
)
df_monthly['Month'] = df_monthly['Month'].astype(month_cat)

df_monthly['Hours'] = df_monthly['Beweegtijd'] / 3600

from pandas.api.types import CategoricalDtype

# Create the correct order
month_order = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
]


# Set as categorical
df_monthly['Activity'] = df_monthly['Activiteitstype'].astype(str)
df_monthly["Activity"] = (
    df_monthly["Activiteitstype"]
    .astype(str)
    .str.strip()
)
df_monthly['Activity']= df_monthly['Activity'].replace({
    'Fietsen': 'Cycling',
    'Hardlopen': 'Running',
    'Wandelen': 'Walking',
    'Zwemmen': 'Swimming'
})

# Plotly chart
fig = px.bar(df_monthly, x="Month", y="Hours", color="Activity")

fig.update_xaxes(
    categoryorder="array",
    categoryarray=df_monthly['Month'].cat.categories.tolist()
)

fig.update_layout(
    xaxis=dict(
        rangeslider=dict(visible=True),
        tickangle=-45
    )
)

st.plotly_chart(fig, width="stretch")

