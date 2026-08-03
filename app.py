import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.api.types import CategoricalDtype
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


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
    .groupby(['Month', 'Activiteitstype'], observed=True, as_index=False).agg({'Beweegtijd': 'sum',
                                                                               'Afstand.1':'sum'}))

df_monthly['Month'] = df_monthly['Month'].astype(month_cat)

df_monthly['Hours'] = df_monthly['Beweegtijd'] / 3600
df_monthly['Minutes'] = df_monthly['Beweegtijd'] / 60


# Set as categorical
df_monthly["Activity"] = (
    df_monthly["Activiteitstype"]
    .astype(str)
    .str.strip()
)
df_monthly['Activity']= df_monthly['Activity'].replace({
    'Fietsrit': 'Cycling',
    'Hardloopsessie': 'Running',
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



################################################
# Running pace calculations
df_run = df_monthly[df_monthly['Activity'] == 'Running'].copy()

# Average running speed (min/km)
df_run['Running pace numeric'] = (1000*df_run['Minutes']) / (df_run['Afstand.1']).replace(0, np.nan)

# Convert to min:sec/km safely
def min_to_mmss(x):
    if pd.isna(x) or not math.isfinite(x):
        return np.nan

    minutes = int(x)
    seconds = int(round((x - minutes) * 60))

    # handle rounding edge case (e.g. 4.999 → 5:00)
    if seconds == 60:
        minutes += 1
        seconds = 0

    return f"{minutes}:{seconds:02d}"


df_run['Running pace string'] = df_run['Running pace numeric'].apply(min_to_mmss)

st.subheader("Average Running pace by month")


fig, ax = plt.subplots(figsize=(10, 5))

sns.lineplot(
    data=df_run,
    x="Month",
    y="Running pace numeric",
    ax=ax,
    marker="o"
    )
ax.set_ylim(4.5, 5.5)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.set(xlabel="")

ax.set_ylabel("Pace (min/km)")
ax.set_xlabel("Month")

ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: min_to_mmss(x))
)

st.pyplot(fig)