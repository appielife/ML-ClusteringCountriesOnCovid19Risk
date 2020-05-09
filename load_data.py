import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly.subplots as tls
import plotly.graph_objs as go

INPUT_FILE = "./dataSource_emma/features_combined.csv"
NULL_FEATURES = ['country','Country_Region','entity','total_covid_19_tests']
FILTER_FEATURES = ['Country_Region', 'entity', 'total_covid_19_tests','Confirmed', 'Deaths', 'Recovered', 'Active', 'pop2020', 'Lat','Long_']
COUNTRY_FROM, COUNTRY_TO = [], []

dataset_feature = pd.read_csv(INPUT_FILE)
not_null_features_df = dataset_feature[dataset_feature[NULL_FEATURES].notnull().all(1)]
dataset_features_by_country = not_null_features_df[FILTER_FEATURES]
dataset_features_by_country.fillna(0)


fig = tls.make_subplots(rows=2, cols=2,
    column_widths=[0.5, 0.5],
    row_heights=[0.5, 0.5],
    specs=[
        [{"type": "choropleth", "rowspan": 2}, {"type": "choropleth", "rowspan": 2}],
        [None, None]
    ],
    subplot_titles=('123','','',''))


fig.add_trace(
    go.Choropleth(type='choropleth',
             locations = dataset_features_by_country['Country_Region'].astype(str),
             z=dataset_features_by_country['Confirmed'].astype(int),
             locationmode='country names',
             text='Confirmed_Case', colorscale='Reds', showscale=True, autocolorscale=True,
             colorbar=dict(
                 title="Confirmed Cases",
                 yanchor="top", x =-0.2, y=1,
                 ticks="outside", ticksuffix="(num)",
             ),),
    row=1, col=1
)

fig.add_trace(
    go.Choropleth(type='choropleth',
             locations = dataset_features_by_country['entity'].astype(str),
             z=dataset_features_by_country['total_covid_19_tests'].astype(int),
             locationmode='country names',  colorscale='Reds', showscale=True, autocolorscale=True,
             colorbar=dict(
                 title="Total Tests",
                 yanchor="top", x=1.05, y=1,
                 ticks="outside", ticksuffix="(num)",
            ), ),
    row=1, col=2
)

#fig.update_geos(projection_type="orthographic")
#fig.update_geos(projection_type="natural earth")

fig.update_layout(
    #template="plotly_dark",
    title = ' Machine Learning Final Project - COVID-19<br>\
             Confirmed Cases vs Total Test',
    autosize = True,
    width = 1400,
    height = 900,
    annotations=[
        dict(
            text="TEST 1234567",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=0)
    ],
)
#fig['layout']['scene1'].update(annotations=[dict(z=40, text='my title', showarrow=False)])
fig.show()

#choromap = go.Figure(data=[data1, data2, marker], layout=layout)
#choromap.show()
#plotly.offline.plot(choromap)

# cleaning the country names for joining
dataset_features_by_country.loc[dataset_features_by_country.Country_Region=='US','Country_Region']='United States of America'
dataset_features_by_country.loc[dataset_features_by_country.country=='Viet Nam','country']='Vietnam'
dataset_features_by_country.loc[dataset_features_by_country.country=='Russian Federation','country']='Russia'
dataset_features_by_country.loc[dataset_features_by_country.country=='Korea Republic of','country']='Korea, South'
dataset_features_by_country.loc[dataset_features_by_country.country=='Moldova Republic of','country']='Moldova'
dataset_features_by_country.loc[dataset_features_by_country.entity=='United States','entity']='United States of America'

#replace No Data with 0
#dataset_indicators=dataset_indicators.replace("No data", 0)

