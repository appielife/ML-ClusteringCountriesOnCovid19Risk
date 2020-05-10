import pandas as pd
import numpy as np 
import os
import plotly.subplots as tls
import plotly.graph_objs as go
import plotly.express as px
import plotly
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

INPUT_FILE = "./dataSource_emma/features_combined.csv"

## Add feature names that you want to filter out
NULL_FEATURES = ['country','Country_Region','entity','total_covid_19_tests']

## Add Features that you want to load from features_combined.csv
FILTER_FEATURES = ['Country_Region', 'total_covid_19_tests','Confirmed', 'pop2020',
       'HDI Rank (2018)','inform_risk', 'inform_p2p_hazard_and_exposure_dimension',
       'population_density', 'population_living_in_urban_areas',
       'proportion_of_population_with_basic_handwashing_facilities_on_premises',
       'people_using_at_least_basic_sanitation_services',
       'inform_vulnerability', 'inform_health_conditions',
       'inform_epidemic_vulnerability', 'mortality_rate_under_5',
       'prevalence_of_undernourishment', 'inform_lack_of_coping_capacity',
       'inform_access_to_healthcare', 'current_health_expenditure_per_capita',
       'maternal_mortality_ratio']

def plot_clusters(df_k, title):
    colorscale = [[0, 'blue'], [0.25, 'green'], [0.5, 'yellow'], [0.75, 'orange'], [1, 'red']]

    data = [dict(type='choropleth',
                 locations=df_k['country_region'].astype(str),
                 z=df_k['cluster'].astype(int),
                 locationmode='country names',
                 colorscale=colorscale)]

    fig = dict(data=data,
               layout_title_text="<b>" + title + "</b>")
    plotly.offline.plot(fig)

def plot_multiple_maps(df_list, title = None):
    ## plot result
    _colorscale = [[0, 'blue'], [0.25, 'green'], [0.5, 'yellow'], [0.75, 'orange'], [1, 'red']]
    ROW, COL = 3, 1
    if not title: title = 'UnScale vs Scale vs Scale With Top K'
    # 2 * 2 subplots
    #fig = tls.make_subplots(rows=2, cols=2, column_widths=[0.5, 0.5], row_heights=[0.5, 0.5],
    #                        specs=[[{"type": "choropleth", "rowspan": 2}, {"type": "choropleth", "rowspan": 2}],[None, None]])

    # 3 * 1 subplots
    fig = tls.make_subplots(rows=ROW, cols=COL, column_widths=[1], row_heights=[0.33, 0.33, 0.33],
                            specs=[[{"type": "choropleth"}], [{"type": "choropleth"}], [{"type": "choropleth"}]])

    for r in range(ROW):
        for c in range(COL):
            _df = df_list[c*ROW+r]
            fig.add_trace(
                go.Choropleth(type='choropleth',
                              locations=_df['country_region'].astype(str),
                              z=_df['cluster'].astype(int),
                              locationmode='country names',
                              showscale=True, colorscale=_colorscale,
                              colorbar=dict(
                                  title="Cluster Index",
                                  yanchor="top", x=-0.2, y=1,
                                  ticks="outside", ticksuffix="(num)",
                              ), ),
                row=r+1, col=c+1
            )

    fig.update_layout(
        title= title,
        autosize=True,
        width=1400,
        height=900,
    )

    fig.show()

dataset_feature = pd.read_csv(INPUT_FILE)

not_null_features_df = dataset_feature[dataset_feature[NULL_FEATURES].notnull().all(1)]
not_zero_total_tests_df = not_null_features_df.loc[not_null_features_df['total_covid_19_tests']!=0]
dataset_features_by_country = not_zero_total_tests_df[FILTER_FEATURES]
dataset_features_by_country.fillna(0)

dataset_features_by_country.loc[dataset_features_by_country.Country_Region=='US','Country_Region']='United States of America'

data_tmp = dataset_features_by_country.sort_values(by=["Country_Region"])
data_tmp = data_tmp.reset_index(drop=True)

data_tmp['pop2020'] = data_tmp['pop2020'].apply(lambda x: x*1000)
data_tmp["confirmed_ratio"] = data_tmp["Confirmed"]/data_tmp["pop2020"]
data_tmp["confirmed_ratio"] = data_tmp["confirmed_ratio"].apply(lambda x: x*1000)

data_tmp["test_ratio"] = data_tmp["total_covid_19_tests"]/data_tmp["pop2020"]
data_tmp["test_ratio"] = data_tmp["test_ratio"].apply(lambda x: x*1000)
data_tmp=data_tmp.replace("No data", 0)
data_tmp=data_tmp.replace(np.inf, 0)
data_tmp=data_tmp.replace(np.nan, 0)
data_tmp=data_tmp.replace('x', 0)

print(data_tmp)


data = data_tmp.drop(columns=["Country_Region","pop2020", "Confirmed", "total_covid_19_tests"])
print("DATA FOR CLUSTERING\n", data.tail(10))
print("\nfeatures:", data.columns)

#------------------------------------------------------------------------------------------
# CLUSTER WITH UNSCALED DATA
#------------------------------------------------------------------------------------------
data_unscaled = data_tmp.drop(columns=["Country_Region","pop2020", "Confirmed", "total_covid_19_tests"])

#Plot WCSS to find the best number of clusters to use
wcss=[]
for i in range(1,15):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(data_unscaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15), wcss)
plt.xlabel("No. of clusters")
plt.ylabel(" Within Cluster Sum of Squares")
plt.show()

# Find the factor that impacts the confirmed ratio the most to visualize the clusters
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
mi = mutual_info_regression(data.drop(columns=['confirmed_ratio']), data['confirmed_ratio'] )
mi = pd.Series(mi)
mi.index = data.drop(columns=['confirmed_ratio']).columns
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))
plt.title("Factor impacting COVID-19 confirmed cases ratio (UNSCALED)")
plt.show()

# Cluster without scaling
kmeans = KMeans(n_clusters = 5, init='k-means++')
kmeans.fit(data_unscaled)
pred = kmeans.predict(data_unscaled)

data_unscaled["country_region"]=data_tmp["Country_Region"]
data_unscaled['cluster'] = pred
print("\nDATAFRAME WITHOUT SCALING")
print(data_unscaled.tail(30))
print("\nCluster counts:")
print(data_unscaled['cluster'].value_counts())

### #01 Emma - call plot_clusters function to plot clusters with unscaled_data
plot_clusters(data_unscaled, title= "Clusters With UnScaled Data Based On All Factors")

print("\nCLUSTERS WITHOUT SCALING")
for group in range(0,5):
    countries=data_unscaled.loc[data_unscaled['cluster']==group]
    listofcoutries= list(countries['country_region'])
    print("Group", group, ":", listofcoutries, "\n-------------------")

# Plot cluster visualization
plt.figure(figsize=(10, 8))
plt.scatter(data_unscaled['current_health_expenditure_per_capita'], data_unscaled["confirmed_ratio"],c=pred, cmap='rainbow')

plt.title('Covid Clustering for UNSCALED DATA')
plt.xlabel("Current Health Expenditure Per Capita")
plt.ylabel("No. of confirmed cases")
plt.show()

cluster_avgs = pd.DataFrame(round(data_unscaled.groupby('cluster').mean(),1))
print("\nCLUSTER UNSCALED AVERAGES\n", cluster_avgs)

#------------------------------------------------------------------------------------------
# CLUSTER WITH SCALED DATA
#------------------------------------------------------------------------------------------
scaler = StandardScaler()
print("DATA FOR CLUSTERING\n", data.tail(10))
data_k = scaler.fit_transform(data)

#Plot WCSS to find the best number of clusters to use
wcss=[]
for i in range(1,15):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(data_k)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15), wcss)
plt.xlabel("No. of clusters")
plt.ylabel(" Within Cluster Sum of Squares")
plt.show()

data_scaled = pd.DataFrame(data_k)
data_scaled.columns = data.columns
# Find the factor that impacts the confirmed ratio the most to visualize the clusters
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
mi = mutual_info_regression(data_scaled.drop(columns=['confirmed_ratio']), data_scaled['confirmed_ratio'] )
mi = pd.Series(mi)
mi.index = data_scaled.drop(columns=['confirmed_ratio']).columns
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))
plt.title("Factor impacting COVID-19 confirmed cases ratio (SCALED)")
plt.show()

# Cluster based on ALL features
kmeans = KMeans(n_clusters = 5, init='k-means++')
kmeans.fit(data_k)
pred = kmeans.predict(data_k)

# Convert scaled matrix back into dataframe and add in column names
df_k = pd.DataFrame(data_k)
df_k.columns = data.columns
df_k["country_region"]=data_tmp["Country_Region"]
df_k['cluster'] = pred
print("\nDATAFRAME K")
print(df_k.tail(30))
print("\nCluster counts:")
print(df_k['cluster'].value_counts())

print("\nCLUSTERS WITH SCALING")
for group in range(0,5):
    countries=df_k.loc[df_k['cluster']==group]
    listofcoutries= list(countries['country_region'])
    print("Group", group, ":", listofcoutries ,"\n-------------------")

### #02 Emma - call plot_clusters function to plot clusters with scaled_data
plot_clusters(df_k, title= "Clusters With Scaled Data Based On All Factors")


# Plot cluster visualization
plt.figure(figsize=(10, 8))
plt.scatter(df_k['current_health_expenditure_per_capita'], df_k["confirmed_ratio"],c=pred, cmap='rainbow')
plt.title('Covid Clustering for SCALED DATA')
plt.xlabel("Current Health Expenditure Per Capita")
plt.ylabel("No. of confirmed cases")
plt.show()

# Find cluster averages 
df_cluster = df_k[['cluster', 'confirmed_ratio', 'current_health_expenditure_per_capita', 'test_ratio', 'inform_risk', 'HDI Rank (2018)', 'mortality_rate_under_5' ]]
cluster_avgs = pd.DataFrame(round(df_cluster.groupby('cluster').mean(),1))
print("\nCLUSTER AVERAGES\n", cluster_avgs)

#------------------------------------------------------------------------------------------
# CLUSTER WITH TOP FACTORS & SCALED DATA
#------------------------------------------------------------------------------------------
df_top = df_cluster.drop(columns=['cluster'])
kmeans = KMeans(n_clusters = 5, init='k-means++')
kmeans.fit(df_top)
pred = kmeans.predict(df_top)

df_top["country_region"]=data_tmp["Country_Region"]
df_top['cluster'] = pred
print("\nDATAFRAME TOP")
print(df_top.tail(30))
print("\nCluster counts:")
print(df_top['cluster'].value_counts())


for group in range(0,5):
    countries=df_top.loc[df_top['cluster']==group]
    listofcoutries= list(countries['country_region'])
    print("Group", group, ":", listofcoutries, "\n")

### #03 Emma - call plot_clusters function to plot clusters with top k data
plot_clusters(df_top, title= "5 Clusters Based On Top 5 Factors")

# Plot cluster visualization
plt.figure(figsize=(10, 8))
plt.scatter(df_top['current_health_expenditure_per_capita'], df_top["confirmed_ratio"],c=pred, cmap='rainbow')

plt.title('Covid Clustering for TOP FACTORS AND SCALED DATA')
plt.xlabel("Current Health Expenditure Per Capita")
plt.ylabel("No. of confirmed cases")
plt.show()

cluster_avgs = pd.DataFrame(round(df_top.groupby('cluster').mean(),1))
print("\nCLUSTER TOP AVERAGES\n", cluster_avgs)


### #04 Emma - call plot_multiple_maps function
plot_multiple_maps([data_unscaled,df_k, df_top], title= None)