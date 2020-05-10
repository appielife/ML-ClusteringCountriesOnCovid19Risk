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


# GMM
from sklearn.mixture import GaussianMixture as GMM
##

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
dataMap=data_tmp


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
# plt.plot(range(1,15), wcss)
# plt.xlabel("No. of clusters")
# plt.ylabel(" Within Cluster Sum of Squares")
# plt.show()

# Find the factor that impacts the confirmed ratio the most to visualize the clusters
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
mi = mutual_info_regression(data.drop(columns=['confirmed_ratio']), data['confirmed_ratio'] )
mi = pd.Series(mi)
mi.index = data.drop(columns=['confirmed_ratio']).columns
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))
plt.title("Factor impacting COVID-19 confirmed cases ratio (UNSCALED) GMM")
plt.show()

# Cluster without scaling



# kmeans = KMeans(n_clusters = 5, init='k-means++')
# kmeans.fit(data_unscaled)
# pred = kmeans.predict(data_unscaled)


gmm = GMM(n_components=5).fit(data_unscaled)
pred = gmm.predict(data_unscaled)
# plt.scatter(data_k.iloc[:,0], data_k.iloc[:,1], c=pred, s=40, cmap='viridis')
# plt.title('Covid Clustering using GMM')
# plt.xlabel("Test Perfomed by the country")
# plt.ylabel("No. of confirmed cases")
# plt.show()

# 

data_unscaled["country_region"]=data_tmp["Country_Region"]
data_unscaled['cluster'] = pred
print("\nDATAFRAME WITHOUT SCALING")
print(data_unscaled.tail(30))
print("\nCluster counts:")
print(data_unscaled['cluster'].value_counts())

print("\nCLUSTERS WITHOUT SCALING")
for group in range(0,5):
    countries=data_unscaled.loc[data_unscaled['cluster']==group]
    listofcoutries= list(countries['country_region'])
    print("Group", group, ":", listofcoutries, "\n-------------------")

# Plot cluster visualization
plt.figure(figsize=(10, 8))
plt.scatter(data_unscaled['current_health_expenditure_per_capita'], data_unscaled["confirmed_ratio"],c=pred, cmap='rainbow')
plt.title('Covid Clustering (GMM)')
plt.xlabel("Current Health Expenditure Per Capita")
plt.ylabel("No. of confirmed cases")
plt.show()





data_map = [dict(type='choropleth',
             locations = dataMap['Country_Region'].astype(str),
             z=dataMap['Confirmed'].astype(int),
             locationmode='country names')]
fig = dict(data=data_map, 
           layout_title_text="GMM COVID-19 Confirmed cases")
plotly.offline.plot(fig)

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
# for i in range(1,15):
#     kmeans = KMeans(n_clusters=i, init='k-means++')
#     kmeans.fit(data_k)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1,15), wcss)
# plt.xlabel("No. of clusters")
# plt.ylabel(" Within Cluster Sum of Squares")
# plt.show()

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
plt.title("Factor impacting COVID-19 confirmed cases ratio (SCALED) GMM")
plt.show()

# Cluster based on ALL features
# kmeans = KMeans(n_clusters = 5, init='k-means++')
# kmeans.fit(data_k)
# pred = kmeans.predict(data_k)

# Cluster based on ALL features
#GMM
gmm = GMM(n_components=5).fit(data_k)
pred = gmm.predict(data_k)




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

# Plot cluster visualization
plt.figure(figsize=(10, 8))
plt.scatter(df_k['current_health_expenditure_per_capita'], df_k["confirmed_ratio"],c=pred, cmap='rainbow')
plt.title('Covid Clustering GMM')
plt.xlabel("Current Health Expenditure Per Capita")
plt.ylabel("No. of confirmed cases")
plt.show()


data = [dict(type='choropleth',
             colorscale='reds',
             locations =df_k['country_region'].astype(str),
             z= df_k['cluster'].astype(int),
             locationmode='country names')]
# Plotting the groups of countries on world map¶

fig = dict(data=data, 
           layout_title_text="Country grouped based on GMM Health care quality, no. of COVID-19 cases and tests performed")

plotly.offline.plot(fig)


# Find cluster averages 
df_cluster = df_k[['cluster', 'confirmed_ratio', 'current_health_expenditure_per_capita', 'test_ratio', 'inform_risk', 'HDI Rank (2018)', 'mortality_rate_under_5' ]]
cluster_avgs = pd.DataFrame(round(df_cluster.groupby('cluster').mean(),1))
print("\nCLUSTER AVERAGES\n", cluster_avgs)

