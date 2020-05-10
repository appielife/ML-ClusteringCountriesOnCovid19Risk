import pandas as pd
import numpy as np 
import os
import plotly.subplots as tls
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

INPUT_FILE = "./dataSource_emma/features_combined.csv"

## Add feature names that you want to filter out
NULL_FEATURES = ['country','Country_Region','entity','total_covid_19_tests']

## Add Features that you want to load from features_combined.csv
FILTER_FEATURES = ['Country_Region', 'total_covid_19_tests','Confirmed', 'pop2020','inform_risk', 'inform_p2p_hazard_and_exposure_dimension',
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

# SCALE DATA
scaler = StandardScaler()
data = data_tmp.drop(columns=["Country_Region","pop2020", "Confirmed", "total_covid_19_tests"])
print("DATA FOR CLUSTERING\n", data.tail(10))
print("\nfeatures:", data.columns)
data_k = scaler.fit_transform(data)
# print("\nSCALED DATA\n", data_k)


from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
mi = mutual_info_regression(data.drop(columns=['confirmed_ratio']), data['confirmed_ratio'] )
mi = pd.Series(mi)
mi.index = data.drop(columns=['confirmed_ratio']).columns
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))
plt.title("Factor impacting COVID-19 confirmed cases")
plt.show()



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


for group in range(0,5):
    countries=df_k.loc[df_k['cluster']==group]
    listofcoutries= list(countries['country_region'])
    print("Group", group, ":", listofcoutries, "\n")

# Plot cluster visualization
plt.figure(figsize=(10, 8))
plt.scatter(df_k['current_health_expenditure_per_capita'], df_k["confirmed_ratio"],c=pred, cmap='rainbow')
plt.title('Covid Clustering')
plt.xlabel("Current Health Expenditure Per Capita")
plt.ylabel("No. of confirmed cases")
plt.show()


