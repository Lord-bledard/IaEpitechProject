import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the Olympic Data dataset
olympic_data = pd.read_csv("dataset_olympics.csv")

# Display basic information
print(olympic_data.info())

# Display the first few rows of the dataset
print(olympic_data.head())

# General Analysis
stats = olympic_data.describe()
missing_values = olympic_data.isnull().sum()

# Distribution of medals by country
medals_by_country = olympic_data.groupby('Team')['Medal'].count().sort_values(ascending=False).head(10)
sns.barplot(x=medals_by_country.index, y=medals_by_country.values, palette="Set2")
plt.title("Top 10 Countries by Medal Count")
plt.xlabel("Team")
plt.ylabel("Number of Medals")
plt.xticks(rotation=45)
plt.show()


# Preprocessing
imputer = SimpleImputer(strategy="most_frequent")
olympic_data_imputed = pd.DataFrame(imputer.fit_transform(olympic_data), columns=olympic_data.columns)
label_encoder = LabelEncoder()
olympic_data_imputed['Sex'] = label_encoder.fit_transform(olympic_data_imputed['Sex'])

# Select relevant columns for further processing
team_year_medals_data = olympic_data_imputed[['Team', 'Year', 'Medal']]

# Group by Team and Year, count the number of medals
team_year_medals_count = team_year_medals_data.groupby(['Team', 'Year'])['Medal'].count().reset_index()

# Visualize the distribution of medals over the years for the top N teams
top_teams = team_year_medals_count.groupby('Team')['Medal'].sum().sort_values(ascending=False).head(5).index

# Filter data for the top teams
top_teams_data = team_year_medals_count[team_year_medals_count['Team'].isin(top_teams)]

# Plot the distribution of medals over the years for the top teams
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Medal', hue='Team', data=top_teams_data, palette="Set1", marker='o')
plt.title("Distribution of Medals Over the Years for Top Teams")
plt.xlabel("Year")
plt.ylabel("Number of Medals")
plt.legend(title='Team', loc='upper left')
plt.show()

# Clustering or Dimensionality Reduction
features_for_processing = olympic_data_imputed.drop(['Team', 'Year', 'Medal'], axis=1)

# Apply K-means clustering
numeric_data = features_for_processing.select_dtypes(include='number')
kmeans = KMeans(n_clusters=3, random_state=42)
olympic_data_imputed['Cluster'] = kmeans.fit_predict(numeric_data)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(numeric_data)
olympic_data_imputed['PCA1'] = principal_components[:, 0]
olympic_data_imputed['PCA2'] = principal_components[:, 1]

# Visualize clusters
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=olympic_data_imputed, palette="viridis")
plt.title("K-means Clustering: PCA Components")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()