from tarfile import PAX_FIELDS
import pandas as pd
import csv 
import plotly.express as px

df = pd.read_csv("stars.csv")

print(df.head())
fig = px.scatter(df, x = "Size", y = "Light")
fig.show()

from sklearn.cluster import KMeans

X = df.iloc[:, [0,1]].values
print(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = 1, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.lineplot(range(1,11), wcss, marker='o', color ='red')
plt.title("The Elbow Method")
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow')
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue')
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green')
plt.title('Clusters of Stars')
plt.xlabel('Size')
plt.ylabel('Light')
plt.legend()
plt.show()





