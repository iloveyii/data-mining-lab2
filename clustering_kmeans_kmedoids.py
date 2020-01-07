
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# for medoids
# from sklearn_extra.cluster import KMedoids # no way to install
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import read_sample
from pyclustering.cluster import cluster_visualizer
import time

# %matplotlib inline

k = 2
colors = ['green', 'blue', 'orange', 'red', 'purple', 'yellow', 'cyan', 'black', 'pink', 'sky']

df = pd.read_csv('rawdata200K.dat', delim_whitespace=True)
df.columns = ["X", "Y"]
df.isnull().sum()


plt.scatter(df.X, df.Y)
plt.title('Scatter plot of X, Y')
plt.xlabel('X')
plt.ylabel('Y')

plt.savefig('images/scatter-plot.png', dpi=300, bbox_inches='tight')
plt.show()


# Find kMeans
# For capturing the execution time 
# Scale data
scaler = MinMaxScaler()
df.X = scaler.fit(df[['X']]).transform(df[['X']])
df.Y = scaler.fit(df[['Y']]).transform(df[['Y']])

start = time.time()
k_means = KMeans(n_clusters=k)
cluster = k_means.fit_predict(df[['X', 'Y']])
df['cluster'] = cluster
end = time.time()
print(end-start)

# Visualize cluster
dfs = []
for i in range(0,k):
    dfs.append(df[df.cluster==i])

# Plot it
for i in range(0,k):
    plt.scatter(dfs[i].X, dfs[i].Y, color=colors[i], marker='*', label='cluster ' + str(i))
    
plt.title('Scatter plot of scaled X, Y using K Means with K='+str(k))
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.savefig('images/scatter-plot-kmeans-scaled-k-'+str(k)+'.png', dpi=300, bbox_inches='tight')
plt.show()


# Elbow plot method - find best k value
# For capturing the execution time 
start = time.time()
k_list = range(1, 10)
sse = []

for k_i in k_list:
    k_means = KMeans(n_clusters=k_i)
    k_means.fit(df[['X', 'Y']])
    sse.append(k_means.inertia_)
    
end= time.time()
print(end - start)


plt.title('Elbow plot to find best K based on SSE')

plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_list, sse)

plt.savefig('images/elbow-kmeans-scaled.png', dpi=300, bbox_inches='tight')
plt.show()

# K Medoids from pyclustering
start = time.time()

sample = df.values.tolist() # read_sample('./rawdata10K.dat')

# find clusteroids
clusteroids = []
for i in range(0,k):
    clusteroids.append(i)
    
    
k_medoids_instance = kmedoids(sample, clusteroids)
k_medoids_instance.process()
clusters = k_medoids_instance.get_clusters()
medoids = k_medoids_instance.get_medoids()
end = time.time()

print(end-start)

print(medoids)

# Display clusters.
visualizer = cluster_visualizer()
visualizer.set_canvas_title('KMedoids clustering with K='+str(k))
visualizer.append_clusters(clusters, sample)
visualizer.show()



