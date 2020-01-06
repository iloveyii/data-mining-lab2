from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import read_sample


df = pd.read_csv('rawdata1K.csv')
df.columns = ["Age", "Income"]
df.head()
df = (df[['Age', 'Income']]).head(100)
print(df)

sample = read_sample('./rawdata1K.dat')

k_medoids = kmedoids(sample, [12, 16, 17])
k_medoids.process()
clusters = k_medoids.get_clusters()
medoids = k_medoids.get_medoids()

print(medoids)

type(sample)


