import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster import hierarchy

df = pd.read_csv("/home/jadno/Mall_Customers.csv", sep=",")
df.head()

df['Gender'].replace(['Male','Female'],[0,1],inplace=True)
df.head()

df.shape
df.describe()

sns.pairplot(df)

df.values

plt.figure(figsize=(50,50), dpi=500)

# Coeficiente de correlação em um heatmap
corr = np.corrcoef(df.values, rowvar=False)
sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', cbar=False, xticklabels=df.columns, yticklabels=df.columns)


# Padronização dos dados
df_scaled = df.copy()
df_scaled.iloc[:,:] = StandardScaler().fit_transform(df)

df_scaled.values

# Agrupamento hierarquico
Z = hierarchy.linkage(df_scaled, 'ward')
plt.grid(axis='y')

df.index

# Plot do dendrograma
dn = hierarchy.dendrogram(Z, labels=list(df.index))
plt.savefig('dendrograma.png', format='png', bbox_inches='tight')

numero_clusters = 4
cluster = AgglomerativeClustering(n_clusters=numero_clusters, affinity='euclidean', linkage='ward')
clusters = cluster.fit_predict(df)


costumers_group = {i: [] for i in range(numero_clusters)}

for costumer, cluster in zip(df, clusters):
    costumers_group[cluster for cluster in clusters].append(costumer: for costumer in df)

for cluster, costumer in costumers_group.items():
    print(f'Cluster {cluster}: {costumer}\n')






























