
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from google.colab import drive
drive.mount('/content/drive')

creditcard_df = pd.read_csv('/content/drive/MyDrive/Bases de dados/Marketing_data.csv')

creditcard_df.shape

creditcard_df.head()

creditcard_df.info()

creditcard_df.describe()

creditcard_df[creditcard_df['ONEOFF_PURCHASES'] == 40761.250000]

creditcard_df['CASH_ADVANCE'].max()

creditcard_df[creditcard_df['CASH_ADVANCE'] == 47137.211760000006]

"""## Visualização e exploração dos dados"""

sns.heatmap(creditcard_df.isnull());

creditcard_df.isnull().sum()

creditcard_df['MINIMUM_PAYMENTS'].mean()

creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()

creditcard_df.loc()

creditcard_df['CREDIT_LIMIT'].mean()

creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = creditcard_df['CREDIT_LIMIT'].mean()

creditcard_df.isnull().sum()

sns.heatmap(creditcard_df.isnull());

creditcard_df.duplicated().sum()

creditcard_df.drop('CUST_ID', axis = 1, inplace = True)

creditcard_df.head()

creditcard_df.columns

len(creditcard_df.columns)

plt.figure(figsize=(10,50))
for i in range(len(creditcard_df.columns)):
  plt.subplot(17, 1, i + 1)
  sns.distplot(creditcard_df[creditcard_df.columns[i]], kde = True)
  plt.title(creditcard_df.columns[i])
plt.tight_layout();

correlations = creditcard_df.corr()

f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(correlations, annot=True);

"""## Definição do número de clusters usando o Elbow Method

"""

min(creditcard_df['BALANCE']), max(creditcard_df['BALANCE'])



scaler = StandardScaler()# USar standardScaler quando tiver muitos outliers(Transformada normal) MinMAxscaler é usado valores minino e maximos
creditcard_df_scaled = scaler.fit_transform(creditcard_df)

type(creditcard_df_scaled), type(creditcard_df)

min(creditcard_df_scaled[0]), max(creditcard_df_scaled[0])

creditcard_df_scaled

wcss_1 = []
range_values = range(1, 20)
for i in range_values:
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(creditcard_df_scaled)
  wcss_1.append(kmeans.inertia_)

print(wcss_1)

plt.plot(wcss_1, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('WCSS');

"""## Agrupamento com k-means"""

kmeans = KMeans(n_clusters=8)
kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_

labels, len(labels)

np.unique(labels, return_counts=True)

kmeans.cluster_centers_

cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [creditcard_df.columns])
cluster_centers

"""- Grupo 0 (VIP/Prime): limite do cartão alto (15570) e o mais alto percentual de pagamento da fatura completa (0.47). Aumentar o limite do cartão e o hábito de compras

- Grupo 3: Clientes que pagam poucos juros para o banco e são cuidadosos com seu dinheiro. Possui menos dinheiro na conta corrente (104) e não sacam muito dinheiro do limite do cartão (302). 23% de pagamento da fatura completa do cartão de crédito

- Grupo 5: usam o cartão de crédito como "empréstimo" (setor mais lucrativo para o banco), possuem muito dinheiro na conta corrente (5119) e sacam muito dinheiro do cartão de crédito (5246), compram pouco (0.3) e usam bastante o limite do cartão para saques (0.51). Pagam muito pouco a fatura completa (0.03)

- Grupo 7 (clientes novos): clientes mais novos (7.23) e que mantém pouco dinheiro na conta corrente (863) 
"""

cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard_df.columns])
cluster_centers

labels, len(labels)

creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster': labels})], axis = 1)
creditcard_df_cluster.head()

for i in creditcard_df.columns:
  plt.figure(figsize=(35,5))
  for j in range(8):
    plt.subplot(1, 8, j + 1)
    cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{} \nCluster {}'.format(i, j))
  plt.show()

credit_ordered = creditcard_df_cluster.sort_values(by = 'cluster')
credit_ordered.head()

credit_ordered.tail()

credit_ordered.to_csv('cluster.csv')

"""## Aplicação de PCA (principal component analysis) e visualização dos resultados"""

pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_df_scaled)
principal_comp

pca_df = pd.DataFrame(data = principal_comp, columns=['pca1', 'pca2'])
pca_df.head()

pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis = 1)
pca_df.head()

plt.figure(figsize=(10,10))
sns.scatterplot(x = 'pca1', y = 'pca2', hue = 'cluster', data = pca_df, palette = ['red', 'green', 'blue', 'pink', 'yellow', 'gray', 'purple', 'black'])

"""## Aplicação de autoencoders"""

# 18 -> 10
# Elbow
# K-means
# PCA

creditcard_df_scaled.shape

# 17 -> 500 -> 2000 -> 10 -> 2000 -> 500 -> 17
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input_df = Input(shape=(17,))
x = Dense(500, activation='relu')(input_df)
x = Dense(2000, activation='relu')(x)

encoded = Dense(10, activation='relu')(x)

x = Dense(2000, activation='relu')(encoded)
x = Dense(500, activation='relu')(x)

decoded = Dense(17)(x)

# autoencoder
autoencoder = Model(input_df, decoded)

# encoder
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer = 'Adam', loss = 'mean_squared_error')

autoencoder.fit(creditcard_df_scaled, creditcard_df_scaled, epochs = 50)

creditcard_df_scaled.shape

compact = encoder.predict(creditcard_df_scaled)

compact.shape

creditcard_df_scaled[0]

compact[0]

wcss_2 = []
range_values = range(1, 20)
for i in range_values:
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(compact)
  wcss_2.append(kmeans.inertia_)

plt.plot(wcss_2, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('WCSS');

plt.plot(wcss_1, 'bx-', color = 'r')
plt.plot(wcss_2, 'bx-', color = 'g');

kmeans = KMeans(n_clusters=4)
kmeans.fit(compact)

labels = kmeans.labels_
labels, labels.shape

df_cluster_at = pd.concat([creditcard_df, pd.DataFrame({'cluster': labels})], axis = 1)
df_cluster_at.head()

pca = PCA(n_components = 2)
prin_comp = pca.fit_transform(compact)
pca_df = pd.DataFrame(data = prin_comp, columns = ['pca1', 'pca2'])
pca_df.head()

pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis = 1)
pca_df.head()

plt.figure(figsize=(10,10))
sns.scatterplot(x = 'pca1', y = 'pca2', hue = 'cluster', data = pca_df, palette = ['red', 'green', 'blue', 'pink'])

df_cluster_ordered = df_cluster_at.sort_values(by = 'cluster')
df_cluster_ordered.head()

df_cluster_ordered.tail()

df_cluster_ordered.to_excel('cluster_orderededd.xls')