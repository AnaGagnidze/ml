import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=2000, n_features=2, centers=[[1, 5], [8, 10]], shuffle=True, random_state=1)

print(X)
# რამდენი განზომილება (კლასტერის წერტილი იყო)
myKmeans = KMeans(n_clusters=2, max_iter=2000)
myKmeans.fit(X)  # X-ზე დაყრდნობით ითვლის ცენტრებს
y_predicted = myKmeans.predict(X)
# თითოეულ კლასტერს გააფერადებს სხვადასხვა ფრად
plt.scatter(X[:, 0], X[:, 1], c=y_predicted)
plt.show()

#################################################

# cluster std ერთმანეთში რევს წერტილებს
X, _ = make_blobs(n_samples=2000, n_features=2, centers=[[1, 5], [8, 10]], shuffle=True, random_state=1,cluster_std=2)
myKmeans = KMeans(n_clusters=2, max_iter=2000)
myKmeans.fit(X)
# გამოყოფს კლასტერის წერტილს
centers = myKmeans.cluster_centers_
y_predicted = myKmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_predicted)
plt.scatter(centers[:,0], centers[:,1], s=100, c='red')
plt.show()






wine = pd.read_csv(
    "https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv")
# print(wine.head())
wine.drop("Wine", axis=1, inplace=True)
print(wine.head())
# ყველა სვეტი გადადის სტანდარტულ ფორმატში
myscaler = StandardScaler()
wine = myscaler.fit_transform(wine)

# დადის 2 განზომილებაზე
mytsne = TSNE(n_components=2, perplexity=40, n_iter=2000)
wine_embedding = mytsne.fit_transform(wine)
plt.scatter(wine_embedding[:,0], wine_embedding[0:,1])
plt.show()