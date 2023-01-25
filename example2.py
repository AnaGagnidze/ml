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
# print(wine.isnull().any())
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

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

data = pd.DataFrame(pd.read_html("https://github.com/kb22/Heart-Disease-Prediction/blob/master/dataset.csv")[0])
data.drop("Unnamed: 0", axis=1, inplace=True)

# amovarchiot 4 sauketeso sveti, feature selection
y = data['target']
X = data.drop('target', axis=1)
# selector = SelectKBest(score_func=f_classif, k=4)
# selector.fit(X, y)
# print(selector.get_feature_names_out())

pipe = Pipeline(steps=[('selector', SelectKBest(score_func=f_classif)), ('algo', AdaBoostClassifier())])
parameters = {"selector__k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}
hybrid = GridSearchCV(pipe, parameters, scoring='accuracy', cv=2, n_jobs=-1)
hybrid.fit(X, y)
print(hybrid.best_params_)
print(hybrid.best_score_)
