import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("", sep=';')

y = data[''].values
X = data.drop('', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)


mymodel = LogisticRegression(max_iter=100000, n_jobs=-1, C=0.001)
#mymodel = GaussianNB()
#mymodel = MultinomialNB()
#mymodel = Lasso()

mymodel.fit(X, y)
print(mymodel.score(X, y))



hybrid = Pipeline(steps=[(("scaler"), StandardScaler()), ("pca", PCA(n_components=3)), ('algo', Lasso())])
hybrid.fit(X, y)
print(hybrid.score(X, y))
# print(hybrid.named_steps['pca'].explained_variance_ratio_[0])
print(np.sum(hybrid.named_steps['pca'].explained_variance_ratio_[0]))