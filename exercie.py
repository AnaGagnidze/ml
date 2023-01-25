import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# data = pd.read_csv("emails.csv")
# y = data["Prediction"].values
# X = data.drop("Prediction", axis=1).values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# myModel = LogisticRegression(max_iter=1000)
# myModel.fit(X_train, y_train)
# print(myModel.score(X_train, y_train))
# print(myModel.score(X_test, y_test))
#
# best = SelectKBest(score_func=f_classif, k=4).fit_transform(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# myModel = SVC()
# myModel.fit(X_train, y_train)
# print(myModel.score(X_train, y_train))
# print(myModel.score(X_test, y_test))


data = pd.read_csv("regression.csv")
y = data["y"]
X = data["x"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
myModel = Lasso()
myModel.fit(X_train, y_train)
print(myModel.score(X_train, y_train))
print(myModel.score(X_test, y_test))


