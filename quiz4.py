import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


data = pd.read_csv("credit.csv")
print(data.head())

y = data["y"].values
X = data.drop("y", axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, test_size=0.1)

mymodel = SVC()
mymodel.fit(X_train, y_train)
print(mymodel.score(X, y)) # პასუხი, გამოვიდა 0.6521739130434783, რაც ჩემი აზრით არც თუ ისე კარგი მაჩვენებელია

selector = SelectKBest(k=6)
print(selector.get_feature_names_out()) #['x4' 'x5' 'x6' 'x7' 'x8' 'x9']
X = data[['X4', 'X5', 'X6', 'X7', 'X8', 'X9']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, test_size=0.1)
mymodel = SVC()
mymodel.fit(X_train, y_train)
print(mymodel.score(X, y)) # პასუხი, გამოვიდა 0.8565217391304348, რამაც აჯობა წინა შედეგს
