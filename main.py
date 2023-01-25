import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

data = pd.read_csv("titanic.csv")
data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1, inplace=True)
# String -> Number
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
# Fill Column (Mean() -> Avg, Median() -> median
data['Age'] = data['Age'].fillna(data['Age'].mean())
# X & Y
y = data['Survived'].values
x = data.drop('Survived', axis=1).values
# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
# Formatting data
my_scaler = StandardScaler()
x_train = my_scaler.fit_transform(x_train)
x_test = my_scaler.fit_transform(x_test)
print(data.head())

x_new = SelectKBest(score_func=f_classif, k=4).fit_transform(x, y)