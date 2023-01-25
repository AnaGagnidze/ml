import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/kvinlazy/Dataset/master/drug200.csv")
# print(data.head())
encoder = LabelEncoder()
data['Sex'] = encoder.fit_transform(data['Sex'])
data['BP'] = encoder.fit_transform(data['BP'])
data['Cholesterol'] = encoder.fit_transform(data['Cholesterol'])
y = data["Drug"].values
X = data.drop("Drug", axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, test_size=0.1)
my_model = AdaBoostClassifier()
my_model.fit(X_train, y_train)
print(my_model.score(X_train, y_train)) # მივიღე 0.84444444444, რაც მეტია
print(my_model.score(X_test, y_test)) # 0.8-ზე

params1 = {"learning_rate": [0.1, 0.2, 0.3, 0.4, 0.7, 0.9]}
hybrid1 = GridSearchCV(my_model, params1, scoring='accuracy', cv=3, n_jobs=-1)
hybrid1.fit(X, y)
print(hybrid1.best_score_) # best score- 0.8349917081260365

params2 = {"n_estimators": [30, 40, 60], "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.7, 0.9]}
hybrid2 = GridSearchCV(my_model, params2, scoring='accuracy', cv=3, n_jobs=-1)
hybrid2.fit(X, y)
print(hybrid2.best_params_) # 'learning_rate': 0.1, 'n_estimators': 30 საუკეთესო კომბინაციაა
