import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

heart = pd.read_csv("https://raw.githubusercontent.com/Ravjot03/Heart-Disease-Prediction/master/framingham.csv")
heart.dropna(axis=1, inplace=True)
y = heart["TenYearCHD"].values
X = heart.drop("TenYearCHD", axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
my_model = LogisticRegression(max_iter=100000, n_jobs=1)
my_model.fit(X_train, y_train)
# print(my_model.score(X_test, y_test))
# მივიღე 0.8471698113207548, ანუ დაახლოებით 85%, რაც ნორმალურია, მაგრამ არც ისე კარგი შედეგია,
# C=0.05 მივუთითე და მაინც იგივე შედეგი დააბრუნა.

my_gaussian_model = GaussianNB()
my_gaussian_model.fit(X_train, y_train)
# print(my_gaussian_model.score(X_test, y_test))
# მივიღე 0.8264150943396227, ანუ დაახლოებით 83%, რაც უფრო ცუდი შედეგია ვიდრე logistic regression-ით მიღებული.

X_new = heart[["male", "currentSmoker", "prevalentHyp"]].values
X_new_train, X_new_test, y_train, y_test = train_test_split(X_new, y, stratify=y, random_state=1)
my_model = LogisticRegression(max_iter=100000, n_jobs=1)
my_model.fit(X_new_train, y_train)
print(my_model.score(X_new_test, y_test))
my_gaussian_model = GaussianNB()
my_gaussian_model.fit(X_new_train, y_train)
print(my_gaussian_model.score(X_new_test, y_test))

# ორივე შემთხვევაში მივიღე 0.8481132075471698, ანუ დაახლოებით 85%, მაგრამ ყველაზე მაღალი შედეგები დაიდო წინებთან შედარებით.
# მომიწია education სვეტი გამომეტოვებინა, რადგან Na პარამეტრს შეიცავდა