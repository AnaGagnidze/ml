import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

encoder = LabelEncoder()
data['sex'] = encoder.fit_transform(data['sex'])
data['smoker'] = encoder.fit_transform(data['smoker'])
data['region'] = encoder.fit_transform(data['region'])

y = data['charges'].values
X = data.drop('charges', axis=1).values
my_model = LinearRegression()
my_model.fit(X, y)
print(my_model.score(X, y))

hybrid = Pipeline(steps=[("scaler", StandardScaler()), ("pca", PCA(n_components=2)), ("algo", Lasso())])
hybrid.fit(X, y)
print(hybrid.score(X, y))


"""
Linear regression-ით გამოთვლილი score არის 0.7507372027994937, ხოლო pipeline-ს გამოყენებით, სტეპების გაერთიანების შემდეგ
შედეგი არის 0.7505030016841143. ნორმალური შედეგია, თუმცა pipeline-ში PCA პარამეტრზე იყო დამოკიდებული შედეგის სიზუსტე, 
რაც უფრო მცირე რიცხვს ჩავწერდი n_component-ში, მით უფრო ცუდი შედეგი იძლეოდა (მაგალითად 2 სვეტი როცა მივუთითე შედეგი
მივიღე 0.27159473030763004 
"""