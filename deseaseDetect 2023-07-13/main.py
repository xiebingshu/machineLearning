from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33, test_size=0.2)


lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)
print(r2_score(y_test, y_predict))