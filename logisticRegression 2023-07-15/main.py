from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33, test_size=0.3)
breast_cancer_ss = StandardScaler()
x_train = breast_cancer_ss.fit_transform(x_train)
x_test = breast_cancer_ss.transform(x_test)
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
print('Accuracy:', lr.score(x_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['benign', 'malignant']))


