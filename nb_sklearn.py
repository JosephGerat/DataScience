
import numpy as np
from sklearn.naive_bayes import BernoulliNB

X = np.random.randint(2, size=(5, 3))
Y = np.random.randint(2, size=(5, 1))

X[0] = [1, 0, 0]
X[1] = [1, 0, 1]
X[2] = [0, 1, 1]
X[3] = [0, 1, 0]
X[4] = [1, 1, 1]
Y[0] = 0
Y[1] = 1
Y[2] = 0
Y[3] = 1
Y[4] = 1

clf = BernoulliNB()
model = clf.fit(X, Y)

x_test = np.random.randint(2, size=(1, 3))
x_test[0] = [1, 1, 0]
y_pred = clf.predict_proba(x_test)
print(y_pred)