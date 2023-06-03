import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.svm import SVC


class classify(object):
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        print(self.X)
        print(self.y)
        self.acc = []

    def split(self, testpr = 0.3):
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(self.X, self.y, test_size=testpr)
        print(self.X_te, self.X_tr, self.y_tr, self.y_te)
        self.X_tr = np.array(self.X_tr, dtype='int32')
        self.X_te = np.array(self.X_te, dtype='int32')
        self.y_tr = np.array(self.y_tr, dtype='int32')
        self.y_te = np.array(self.y_te, dtype='int32')
        self.length = len(self.X_tr)

    def getbest(self):
        data = self.X
        data1 = data
        

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.X_tr, self.y_tr)
        yknn = knn.predict(self.X_te)
        print(yknn)
        knnacc = accuracy_score(self.y_te, yknn)
        self.acc.append(("knn", knnacc))


        clf = GaussianNB()
        clf.fit(self.X_tr, self.y_tr)
        xx = clf.predict(self.X_te)
        print(xx)
        gnb = accuracy_score(self.y_te, xx)
        self.acc.append(("GaussianNB", gnb))

        clf = linear_model.LinearRegression()
        clf.fit(self.X_tr, self.y_tr)
        lrgacc = clf.score(self.X_te, self.y_te)
        self.acc.append(("LinearReg", lrgacc))

        clf = SVC(kernel='rbf')
        clf.fit(self.X_tr, self.y_tr)
        ysvc = clf.predict(self.X_te)
        print(ysvc)
        svcaccr = accuracy_score(self.y_te, ysvc)
        self.acc.append(("SVC-rbf", svcaccr))

        clf = SVC(kernel='linear')
        clf.fit(self.X_tr, self.y_tr)
        ysvc = clf.predict(self.X_te)
        print(ysvc)
        svcaccl = accuracy_score(self.y_te, ysvc)
        self.acc.append(("SVC-linear", svcaccl))

        self.acc.sort(key=lambda tup: tup[1], reverse=True)
        for i in self.acc:
            print(i[0], " ", i[1] * 100)
