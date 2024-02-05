import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('Auto.csv')

df = df.drop('name', axis=1)
df = df[df.horsepower != '?']
df['horsepower'] = df['horsepower'].astype('int64')
print(df)
mpg01 = []
mpgMedian = df["mpg"].median()
for x in df["mpg"]:
    if x > mpgMedian:
        mpg01.append(1)
    else:
        mpg01.append(0)
df.loc[:, ["mpg01"]] = mpg01

plt.scatter(df.mpg01, df.cylinders)
plt.title("cylinders")
#plt.show()
plt.scatter(df.mpg01, df.displacement)
plt.title("displacement")
#plt.show()
plt.scatter(df.mpg01, df.horsepower)
plt.title("horsepower")
#plt.show()
plt.scatter(df.mpg01, df.weight)
plt.title("weight")
#plt.show()
plt.scatter(df.mpg01, df.acceleration)
plt.title("acceleration")
#plt.show()
plt.scatter(df.mpg01, df.year)
plt.title("year")
#plt.show()
plt.scatter(df.mpg01, df.origin)
plt.title("origin")
#plt.show()

train, test = train_test_split(df, test_size=0.2)
x_train = train.drop("mpg01", axis=1)
x_train = x_train.drop("displacement", axis=1)
x_train = x_train.drop("cylinders", axis=1)
x_train = x_train.drop("year", axis=1)
x_train = x_train.drop("origin", axis=1)

y_train = train["mpg01"]

x_test = test.drop("mpg01", axis=1)
x_test = x_test.drop("displacement", axis=1)
x_test = x_test.drop("cylinders", axis=1)
x_test = x_test.drop("year", axis=1)
x_test = x_test.drop("origin", axis=1)
y_test = test["mpg01"]

clf = LinearDiscriminantAnalysis()
clf.fit(x_train, y_train)
predictionsLDA = clf.predict(x_test)

dlf = QuadraticDiscriminantAnalysis()
dlf.fit(x_train, y_train)
predictionsQDA = dlf.predict(x_test)


correct_predictionsLDA = (y_test == predictionsLDA).sum()
correct_predictionsQDA = (y_test == predictionsQDA).sum()

print("LDA: ", (1 - correct_predictionsLDA/predictionsLDA.size) * 100)
print("QDA: ", (1 - correct_predictionsQDA/predictionsQDA.size) * 100)

llf = LogisticRegression(random_state=0).fit(x_train, y_train)
predictionsLR = llf.predict(x_test)
correct_predictionsLR = (y_test == predictionsLR).sum()
print("LR: ", (1 - correct_predictionsLR/predictionsLR.size) * 100)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
predictionsKNN = knn.predict(x_test)
correct_predictionsKNN = (y_test == predictionsKNN).sum()
print("KNN: ", (1 - correct_predictionsKNN/predictionsKNN.size) * 100)


