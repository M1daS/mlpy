import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix


import mglearn
from sklearn.datasets import load_iris
iris_dataset = load_iris()

### Classify and predict using KNN classifier of iris flowers dataset

## Keys##


print("Keys: " + str(iris_dataset.keys()))
print("Target names: {}".format(iris_dataset['target_names']))
print("feature names: {}".format(iris_dataset['feature_names']))
print("type of data: {}".format(iris_dataset['data'][:5])) #print first n rows of data
print("target names: {}".format(iris_dataset['target']))
print("target names: {}".format(iris_dataset['target'].shape))
### 0=setosa, 1=versicolor, 2=verginica ###

#######  Test/Train & Measure Success #####

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

print("xtrain shape {}".format(Xtrain.shape))
print("ytrain shape {}".format(Ytrain.shape))

print("xtest shape {}".format(Xtest.shape))
print("ytest shape {}".format(Ytest.shape))

iris_dataframe = pd.DataFrame(Xtrain, columns = iris_dataset.feature_names)
grr = scatter_matrix(iris_dataframe, c = Ytrain, alpha=8,figsize = (8,8), marker = 'o', hist_kwds = {'bins':20}, s=60, cmap=mglearn.cm3)


#display pajdas plot
import matplotlib.pyplot as plt
plt.show(block=True) #display the pandas plot grr created above w block true so it does not close when program run ends

#### KNN Classifier #####
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xtrain, Ytrain)

Xnew = np.array([[2.4,4,8,0.9]])
print("Xnew shape {}".format(Xnew.shape))

prediction = knn.predict(Xnew)
print("prediction shape {}".format(prediction.shape))
print("prediction  target name  {}".format(iris_dataset['target_names'][prediction]))

Ypred = knn.predict(Xtest)
### Score ###
print("test set score {:.2f}".format(np.mean(Ypred == Ytest)))
print("test knn  score {:.2f}".format(knn.score(Xtest, Ytest)))




