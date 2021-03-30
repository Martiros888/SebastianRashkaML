import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from Perceptron import Perceptron
from matplotlib.colors import ListedColormap
from Adaline import AdalineGD
from AdalineSGD import AdalineSGD

# page 63
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# df = pd.read_csv(url,header=None,encoding='utf-8')
# y = df.iloc[0:100,4].values
# y = np.where(y == 'Iris-setosa',-1,1)
# X = df.iloc[0:100,[0,2]].values
# plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='Шетинистый')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')
# plt.xlabel('длина чашелистика см')
# plt.ylabel('длина лепестка см')
# plt.legend(loc='upper left')
# plt.show()


# page 64
# ppn = Perceptron(eta=0.01,n_iter=10)
# ppn.fit(X,y)
# print(ppn.errors_)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Эпохи')
# plt.ylabel('Количества обновление')
# plt.show()


# page 65
# def plot_decision_regions(X, y, classifer, resolution=0.02):
#     markers = ('s','x','o','4','4')
#     colors = ('red','blue','lightgreen','gray','cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                            np.arange(x2_min, x2_max, resolution))
#     Z = classifer.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())

#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0],
#                     y=X[y == cl, 1],
#                     alpha=0.8,
#                     c=colors[idx],
#                     marker=markers[idx],
#                     label=cl,
#                     edgecolor='black')
        
# plot_decision_regions(X, y, ppn)
# plt.xlabel('длина чашелистика')
# plt.ylabel('длина лепестка')
# plt.legend(loc='upper left')
# plt.show()


# page 74
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
# ax[0].plot(range(1, len(ada1.cost_) + 1),
#            np.log10(ada1.cost_), marker='o' )
# ax[0].set_xlabel('Эпохи')
# ax[0].set_ylabel('log(сумма квадратных ошибок)')
# ax[0].set_title('Adaline - скорость обучения 0.01')
# ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
# ax[1].plot(range(1, len(ada2.cost_) + 1),
#            ada2.cost_, marker='o')
# ax[1].set_xlabel('Эпохи')
# ax[1].set_ylabel('log(сумма квадратных ошибок)')
# ax[1].set_title('Adaline - скорость обучения 0.0001')
# plt.show()

# page 77
# X_std = np.copy(X)
# X_std[:, 0] = (X[:, 0] - X[:,0].mean()) / X[:, 0].std()
# X_std[:, 1] = (X[:, 1] - X[:,1].mean()) / X[:, 1].std()
# ada_gd = AdalineGD(n_iter=15, eta=0.01)
# ada_gd.fit(X_std,y)
# plot_decision_regions(X_std, y, classifer=ada_gd)
# plt.title('Adaline - градиентных спуск')
# plt.xlabel('длина чашелистика [стандартизированный]')
# plt.ylabel('Adaline - длина лепестка [стандартизированный]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()
# plt.plot(range(1, len(ada_gd.cost_) + 1),
#          ada_gd.cost_, marker='o')
# plt.xlabel('Эпохи')
# plt.ylabel('сумма квадратных ошибок')
# plt.tight_layout()
# plt.show()

# page 83
# ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# ada_sgd.fit(X_std, y)
# plot_decision_regions(X_std, y, classifer=ada_sgd)
# plt.title('Adaline - стохастический градиентный спуск')
# plt.xlabel('длина чашелистика [стандартизированный]')
# plt.ylabel('длина лепестка [стандартизированный]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()
# plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
# plt.xlabel('Эпохи')
# plt.ylabel('Усреднение издержки')
# plt.tight_layout()
# plt.show()

# page 87
# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris.data[:, [2, 3]]
# y = iris.target
# print(np.unique(y))
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# print(X_train,y_train)

# page 89
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)

# from sklearn.linear_model import Perceptron
# ppn = Perceptron(eta0=0.1, random_state=1)
# ppn.fit(X_train_std, y_train)
# y_pred = ppn.predict(X_test_std)
# print((y_test != y_pred).sum())

# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, y_pred))


# page 92
# def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
#                            np.arange(x2_min, x2_max, resolution))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())

#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
#                     alpha=0.8, c=colors[idx],
#                     marker=markers[idx], label=cl,
#                     edgecolor='black')

#     if test_idx:    
#         X_test, y_test = X[test_idx, :], y[test_idx]
#         plt.scatter(X_test[:, 0], X_test[:, 1],
#                     c='', edgecolor='black', alpha=1.0,
#                     linewidth=1, marker='o',
#                     s=100, label='испытательный набор')

# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
# plt.xlabel('длина лепестка [стандартизированный]')
# plt.ylabel('ширина лепестка [стандартизированный]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()



sigmoid = lambda x: 1/(1+np.exp(-x))




