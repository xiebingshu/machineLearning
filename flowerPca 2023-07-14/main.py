import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition


def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


def pca(data):
    x, y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(x)
    print('可解释的方差占比:%s', str(pca.explained_variance_ratio_))


def plot_pca(data):
    x, y = data
    pca = decomposition.PCA(n_components=2)
    pca.fit(x)
    x_r = pca.transform(x)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0.5), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0), (0, 0.7, 0.3), (0.4, 0.4, 0.2))
    for label, color in zip(np.unique(y), colors):
        position = y == label
        ax.scatter(x_r[position, 0], x_r[position, 1], label='target = %d'%label, color=color)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('y[0]')
    ax.legend(loc='best')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax.set_title('PCA降维样本分布图')
    plt.show()



if __name__ == '__main__':
    x, y = load_data()
    print(x[:6])
    pca(load_data())
    plot_pca(load_data())

