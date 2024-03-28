import random
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt

from utils import load_data
random.seed(100)


def tsne_plot(x, y, labels):
    x_new = TSNE(n_components=2, random_state=100, learning_rate='auto', init='random').fit_transform(x)  # 422*2
    # print(x_new)

    # 绘制t-SNE结果
    plt.figure(figsize=(10, 6))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange'
    for i, c, label in zip(range(8), colors, labels):
        plt.scatter(x_new[y == i, 0], x_new[y == i, 1], c=c, label=label, alpha=0.8, s=10)
    plt.legend()
    plt.title('t-distributed Stochastic Neighbor Embedding')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.savefig("tsne.jpg")
    plt.show()


def pca_plot(x, y, labels):
    x_new = preprocessing.scale(x)
    x_new = PCA(n_components=2).fit_transform(x_new)
    print(x_new.shape)

    plt.figure(figsize=(10, 6))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange'
    for i, c, label in zip(range(8), colors, labels):
        plt.scatter(x_new[y == i, 0], x_new[y == i, 1], c=c, label=label, alpha=0.8, s=10)
    plt.legend()
    plt.title('Principal Component Analysis')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.savefig("PCA.jpg")
    plt.show()


def Spectral_Embedding(x, y, labels):
    x_new = SpectralEmbedding(n_components=2, random_state=100).fit_transform(x)
    print(x_new.shape)

    plt.figure(figsize=(10, 6))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange'
    for i, c, label in zip(range(8), colors, labels):
        plt.scatter(x_new[y == i, 0], x_new[y == i, 1], c=c, label=label, alpha=0.8, s=10)
    plt.legend()
    plt.title('Spectral Embedding')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.savefig("Spectral Embedding.jpg")
    plt.show()


if __name__ == '__main__':
    x, y, label = load_data()
    tsne_plot(x, y, label)
    pca_plot(x, y, label)
    Spectral_Embedding(x, y, label)