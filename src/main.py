# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def load_data_set():
    # Load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    return pandas.read_csv(url, names=names)


def print_basic_data(dataset):
    print(dataset.shape)
    print(dataset.head(20))
    print(dataset.describe)
    print(dataset.groupby("class").size())


def show_box_and_whisker_plots(dataset):
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()


def show_histogram(dataset):
    dataset.hist()
    plt.show()


def show_scatter_matrix(dataset):
    scatter_matrix(dataset)
    plt.show()


if __name__ == '__main__':
    dataset = load_data_set()

    #print_basic_data(dataset)
    #show_box_and_whisker_plots(dataset)
    #show_histogram(dataset)
    #show_scatter_matrix(dataset)

    exit(0)

