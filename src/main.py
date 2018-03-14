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

models = {
    'LR': LogisticRegression(),
    'LDA': LinearDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(),
    'NB': GaussianNB(),
    'SVM': SVC()}


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


def compare_algorithms(results, names):
    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def build_model(model):
    seed = 7
    scoring = 'accuracy'
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    return model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)


def evaluate_models():
    # Test options and evaluation metric

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models.items():
        cv_results = build_model(model)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        # print(msg)
    return results, names


def split_out_validation_dataset(dataset):
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    validation_size = 0.20
    seed = 7
    return model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


def make_predictions_with_model(model):
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


if __name__ == '__main__':
    dataset = load_data_set()

    # print_basic_data(dataset)
    # show_box_and_whisker_plots(dataset)
    # show_histogram(dataset)
    # show_scatter_matrix(dataset)

    X_train, X_validation, Y_train, Y_validation = split_out_validation_dataset(dataset)
    # Split-out validation dataset

    # results, names = evaluate_models()

    # compare_algorithms(results, names)

    # Make predictions on validation dataset
    make_predictions_with_model(models['KNN'])
    make_predictions_with_model(models['SVM'])
