import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from numpy import genfromtxt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, metrics,preprocessing
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis




class ProjectModel:

    def __init__(self):
        self.labels = genfromtxt(r"D:\Documents\UTRGV\Fall Semester 2020\Machine Learning\Group Project\data\Collected\Final\S&P_Price5YearsBuySell.csv", delimiter=',')
        self.labels_weighted = genfromtxt(r"D:\Documents\UTRGV\Fall Semester 2020\Machine Learning\Group Project\data\Collected\Final\S&P_Price5YearsBuySellWeighted.csv")
        self.sentiment_df = pd.read_csv(r"/Group Project/data/Collected/Final/stock_sentiment.csv")
        self.dates = self.sentiment_df.columns
        self.sentiment_df = self.sentiment_df.fillna(0)
        self.sentiment_df = self.sentiment_df.transpose()

        print(self.labels)

    def preprocess(self):
        sentiment_x = self.sentiment_df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        sentiment_scaled = min_max_scaler.fit_transform(sentiment_x)
        self.sentiment_df = pd.DataFrame(sentiment_scaled)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.sentiment_df, self.labels, test_size=0.3,
                                                            random_state=109)  # 70% training and 30% test

    def plot_correlation(self, num_companies):
        #scatter matrix
        scatter_matrix(self.sentiment_df[[num for num in range(0,num_companies)]],figsize=(100,100), s= 1000)
        plt.savefig(r"figure_1.png")

    def plot_labels(self, num_companies):
        fig, axs = plt.subplots(num_companies, num_companies,figsize = (30,30))
        for company_y in range(num_companies):
            for company_x in range(num_companies):
                axs[company_y, company_x].scatter( self.dates, self.sentiment_df[num_companies * company_y + company_x], alpha = .2, c= self.labels)
                axs[company_y, company_x].set_title(f'C {num_companies * company_y + company_x}')
                axs[company_y, company_x].set_yticklabels([])
                axs[company_y, company_x].set_xticklabels([])


        for ax in axs.flat:
            ax.set(xlabel='Date', ylabel='Sentiment')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.show()

    def ex(self):
        from sklearn.datasets import load_iris
        iris = load_iris()
        features = iris.data.T
        print(features[3])
        print(iris.target)

        plt.scatter(features[0], features[1], alpha=0.2,
                    s=100 * features[3], c=iris.target, cmap='viridis')
        plt.xlabel(iris.feature_names[0])
        plt.ylabel(iris.feature_names[1]);

    def SVM_Train(self):
        tuned_C, tuned_gamma = self.svc_param_selection(self.X_train, self.y_train, 10)

        # Create a svm Classifier
        clf = svm.SVC(kernel='rbf', gamma=tuned_gamma, C=tuned_C)  # Linear Kernel

        # Train the model using the training sets
        clf.fit(self.X_train, self.y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(self.X_test)

        # Model Accuracy: how often is the classifier correct?
        print("SVM Accuracy:", metrics.accuracy_score(self.y_test, y_pred))

        # Model Precision: what percentage of positive tuples are labeled as such?
        print("SVM Precision:", metrics.precision_score(self.y_test, y_pred))

        # Model Recall: what percentage of positive tuples are labelled as such?
        print("SVM Recall:", metrics.recall_score(self.y_test, y_pred))

    def svm_param_selection(X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        grid_search.best_params_
        return grid_search.best_params_


    def compare_classifiers(self):

        h = .02  # step size in the mesh

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes", "QDA"]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

        X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        datasets = [make_moons(noise=0.3, random_state=0),
                    make_circles(noise=0.2, factor=0.5, random_state=1),
                    linearly_separable
                    ]

        figure = plt.figure(figsize=(27, 9))
        i = 1
        # iterate over datasets

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

        plt.tight_layout()
        plt.show()


pj = ProjectModel()
pj.preprocess()
 # pj.ex()
pj.plot_labels(5)













