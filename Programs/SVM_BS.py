import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from numpy import genfromtxt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, metrics,preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier





class ProjectModel:

    def __init__(self):
        self.labels = genfromtxt(r"D:\Documents\UTRGV\Fall Semester 2020\Machine Learning\Group Project\data\Collected\Final\S&P_Price5YearsBuySell.csv", delimiter=',')
        self.labels_weighted = genfromtxt(r"D:\Documents\UTRGV\Fall Semester 2020\Machine Learning\Group Project\data\Collected\Final\S&P_Price5YearsBuySellWeighted.csv")
        self.sentiment_df = pd.read_csv(r"D:\Documents\UTRGV\Fall Semester 2020\Machine Learning\Group Project\data\Collected\Final\stock_sentiment.csv")
        self.dates = self.sentiment_df.columns
        self.sentiment_df = self.sentiment_df.fillna(0)
        self.sentiment_df = self.sentiment_df.transpose()

    def preprocess(self):
        sentiment_x = self.sentiment_df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        sentiment_scaled = min_max_scaler.fit_transform(sentiment_x)
        self.sentiment_df = pd.DataFrame(sentiment_scaled)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.sentiment_df, self.labels,
                                                                                test_size=0.3,
                                                                                random_state=109)  # 70% training and 30% test

    def split_data(self):
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

    def SVM_Train(self):
        params = self.svm_param_selection(self.X_train, self.y_train, 10)
        tuned_gamma = params["gamma"]
        tuned_C = params["C"]

        accuracy = []

        # Create a svm Classifier
        clf = svm.SVC(kernel='rbf', gamma=tuned_gamma, C=tuned_C, max_iter=10000)  # Linear Kernel

        # Train the model using the training sets
        clf.fit(self.X_train, self.y_train)

        # Predict the response for test dataset
        y_pred = clf.predict(self.X_test)

        accuracy.append(metrics.accuracy_score(self.y_test, y_pred))

        # Model Accuracy: how often is the classifier correct?
        print("SVM Accuracy:", metrics.accuracy_score(self.y_test, y_pred))

        # Model Precision: what percentage of positive tuples are labeled as such?
        print("SVM Precision:", metrics.precision_score(self.y_test, y_pred))

        # Model Recall: what percentage of positive tuples are labelled as such?
        print("SVM Recall:", metrics.recall_score(self.y_test, y_pred))


    def svm_param_selection(self, X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        grid_search.best_params_
        print(grid_search.best_params_)
        return grid_search.best_params_

    def Random_Forest_Train(self):
        # Create first pipeline for base without reducing features.

        pipe = Pipeline([('classifier', RandomForestClassifier())])

        # Create param grid.

        param_grid = [
            {'classifier': [LogisticRegression()],
             'classifier__penalty': ['l1', 'l2'],
             'classifier__C': np.logspace(-4, 4, 20),
             'classifier__solver': ['liblinear']},
            {'classifier': [RandomForestClassifier()],
             'classifier__n_estimators': list(range(10, 101, 10)),
             'classifier__max_features': list(range(6, 32, 5))}
        ]

        # Create grid search object

        clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

        # Fit on data

        best_clf = clf.fit(self.X_train, self.y_train)
        pred_label = clf.predict(self.X_test)

        # Model Accuracy: how often is the classifier correct?
        print("Random Forest Accuracy:", metrics.accuracy_score(self.y_test, pred_label))

        # Model Precision: what percentage of positive tuples are labeled as such?
        print("Random Forest Precision:", metrics.precision_score(self.y_test, pred_label))

        # Model Recall: what percentage of positive tuples are labelled as such?
        print("Random Forest Recall:", metrics.recall_score(self.y_test, pred_label))


    def LR_Train(self):
        clf = LogisticRegression(random_state=0).fit(self.X_train, self.y_train)
        pred_label = clf.predict(self.X_test)

        # Model Accuracy: how often is the classifier correct?
        print("Logistic Regression Accuracy:", metrics.accuracy_score(self.y_test, pred_label))

        # Model Precision: what percentage of positive tuples are labeled as such?
        print("Logistic Regression Precision:", metrics.precision_score(self.y_test, pred_label))

        # Model Recall: what percentage of positive tuples are labelled as such?
        print("Logistic Regression Recall:", metrics.recall_score(self.y_test, pred_label))





pj = ProjectModel()
pj.split_data()
pj.SVM_Train()
pj.LR_Train()
pj.Random_Forest_Train()

pj.preprocess()
pj.SVM_Train()
pj.LR_Train()
pj.Random_Forest_Train()














