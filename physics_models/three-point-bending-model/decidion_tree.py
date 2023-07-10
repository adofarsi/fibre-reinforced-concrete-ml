from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
import xgboost as xgb
from firedrake_adjoint import *
from firedrake import *
import numpy as np

def decision_tree_model(X_train, y_train, X_test, y_test):
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(X_train.reshape(-1, 1), y_train.ravel())
    print("Decision Tree evaluation(R2 score): {:.2f}".format(decision_tree.score(X_test, y_test)))
    score = decision_tree.score(X_test, y_test)
    print("Decision Tree evaluation (score): ", score)

def random_forest_model(X_train, y_train, X_test, y_test):
    random_forest = RandomForestRegressor(n_estimators=100)
    random_forest.fit(X_train.reshape(-1, 1), y_train.ravel())
    print("Random forest evaluation(R2 score): {:.2f}".format(random_forest.score(X_test, y_test)))
    score = random_forest.score(X_test, y_test)
    print("Random forest evaluation (score): ", score)

def svr_model(X_train, y_train, X_test, y_test):
    svr = SVR()
    svr.fit(X_train.reshape(-1, 1), y_train.ravel())
    print("SVR evaluation(R2 score): {:.2f}".format(svr.score(X_test, y_test)))
    score = svr.score(X_test, y_test)
    print("SVR evaluation (score):", score)

def knn_model(X_train, y_train, X_test, y_test):
    knn = KNeighborsRegressor()
    knn.fit(X_train.reshape(-1, 1), y_train.ravel())
    print("K-nearest neighbors evaluation (R2 score): {:.2f}".format(knn.score(X_test, y_test)))
    score = knn.score(X_test, y_test)
    print("K-nearest neighbors evaluation (score): ", score)

def Ridge_model(X_train, y_train, X_test, y_test):
    Ridge = linear_model.Ridge(alpha=.5)
    Ridge.fit(X_train.reshape(-1, 1), y_train.ravel())
    print("Ridge evaluation (R2 score): {:.2f}".format(Ridge.score(X_test, y_test)))
    score = Ridge.score(X_test, y_test)
    print("Ridge evaluation (score): ", score)

def Lasso_model(X_train, y_train, X_test, y_test):
    Lasso = linear_model.Lasso(alpha=.1)
    Lasso.fit(X_train.reshape(-1, 1), y_train.ravel())
    print("Lasso evaluation (R2 score): {:.2f}".format(Lasso.score(X_test, y_test)))
    score = Lasso.score(X_test, y_test)
    print("Lasso evaluation (score): ", score)
