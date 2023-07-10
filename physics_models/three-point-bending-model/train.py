from decidion_tree import decision_tree_model, random_forest_model, svr_model, knn_model, Ridge_model, Lasso_model
# from MLP import MLP_model
from firedrake import *
import pandas as pd
from sklearn.model_selection import train_test_split


def data_process():
    # Generate a dataset
    df = pd.read_csv('/Users/mh522/Documents/new/graduation design/6.28code/three_point_bending/data.csv')
    X = df['w_max'].values.reshape(-1, 1)
    y = df['force'].values.reshape(-1, 1)

    # # Normalize the data
    # scaler_X = MinMaxScaler()
    # scaler_y = MinMaxScaler()
    # X = scaler_X.fit_transform(X)
    # y = scaler_y.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = data_process()  # parse dataset and split train and test set
    decision_tree_model(X_train, y_train, X_test, y_test)
    random_forest_model(X_train, y_train, X_test, y_test)
    svr_model(X_train, y_train, X_test, y_test)
    knn_model(X_train, y_train, X_test, y_test)
    Ridge_model(X_train, y_train, X_test, y_test)

   