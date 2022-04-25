import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

def build_dataframe(csv_file):
    # make dataframe from csv
    print('Making dataframe from file:', csv_file)
    d_frame = pd.read_csv(csv_file, skiprows=1, header=None)
    print('Dataframe complete.')
    print('Number of rows: ', d_frame.shape[0])
    print('Number of columns: ', d_frame.shape[1])
    return d_frame

def write_csv(id_list, class_list, filename):
    print('Writing csv')
    filename = filename + '.csv'
    indices = np.asarray(id_list[0])

    # concatenate arrays, transpose, and save as csv
    full_array = np.concatenate(([indices], [class_list]), axis=0)
    full_array_transpose = np.transpose(full_array)
    full_dataframe = pd.DataFrame(full_array_transpose, columns = ['id','genre'])
    full_dataframe.to_csv(filename, index=False)

def main():

    # build training and testing dataframes
    mfcc_train_df = build_dataframe('data/mean_mfcc_train.csv')
    mfcc_test_df = build_dataframe('data/mean_mfcc_test.csv')

    # define X and y as features and classes from all training data
    X = mfcc_train_df.iloc[:, 2:]
    y = mfcc_train_df.iloc[:, 1]

    # 80/20 validation split 
    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=0)

    # define model with 10000 iterations and "one vs all" classification
    log_reg = LogisticRegression(max_iter=10000, multi_class='ovr')

    # fit validation split data, make confusion matrix
    log_reg.fit(X_train, y_train)
    log_pred = log_reg.predict(X_test)
    log_accuracy = accuracy_score(log_pred, y_test)
    log_cf = confusion_matrix(y_test, log_pred)

    # train and test with 5-fold cross validation
    log_cv_scores = cross_val_score(log_reg, X, y, cv=5)

    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(log_cf,
                xticklabels=[0,1,2,3,4,5],
                yticklabels=[0,1,2,3,4,5],
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    # display results
    print('Logistic Regression Accuracy:', log_accuracy)
    print('Logistic Regression Cross-Validation Scores:', log_cv_scores)

    # train model with all training data, predict test classes
    log_reg.fit(X, y)
    X_new = mfcc_test_df.iloc[:, 1:]
    y_new = log_reg.predict(X_new)

    # write results to csv
    write_csv(mfcc_test_df.iloc[:, :1], y_new, 'log_reg_submission')

if __name__ == "__main__":
    main()