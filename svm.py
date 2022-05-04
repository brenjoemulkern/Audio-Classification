import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

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

    # make dataframes of training and testing data
    mfcc_train_df = build_dataframe('data/mean_mfcc_train.csv')
    mfcc_test_df = build_dataframe('data/mean_mfcc_test.csv')

    # define X and y as features and classes of all training data
    X = mfcc_train_df.iloc[:, 2:]
    y = mfcc_train_df.iloc[:, 1]

    # validation split for confusion matrix validation
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)

    # train and test with 80/20 split, make confusion matrix
    poly_model = svm.SVC(kernel='poly', degree=2, C=1)
    poly_model.fit(X_train, y_train)
    poly_pred = poly_model.predict(X_test)
    poly_accuracy = accuracy_score(y_test, poly_pred)
    poly_cf = confusion_matrix(y_test, poly_pred)

    # train and test with 5-fold cross validation
    poly_cv_scores = cross_val_score(poly_model, X, y, cv=5)

    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(poly_cf,
                xticklabels=[0,1,2,3,4,5],
                yticklabels=[0,1,2,3,4,5],
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    # display results
    print('SVM Polynomial Accuracy: ', poly_accuracy)
    print('SVM Polynomial Cross-Validation Scores:', poly_cv_scores)

    # use model to predict test data classes
    poly_model.fit(X, y)
    X_new = mfcc_test_df.iloc[:, 1:]
    y_new = poly_model.predict(X_new)

    # write results to csv
    write_csv(mfcc_test_df.iloc[:, :1], y_new, 'svm_submission')



if __name__ == "__main__":
    main()
