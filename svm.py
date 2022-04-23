import librosa
import numpy as np
import os
import pandas as pd
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import warnings

def build_dataframe(csv_file):
    # make dataframe from csv
    print('Making dataframe from file:', csv_file)
    d_frame = pd.read_csv(csv_file, skiprows=1, header=None)
    print('Dataframe complete.')
    print('Number of rows: ', d_frame.shape[0])
    print('Number of columns: ', d_frame.shape[1])
    return d_frame

def build_mfcc_dataframe():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    mfcc_list = []
    for file in sorted(os.listdir('./data/train/')):
        x, sr = librosa.load(('data/train/' + file), sr=None, mono=True)
        #print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

        mfcc = librosa.feature.mfcc(x, sr=sr)
        mean_mfcc = mfcc.mean(axis=1)
        mfcc_list.append(mean_mfcc)
        print('Processing file', file)

    mfcc_df = pd.DataFrame(np.vstack(mfcc_list))
    return mfcc_df

def main():

    mfcc_train_df = build_dataframe('data/trunc_mfcc_train.csv')
    X = mfcc_train_df.iloc[:, 2:]
    print(X.shape)
    y = mfcc_train_df.iloc[:, 1]
    print(y.shape)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=101)

    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
    rbf_cv = svm.SVC(kernel='rbf', gamma=0.5, C=0.1)
    scores = cross_val_score(rbf_cv, X, y, cv=5)

    rbf_pred = rbf.predict(X_test)

    rbf_accuracy = accuracy_score(y_test, rbf_pred)
    rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
    print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
    print('CV Scores: ', scores)

    # poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)

    # poly_pred = poly.predict(X_test)

    # poly_accuracy = accuracy_score(y_test, poly_pred)
    # poly_f1 = f1_score(y_test, poly_pred, average='weighted')
    # print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
    # print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

if __name__ == "__main__":
    main()
