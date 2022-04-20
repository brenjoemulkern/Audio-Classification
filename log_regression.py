from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

def build_dataframe(csv_file):
    # make dataframe from csv
    print('Making dataframe from file:', csv_file)
    d_frame = pd.read_csv(csv_file, skiprows=1, header=None)
    print('Dataframe complete.')
    print('Number of rows: ', d_frame.shape[0])
    print('Number of columns: ', d_frame.shape[1])
    return d_frame



def main():
    log_reg = LogisticRegression(max_iter=10000, multi_class='ovr')

    mfcc_train_df = build_dataframe('data/trunc_mfcc_train.csv')
    X = mfcc_train_df.iloc[:, 2:]
    print(X.shape)
    y = mfcc_train_df.iloc[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=0)

    log_reg.fit(X_train, y_train)
    predictions = log_reg.predict(X_test)
    score = log_reg.score(X_test, y_test)
    print(score)

if __name__ == "__main__":
    main()