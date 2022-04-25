import librosa
import numpy as np
import os
import pandas as pd
import warnings

def build_dataframe(csv_file):
    # make dataframe from csv
    print('Making dataframe from file:', csv_file)
    d_frame = pd.read_csv(csv_file, skiprows=1, header=None)
    print('Dataframe complete.')
    print('Number of rows: ', d_frame.shape[0])
    print('Number of columns: ', d_frame.shape[1])
    return d_frame

def build_trunc_mfcc_dataframe():

    # ignore warnings from using audioread backend for librosa
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    mfcc_list = []
    for file in sorted(os.listdir('data/train/')):
        x, sr = librosa.load(('data/train/' + file), sr=None, mono=True)

        # take first 2400 columns and flatten into vector
        mfcc = librosa.feature.mfcc(x, sr=sr)
        mfcc_trunc = mfcc[:, :2401]
        mfcc_flat = mfcc_trunc.flatten()
        mfcc_list.append(mfcc_flat)
        print('Processing file', file)

    # make dataframe from list
    mfcc_df = pd.DataFrame(np.vstack(mfcc_list))
    return mfcc_df

def build_mean_mfcc_dataframe():

    # ignore warnings from using audioread backend for librosa
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    mfcc_list = []
    for file in sorted(os.listdir('./data/train/')):
        x, sr = librosa.load(('data/train/' + file), sr=None, mono=True)

        # take mean of each row of mfccs and append to list
        mfcc = librosa.feature.mfcc(x, sr=sr)
        mean_mfcc = mfcc.mean(axis=1)
        mfcc_list.append(mean_mfcc)
        print('Processing file', file)

    # make dataframe from list
    mfcc_df = pd.DataFrame(np.vstack(mfcc_list))
    return mfcc_df

def build_trunc_mfcc_test_dataframe():

    # ignore warnings from using audioread backend for librosa
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    mfcc_list = []
    for file in sorted(os.listdir('data/test/')):
        x, sr = librosa.load(('data/test/' + file), sr=None, mono=True)

        # take first 2400 columns then flatten into vector
        mfcc = librosa.feature.mfcc(x, sr=sr)
        mfcc_trunc = mfcc[:, :2401]
        mfcc_flat = mfcc_trunc.flatten()
        mfcc_list.append(mfcc_flat)
        print('Processing file', file)

    # make dataframe from list
    mfcc_df = pd.DataFrame(np.vstack(mfcc_list))
    return mfcc_df

def build_mean_mfcc_test_dataframe():
    # build dataframe from testing mp3s

    # ignore warnings from using audioread backend for librosa
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    mfcc_list = []
    for file in sorted(os.listdir('./data/test/')):
        x, sr = librosa.load(('data/test/' + file), sr=None, mono=True)

        # take mean of each test mfcc and append t list
        mfcc = librosa.feature.mfcc(x, sr=sr)
        mean_mfcc = mfcc.mean(axis=1)
        mfcc_list.append(mean_mfcc)
        print('Processing file', file)

    # make dataframe from list
    mfcc_df = pd.DataFrame(np.vstack(mfcc_list))
    return mfcc_df

def main():

    # make dataframe of training labels
    train_df = build_dataframe('data/train.csv')
    train_df = train_df.sort_values(by=[0])
    train_df = train_df.reset_index(drop=True)

    # make dataframe of testing ids
    test_df = build_dataframe('data/test_idx.csv')

    # build truncated flattened dataframe, combine with training labels, save as csv
    mfcc_trunc_df = build_trunc_mfcc_dataframe()
    mfcc_trunc_df.to_csv('data/trunc_mfcc.csv', index=False)
    mfcc_trunc_train_df = pd.concat([train_df, mfcc_trunc_df], axis=1)
    mfcc_trunc_train_df.to_csv('data/trunc_mfcc_train.csv', index=False)

    # build mean mfcc dataframe, combine with training labels, save as csv
    mfcc_mean_df = build_mean_mfcc_dataframe()
    mfcc_mean_df.to_csv('data/mean_mfcc.csv', index=False)
    mfcc_mean_train_df = pd.concat([train_df, mfcc_mean_df], axis=1)
    mfcc_mean_train_df.to_csv('data/mean_mfcc_train.csv', index=False)

    # build mfcc truncated csv
    mfcc_trunc_test = build_trunc_mfcc_test_dataframe()
    mfcc_trunc_test_full = pd.concat([test_df, mfcc_trunc_test], axis=1)
    mfcc_trunc_test_full.to_csv('data/trunc_mfcc_test.csv', index=False)

    # build mfcc mean csv
    mfcc_mean_test = build_mean_mfcc_test_dataframe()
    mfcc_mean_test_full = pd.concat([test_df, mfcc_mean_test], axis=1)
    mfcc_mean_test_full.to_csv('data/mean_mfcc_test.csv', index=False)


if __name__ == "__main__":
    main()
