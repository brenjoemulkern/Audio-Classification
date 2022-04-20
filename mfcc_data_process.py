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

def build_mfcc_dataframe():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    mfcc_list = []
    for file in sorted(os.listdir('data/train/')):
        x, sr = librosa.load(('data/train/' + file), sr=None, mono=True)
        #print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

        mfcc = librosa.feature.mfcc(x, sr=sr)
        # mean_mfcc = mfcc.mean(axis=1)
        mfcc_trunc = mfcc[:, :2401]
        mfcc_flat = mfcc_trunc.flatten()
        mfcc_list.append(mfcc_flat)
        print('Processing file', file)

    mfcc_df = pd.DataFrame(np.vstack(mfcc_list))
    return mfcc_df

def main():
    # warnings.simplefilter("ignore", UserWarning)
    # warnings.simplefilter("ignore", FutureWarning)
    # x,sr = librosa.load('data/dummy_train/00907299.mp3', sr=None, mono=True)
    # mfcc = librosa.feature.mfcc(x, sr=sr)
    # print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))
    # print(mfcc.shape)

    train_df = build_dataframe('data/train.csv')
    train_df = train_df.sort_values(by=[0])
    train_df = train_df.reset_index(drop=True)

    mfcc_df = build_mfcc_dataframe()
    mfcc_df.to_csv('data/trunc_mfcc.csv', index=False)
    mfcc_train_df = pd.concat([train_df, mfcc_df], axis=1)
    mfcc_train_df.to_csv('data/trunc_mfcc_train.csv', index=False)

if __name__ == "__main__":
    main()


