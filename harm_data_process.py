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

def build_harm_dataframe():
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    harm_list = []

    for file in os.listdir('data/dummy_train/'):
        x,sr = librosa.load('data/dummy_train/00907299.mp3', sr=None, mono=True)
        y = librosa.effects.harmonic(x)
        harm = librosa.feature.tonnetz(y=y, sr=sr)

        harm_trunc = harm[:, :2401]
        harm_flat = harm_trunc.flatten()
        harm_list.append(harm_flat)
        print('Processing file', file)
    
    harm_df = pd.DataFrame(np.vstack(harm_list))
    return harm_df

def main():
    
    train_df = build_dataframe('data/train.csv')
    train_df = train_df.sort_values(by=[0])
    train_df = train_df.reset_index(drop=True)

    harm_df = build_harm_dataframe()
    harm_df.to_csv('data/trunc_harm.csv', index=False)
    harm_train_df = pd.concat([train_df, harm_df], axis=1)
    harm_train_df.to_csv('data/trunc_harm_train.csv', index=False)
    
if __name__ == "__main__":
    main()
