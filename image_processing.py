import librosa
import librosa.display
import IPython.display as ipd
import matplotlib
import matplotlib.pyplot as plt
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

matplotlib.use('Agg')

def sort_and_save_waveforms(train_df):

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    for image in sorted(os.listdir('data/train/')):

        x, sr = librosa.load(('data/train/' + image), sr=None, mono=True)

        cls = train_df.loc[train_df[0] == int(image[0:8]), 1]

        fig = plt.figure(frameon=False)
        fig.set_size_inches(5, 1.5)
        librosa.display.waveshow(x, sr=44100, alpha=0.5)
        plt.axis('off')
        plt.savefig(('waves/train/' + str(cls.iloc[0]) + '/' + image[0:8]), dpi=226, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        print('Processing file', image)

def save_test_waveforms():

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    for image in os.listdir('data/test/'):

        x, sr = librosa.load(('data/test/' + image), sr=None, mono=True)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(5, 1.5)
        librosa.display.waveshow(x, sr=44100, alpha=0.5)
        plt.axis('off')
        plt.savefig(('waves/test/' + image[0:8]), dpi=226, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        print('Processing file', image)

def sort_and_save_spectrograms(train_df):

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    for image in sorted(os.listdir('data/train/')):

        x, sr = librosa.load(('data/train/' + image), sr=None, mono=True)

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        log_mel = librosa.amplitude_to_db(mel)

        cls = train_df.loc[train_df[0] == int(image[0:8]), 1]

        fig = plt.figure(frameon=False)
        fig.set_size_inches(3, 1.5)
        librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(('spectrograms/train/' + str(cls.iloc[0]) + '/' + image[0:8]), dpi=226, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        print('Processing file', image)

def save_test_spectrograms():

    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    for image in os.listdir('data/test/'):

        x, sr = librosa.load(('data/test/' + image), sr=None, mono=True)

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        log_mel = librosa.amplitude_to_db(mel)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(3, 1.5)
        librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(('spectrograms/test/' + image[0:8]), dpi=226, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        print('Processing file', image)

def main():

    training_dataframe = build_dataframe('data/train.csv')

    sort_and_save_waveforms(training_dataframe)
    save_test_waveforms()

    sort_and_save_spectrograms(training_dataframe)
    save_test_spectrograms()

if __name__ == "__main__":
    main()