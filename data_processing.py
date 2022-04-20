import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import os

for image in os.listdir('data/train/'):

    filename = 'data/train/00907299.mp3'
    print('File: {}'.format(filename))

    x, sr = librosa.load(('data/train/' + image), sr=None, mono=True)
    print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

    start, end = 7, 17
    ipd.Audio(data=x[start*sr:end*sr], rate=sr)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(10, 2)
    librosa.display.waveshow(x, sr=44100, alpha=0.5)
    plt.axis('off')
    plt.savefig(('waves/' + image[0:8]), dpi=226, bbox_inches='tight', pad_inches=0)
    plt.close('all')

    # start = len(x) // 2
    # plt.figure()
    # plt.plot(x[start:start+2000])
    # plt.ylim((-1, 1))
    # plt.show()

for file in os.listdir('data/train/'):
    print('Processing MFCC for', file)

    x, sr = librosa.load(file, sr=None, mono=True)
    print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))

    mfccs = librosa.feature.mfcc(x, sr=sr)
    print(mfccs.shape)