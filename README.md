# Audio-Classification

## Project 3 for CS529 Spring 2022

### Setup

Run the setup.bash script to create the directory structure.
To run the bash script, run the following:

<code>./setup.bash</code>

This will create the necessary directories.
After these directories are created, place the training and testing mp3 files in the "train" and "test" folders respectively.

### Data Representation

The two files for processing the data are <code>image_processing.py</code> and <code>mfcc_data_process.py</code>.  

To process the data for use with SVM and Logistic Regression, run:

<code>python mfcc_data_process.py</code>

This will use librosa to process the MFCC for each mp3 in the training and testing folders and will print out the name of each file as it processes it.

To process the data for use with Convolutional Neural Networks, run:

<code>python image_processing.py</code>

This will use librosa to process the spectrogram and waveform for each file and will save them in the subdirectories of the "spectrograms" and "waveforms" folders.
This will save the training files into subdirectories based on their classes and the test files all in the same folder.

### Classification

There are four classification files: <code>svm.py</code>, <code>log_regression.py</code>, <code>wave_neural_net.py</code>, and <code>spectrogram_neural_net.py</code>.

To run the SVM classifier, run:

<code>svm.py</code>

which requires the MFCC data to have been processed (there should be <code>mean_mfcc_train.csv</code> and <code>mean_mfcc_test.csv</code> in the data directory).

This will make dataframes from the csv files, perform 5-fold cross validation on the training data, and classify the test data. The results of the validation will be printed, a plot of the confusion matrix will be saved to the "plots" directory, and a csv of the classified test data will be saved to the "submissions" directory.

To run the Logistic Regression classifier, run:

<code>log_regression.py</code>

which requires the MFCC data to have been processed (there should be <code>mean_mfcc_train.csv</code> and <code>mean_mfcc_test.csv</code> in the data directory).

As with the SVM classifier, this will make dataframes from the csv files, perform 5-fold cross validation on the training data, and classify the test data. The results of the validation will be printed, a plot of the confusion matrix will be saved to the "plots" directory, and a csv of the classified test data will be saved to the "submissions" directory.