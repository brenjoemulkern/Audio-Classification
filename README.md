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

#### SVM

To run the SVM classifier, run:

<code>python svm.py</code>

which requires the MFCC data to have been processed (there should be <code>mean_mfcc_train.csv</code> and <code>mean_mfcc_test.csv</code> in the data directory).

This will make dataframes from the csv files, perform 5-fold cross validation on the training data, and classify the test data. The results of the validation will be printed, a plot of the confusion matrix will be saved to the "plots" directory, and a csv of the classified test data will be saved to the "submissions" directory.

#### Logistic Regression

To run the Logistic Regression classifier, run:

<code>python log_regression.py</code>

which requires the MFCC data to have been processed (there should be <code>mean_mfcc_train.csv</code> and <code>mean_mfcc_test.csv</code> in the data directory).

As with the SVM classifier, this will make dataframes from the csv files, perform 5-fold cross validation on the training data, and classify the test data. The results of the validation will be printed, a plot of the confusion matrix will be saved to the "plots" directory, and a csv of the classified test data will be saved to the "submissions" directory.

#### CNN: Waveforms

To run the CNN waveform classifier, run:

<code>python wave_neural_net.py</code>

which requires the waveform image data to have been processed (there should be 400 images in each of the 6 subdirectories of waveforms/train/ (2400 images total) and 1200 images in waveforms/test/).

This will load the images, determine the classes from the directory structure, and train the neural net using 80% of the data for training and 20% for validation.  The console will display the files found, the class names, and a summary of the neural network.  Then, when training, the progress of each epoch and its accuracy and loss will be displayed.  A confusion matrix of the validation data will be saved to the "plots" directory.

The program will train using the validation data, then will train the model again using all of the training data and provide a predicton on the test data.  The prediction will be saved to a csv in the "submissions" directory, and a plot of the training and validation loss and accuracy will be saved in the "plots" directory.

If desired, the user can run the program without the validation step, using all training data and saving a prediction csv by running:

<code>python wave_neural_net.py -p</code>

This will bypass the validation training and will save a prediction to the "submissions" directory.

#### CNN: Spectrograms

To run the CNN waveform classifier, run:

<code>python spectrogram_neural_net.py</code>

which requires the spectrogram image data to have been processed (there should be 400 images in each of the 6 subdirectories of spectrograms/train/ (2400 images total) and 1200 images in spectrograms/test/).

This will load the images, determine the classes from the directory structure, and train the neural net using 80% of the data for training and 20% for validation.  The console will display the files found, the class names, and a summary of the neural network.  Then, when training, the progress of each epoch and its accuracy and loss will be displayed.  A confusion matrix of the validation data will be saved to the "plots" directory.

The program will train using the validation data, then will train the model again using all of the training data and provide a predicton on the test data.  The prediction will be saved to a csv in the "submissions" directory, and a plot of the training and validation loss and accuracy will be saved in the "plots" directory.

If desired, the user can run the program without the validation step, using all training data and saving a prediction csv by running:

<code>python spectrogram_neural_net.py -p</code>

This will bypass the validation training and will save a prediction to the "submissions" directory.