# Audio-Classification

## Project 3 for CS529 Spring 2022

### Setup

Run the setup.bash script to create the directory structure.
To run the bash script, run the following:

<code>./setup.bash<code>

This will create the necessary directories.
After these directories are created, place the training and testing mp3 files in the "train" and "test" folders respectively.

### Data Representation

The two files for processing the data are <code>image_processing.py<code> and <code>mfcc_data_process.py<code>.  

To process the data for use with SVM and Logistic Regression, run:

<code>python mfcc_data_process.py<code>

This will use librosa to process the MFCC for each mp3


