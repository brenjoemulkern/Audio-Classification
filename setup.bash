#!/bin/bash

# build data, plots, and submissions directories
mkdir data
mkdir plots
mkdir submissions

# build spectrogram directory with train and test subdirectories
# build class subdirectories in train
mkdir spectrograms
cd spectrograms
mkdir train
mkdir test
cd train
mkdir 0
mkdir 1
mkdir 2
mkdir 3
mkdir 4
mkdir 5

cd ..
cd ..

# build wave directory with train and test subdirectories
# build class subdirectories in train
mkdir waves
cd waves
mkdir train
mkdir test
cd train
mkdir 0
mkdir 1
mkdir 2
mkdir 3
mkdir 4
mkdir 5

cd ..
cd ..
