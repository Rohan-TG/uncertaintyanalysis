import os

import xgboost
import pandas as pd
import random


error_data_directory = input('Error data directory: ')

all_files = os.listdir(error_data_directory)

training_fraction = float(input('Training fraction: '))
n_training_samples = int(training_fraction * len(all_files))

training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_files)
	if choice not in training_files:
		training_files.append(choice)


validation_files = []
for f in all_files:
	if f not in training_files:
		validation_files.append(f)

