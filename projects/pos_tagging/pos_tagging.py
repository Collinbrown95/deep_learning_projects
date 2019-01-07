# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:53:08 2018

@author: collin.brown
"""
#==============================================================================
# Imports
#==============================================================================
import os
import pandas as pd
import numpy as np
import string 
import re
from time import time
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
#==============================================================================
# Folders
#==============================================================================
data_folder = r"" # ENTER PATH TO DATA HERE
train_path = os.path.join(data_folder, 'train.col')
dev_path = os.path.join(data_folder, 'dev.col')

#==============================================================================
# Functions
#==============================================================================
def import_col_data(path_to_data):
    """Imports data from plain text (.col) file.
    Args:
        path_to_data:
            A string containing the path to the .col file

    Returns:
        DataFrame:
            A pandas dataframe containing input data from file
    """
    data = []
    with open(path_to_data, 'rt', encoding='utf-8') as col_data:
        for line in col_data:
            if not line.startswith('-DOCSTART-'):
                line = line.strip()
                if line != '':
                    row = line.split('\t')
                else:
                    row = ['-BREAK-', '-BREAK-']
                data.append(row)
    data.append(['-BREAK-', '-BREAK-'])
    data = pd.DataFrame(data)
    return data

def generate_features(X):
    '''Creates a dict of features 
    '''
    temp_dict = {}
    for idx, i in enumerate(X):
        # Start sent_index to store position in the index
        sent_index = 0
        if i == "-BREAK-":
            sent_index = 0
        else:
            temp_dict[idx] = {}
            # Adding word length
            temp_dict[idx]['num_chars'] = len(i)
            # Add position in text 
            temp_dict[idx]['position'] = sent_index
            try:
                # Suffixes
                # Add suffix "y"
                temp_dict[idx]['has_suffix_y'] = True if re.match(".*y", X[idx]) else False
                temp_dict[idx]['has_suffix_ing'] = True if re.match(".*ing", X[idx]) else False
                temp_dict[idx]['has_suffix_ed'] = True if re.match(".*ed", X[idx]) else False
                temp_dict[idx]['has_suffix_acy'] = True if re.match(".*acy", X[idx]) else False
                temp_dict[idx]['has_suffix_al'] = True if re.match(".*al", X[idx]) else False
                temp_dict[idx]['has_suffix_able'] = True if re.match(".*able", X[idx]) else False
                temp_dict[idx]['has_suffix_er'] = True if re.match(".*er", X[idx]) else False
                temp_dict[idx]['has_suffix_ity'] = True if re.match(".*ity", X[idx]) else False
                temp_dict[idx]['has_suffix_ive'] = True if re.match(".*ive", X[idx]) else False
                temp_dict[idx]['has_suffix_less'] = True if re.match(".*less", X[idx]) else False
                temp_dict[idx]['has_suffix_ous'] = True if re.match(".*ous", X[idx]) else False
                # Prefixes
                temp_dict[idx]['has_prefix_pre'] = True if re.match("pre.*", X[idx]) else False
                temp_dict[idx]['has_suffix_anti'] = True if re.match("anti.*", X[idx]) else False
                temp_dict[idx]['has_suffix_auto'] = True if re.match("auto.*", X[idx]) else False
                temp_dict[idx]['has_suffix_de'] = True if re.match("de.*", X[idx]) else False
                temp_dict[idx]['has_suffix_dis'] = True if re.match("dis.*", X[idx]) else False
                temp_dict[idx]['has_suffix_in'] = True if re.match("in.*", X[idx]) else False
                temp_dict[idx]['has_suffix_en'] = True if re.match("en.*", X[idx]) else False
                temp_dict[idx]['has_suffix_inter'] = True if re.match("inter.*", X[idx]) else False
                temp_dict[idx]['has_suffix_re'] = True if re.match("re.*", X[idx]) else False
                temp_dict[idx]['has_suffix_under'] = True if re.match("under.*", X[idx]) else False
                # Specific last words
                temp_dict[idx]['prev_word_the'] = True if re.match("[T|t]he", X[idx-1]) else False
                temp_dict[idx]['prev_word_be'] = True if re.match("is|was|am|are", X[idx-1]) else False
                temp_dict[idx]['prev_word_have'] = True if re.match("have|has", X[idx-1]) else False
            except:
                pass
            # Add next word
            #try:
                #temp_dict[idx]['next_1_word'] = X[idx+1] if X[idx+1] != "-BREAK-" else "-BREAK-"
                # Add two words ahead
                #temp_dict[idx]['next_2_word'] = X[idx+2] if X[idx+1]!= "-BREAK-" else "-BREAK-"
                # Add previous word (Note that sent_index == -1 means that `i' is the first word in the sentence)
                #temp_dict[idx]['prev_1_word'] = X[idx-1] if X[idx-1] != "-BREAK-" else "-BREAK-"
                # Add two words behind
                #temp_dict[idx]['prev_2_word'] = X[idx-2] if X[idx-1] != "-BREAK-" else "-BREAK-"  
            #except:
                #pass
            # Is lowercase
            temp_dict[idx]['is_lowercase'] = bool(i.islower())
            # Is punctuation
            temp_dict[idx]['is_punct'] = bool(i in string.punctuation)
            # Is last word in sentence (If the token two positions ahead is "-BREAK-", then the token one position ahead
            # must be the punctuation that ends the sentence)
            try:
                temp_dict[idx]['is_last_word'] = bool(X[idx + 2] == "-BREAK-")
            except:
                pass
            # Is first word in sentence (note that this implementation changes in data Bethany sent)
            temp_dict[idx]['is_first_word'] = True if sent_index == 0 else False
            # Update the iterator
            sent_index += 1
    # Return the dictionary that stores features for each token
    return temp_dict

def prepare_feature_data(X_train, X_test):
    '''
    Args:
        X_train: 
            A dict that contains the features of each token of the training data
        X_test:
            A dict that contains the features of each token of the validation data
    Returns:
        feature_map:
            A dict that contains all of the unique feature-value pairs in both the training and the validation set
    '''
    # Define the feature map
    feature_map = {}
    feature_counter = 0
    for X in [X_train, X_test]:
        for token in X.keys():
            for kv_pair in X[token].items():
                if kv_pair not in feature_map.keys():
                    feature_map[kv_pair] = feature_counter
                    feature_counter += 1
    # Update the entries in X_train and X_test to have
    new_X_train = {}
    new_X_test = {}
    for tup in [(new_X_train, X_train), (new_X_test, X_test)]:
        new_X, X = tup
        for token in X.keys():
            new_X[token] = {}
            for feature in X[token].items():
                new_X[token][feature] = feature_map[feature]
    return new_X_train, new_X_test, feature_map

# Prepare training labels
# Note that the labels are just one-hot vectors which have an element equal to one in the index that corresponds to
# the true label and zero everywhere else. 
# e.g. If "NN" is at index position 3 in label_map, and there are 10 different labels, then the one-hot 
# representation of "NN" is: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def prepare_training_labels(y_train, y_test):
    '''Turns the training labels into vectors 
    '''
    # Construct a label_map dictionary that stores the unique POS tags that can exist in the dataset
    label_map = {}
    label_counter = 0
    for y in [y_train, y_test]:
        for sent_key in y:
            if sent_key not in label_map.keys():
                label_map[sent_key] = label_counter
                label_counter += 1
    new_y_train = {}
    new_y_test = {}
    for tup in [(new_y_train, y_train), (new_y_test, y_test)]:
        new_y, y = tup
        for idx, row in enumerate(y):
            if row == "-BREAK-":
                pass
            else:
                # Replace each row of the pandas series with the number corresponding to its position
                # in the label_map dictionary.
                new_y[idx] = {}
                new_y[idx]['label'] = label_map[row]
    return new_y_train, new_y_test, label_map

#==============================================================================
# Batch Generator Class
#==============================================================================
class BatchGenerator(Sequence):
    '''
    Attributes:
        mini_batch_size:
            An int containing the size of mini-batches 
        
    '''
    def __init__(self, X_dict, y_dict, feature_map, label_map, mini_batch_size):
        self.num_samples = len(X_dict)
        self.feature_map = feature_map
        self.label_map = label_map
        self.mini_batch_size = mini_batch_size
        self.mini_batches = self.prepare_mini_batches(X_dict, y_dict, mini_batch_size)
    
    def __len__(self):
        ''' Returns number of batches per epoch.'''
        return int(self.num_samples / self.mini_batch_size)
        
    def prepare_mini_batches(self, X_dict, y_dict, mini_batch_size):
        '''Generates dictionaries to organize the generation of
           the mini-batches
        Args:
            X_dict: 
                A dict containing all of the feature-value pairs for each observation
            y_dict:
                A dict containing all of the labels for each token
        Returns:
            mini_batches:
                A dict containing the information required to generate K mini-batches.
                A single mini-batch has the following format:
                
                mini_batch = {{'X_1': {'feature-value pair 1': feature_map_ID,
                                         'feature-value pair 2': feature_map_ID,
                                         ...},
                                         'y_1': label_map_label},
                                         ...
                                        {'X_M': {'feature-value pair 1': feature_map_ID,
                                                 'feature-value pair 2': feature_map_ID,
                                                 ...},
                                         'y_M': label_map_label}}
        '''
        # Generate batch dictionary
        mini_batches = {}
        lower_bound = 0
        upper_bound = mini_batch_size
        self.len = len(X_dict.keys())  # Length of the training corpus
        # batch_counter plays the role of a mini-batch "ID"
        batch_counter = 0

        for i in range(0, self.len, mini_batch_size):
            mini_batches[batch_counter] = {}
            mini_batches[batch_counter]["X"] = {k: v for k,v in X_dict.items() if k < upper_bound and k >= lower_bound}
            mini_batches[batch_counter]["y"] = {k: v for k,v in y_dict.items() if k < upper_bound and k >= lower_bound}
            lower_bound += mini_batch_size
            upper_bound += mini_batch_size
            batch_counter += 1
        
        return mini_batches
    
    def __data_generation(self, mini_batch):
        '''Generates a single mini_batch, given an id in the mini_batches dictionary.
        Args:
            mini_batch:
                A single entry in the self.mini_batches dictionary
        '''
        # Initialize empty 2D numpy arrays
        X = np.zeros((self.mini_batch_size, len(self.feature_map)))
        y = np.zeros((self.mini_batch_size, len(self.label_map)))
        # Batch timer
        #start_time = time.time()
        batch_list = []
        # Generate numpy arrays for the feature data 
        for idx, token in enumerate(mini_batch['X'].keys()):
            word_vector = [0] * len(feature_map)
            for indx in mini_batch['X'][token].keys():
                word_vector[feature_map[indx]] = 1
            X[idx,] = np.array(word_vector)
        # Generate numpy arrays for the label data now
        for idx, token in enumerate(mini_batch['y'].keys()):   
            label_vector = [0] * len(label_map)
            # Assign label to label_vector
            label_vector[mini_batch['y'][token]['label']] = 1
            y[idx,] = np.array(label_vector)
        return X, y
    
    def __getitem__(self, index):
        '''Generates one batch of data
        '''
        # Gets the data needed to generate one mini-batch
        mini_batch = self.mini_batches[index]
        X, y = self.__data_generation(mini_batch)
        return X,y
    
    def on_epoch_end(self):
        pass
#==============================================================================
# Preparing data for the model
#==============================================================================
# Load in the train/dev/test sets
train_data = import_col_data(train_path)
train_data.columns = ['token', 'part_of_speech']
dev_data = import_col_data(dev_path)
dev_data.columns = ['token', 'part_of_speech']

#train_data = train_data[:100000]
X_train = list(train_data['token'])
y_train = train_data['part_of_speech']

#dev_data = dev_data[10000:20000]
X_test = list(dev_data['token'])
y_test = dev_data['part_of_speech']

# Generate features for train/test data
X_train_dict = generate_features(X_train)
X_test_dict = generate_features(X_test)
# Re-assign the feature dictionaries values based on the feature map
X_train, X_test, feature_map = prepare_feature_data(X_train_dict, X_test_dict)
# Generate labels for train/test data
y_train, y_test, label_map = prepare_training_labels(y_train, y_test)

#==============================================================================
# Initialize batch generators
#==============================================================================
training_generator = BatchGenerator(X_train, y_train, feature_map, label_map, mini_batch_size=512)
test_generator = BatchGenerator(X_train, y_train, feature_map, label_map, mini_batch_size=512)

# Clear all other variables from memory to save space
del X_train, y_train, X_test, y_test, train_data, dev_data


#==============================================================================
# Setting up TensorBoard
#==============================================================================
NAME = 'POS-tagging-exercise-{}'.format(int(time()))

current_dir = r"" # ENTER PATH TO SAVE TENSORBOARD CALLBACK
# Change current directory
os.chdir(current_dir)
#==============================================================================
# # Setting up and training the model
#==============================================================================
sgd = SGD(lr=0.1)
model = Sequential()
model.add(Dense(4092, input_dim=len(feature_map),activation='relu'))
model.add(Dense(2048,activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(len(label_map), activation='sigmoid'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd,
             metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), write_grads=True,  
          write_graph=True, write_images=True, update_freq="batch",
          batch_size=2, histogram_freq=0)

# Note that we can't view histograms/distributions if validation data is 
# a generator (unresolved issue)
model.fit_generator(generator=training_generator, 
                    validation_data=test_generator, 
                    epochs=8, callbacks=[tensorboard])

import os
model.save(os.path.join("C:\\Data\\toy_models","pos_model_1.h5"))


os.path.dirname(os.path.realpath(__file__))