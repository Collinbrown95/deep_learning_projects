#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 17:45:37 2019

@author: collinbrown
"""
import os

import numpy as np 
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from keras.utils import Progbar
from keras.initializers import RandomUniform

from keras.preprocessing.sequence import pad_sequences

# =============================================================================
# Set cwd and relative paths
# =============================================================================
ROOT_PATH = "/Users/collinbrown/Desktop/Active_Projects/deep_learning_projects/projects/ner"
DATA_PATH = "./data"
EMB_PATH = "./embeddings"
MODEL_PATH = "./model"

# Set all paths relative to root
os.chdir(ROOT_PATH)

# =============================================================================
# Functions
# =============================================================================

def read_conll_2003(filename):
    '''
    Reads data from the CoNLL-2003 dataset. Returns data in the following
    format:

    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], 
    ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], 
    ['.', 'O'] ]
    
    Args:
        filename:
            A string or filepath object that contains the path to the 
            CoNLL-2003 train/valid/test dataset
    Returns:
        sentences:
            A nested list containing (1) a list of sentences (2) A list of 
            tokens in those sentences and (3) pairs of tokens and their 
            corresponding 
    '''
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        # Grab the first and last token on the line (i.e. token-NE pairs)
        sentence.append([splits[0],splits[-1]])

    if len(sentence) >0:
        sentences.append(sentence)
        sentence = []
    return sentences

def getCasing(word, caseLookup):
    ''' For each token, categorize it as belonging to one of five "case 
        categories".
    
    Args:
        word:
            A string that represents a token.
        caseLookup:
            A python dict that maps a case category to an index that uniquely
            identifies that category.
            
    Returns:
        caseLookup[casing]:
            An int that uniquely identifies the case category of the token.
    '''
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
   
    return caseLookup[casing]
    

def createBatches(data):
    ''' 
    '''
    l = []
    for i in data:
        l.append(len(i[0]))
    # Get the set of different sentence lengths
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    # For each sentence length
    for i in l:
        # For each sentence in the data
        for batch in data:
            # If the length of the sentence is equal i, then append it to 
            # batches
            if len(batch[0]) == i:
                batches.append(batch)
                # z keeps track of the cumulative sentences with length that is
                # at least i
                z += 1
        batch_len.append(z)
    return batches,batch_len

def createMatrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx):
    '''
    
    Args:
        sentences:
            A python list with the following structure:
                [['EU', ['E', 'U'], 'B-ORG\n'], 
                 ['rejects', ['r', 'e', 'j', 'e', 'c', 't', 's'], 'O\n'], 
                 ['German', ['G', 'e', 'r', 'm', 'a', 'n'], 'B-MISC\n'], 
                 ['call', ['c', 'a', 'l', 'l'], 'O\n'], 
                 ['to', ['t', 'o'], 'O\n'], 
                 ['boycott', ['b', 'o', 'y', 'c', 'o', 't', 't'], 'O\n'], 
                 ['British', ['B', 'r', 'i', 't', 'i', 's', 'h'], 'B-MISC\n'], 
                 ['lamb', ['l', 'a', 'm', 'b'], 'O\n'], 
                 ['.', ['.'], 'O\n']]
            At this point in the program, sentences should be tokenized such
            that the first inner list element is the original token, the next
            inner list element should be the individual characters that
            comprise the first token, and the third element should be the NER
            tag corresponding to that token. 
        word2Idx:
            A python dict that maps each token contained in the training, 
            development, and test datasets to a unique integer. 
        label2Idx:
            A python dict that maps each NER label conatined in the training,
            development, and test datasets to a unique integer. 
        case2Idx:
            A python dict that maps each case label to a numeric index. Case
            labels include: {'numeric': 0, 'allLower': 1, 'allUpper': 2, 
                             'initialUpper': 3, 'other': 4, 
                             'mainly_numeric': 5, 'contains_digit': 6,
                             'PADDING_TOKEN': 7}
        char2Idx:
            A python dict that maps the ASCII characters contained in the text 
            to a unique integer. Note that the "UNKNOWN" token is used to 
            indicate an ASCII character that is not contained in the index. 
            
    Returns:
        dataset:
            A python list where each entry represents a sentence and has the 
            following structure:
                [[641, 6732, 512, 578, 6, 4940, 295, 8353, 4],
                 [2, 1, 3, 1, 1, 1, 3, 1, 4],
                 [[43, 59],
                  [30, 17, 22, 17, 15, 32, 31],
                  [45, 17, 30, 25, 13, 26],
                  [15, 13, 24, 24],
                  [32, 27],
                  [14, 27, 37, 15, 27, 32, 32],
                  [40, 30, 21, 32, 21, 31, 20],
                  [24, 13, 25, 14],
                  [65]],
                 [4, 2, 3, 2, 2, 2, 3, 2, 2]]
                The first list represents the index of each unique token. The
                second list represents the various case index associated with
                each word. The third list is a nested list with the same number
                of elements as there are tokens in the sentence; each inner
                list represents the character index associated with each token.
                The fourth list represents the label index associated with each
                token.
                            
    '''
    # Unknown index
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    # Initialize dataset as empty list
    dataset = []
    # Initialize counts for words and unknown words
    wordCount = 0
    unknownWordCount = 0
    for sentence in sentences:
        # For each sentence, initialize empty lists for each of the below
        wordIndices = []    
        caseIndices = []
        charIndices = []
        labelIndices = []
        # Each sentence has three elements: (1) tokens (2) character lists and
        # (3) NER labels. 
        for word, char, label in sentence:  
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            # For each word, initialize an empty character index.
            charIdx = []
            for x in char:
                charIdx.append(char2Idx[x])
            #Get the label and map to int            
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx))
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])
        # Append each sentence list to the dataset
        dataset.append([wordIndices, caseIndices, charIndices, labelIndices]) 
        
    return dataset

def iterate_minibatches(dataset,batch_len):
    ''' A generator for the batches generated by createBatches().
    
    Args:
        dataset:
            A dataset in the format of what is output by createBatches().
        batch_len:
            
    
    Yields:
        A 4-tuple of numpy arrays containing the indices for (1) labels 
        (2) tokens (3) case-ing and (4) characters.
        
    '''
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t,c,ch,l = dt
            l = np.expand_dims(l,-1)
            tokens.append(t)
            caseing.append(c)
            char.append(ch)
            labels.append(l)
        yield (np.asarray(labels),
               np.asarray(tokens),
               np.asarray(caseing),
               np.asarray(char))

def addCharInformation(Sentences):
    '''
    '''
    for i,sentence in enumerate(Sentences):
        for j,data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0],chars,data[1]]
    return Sentences

def padding(sentences):
    ''' 
    Adds up to 52 zeros of padding to the list of character indices that
    represents each token.
    
    Args:
        sentences:
            A python list where each entry represents a sentence and has the 
            following structure:
                [[641, 6732, 512, 578, 6, 4940, 295, 8353, 4],
                 [2, 1, 3, 1, 1, 1, 3, 1, 4],
                 [[43, 59],
                  [30, 17, 22, 17, 15, 32, 31],
                  [45, 17, 30, 25, 13, 26],
                  [15, 13, 24, 24],
                  [32, 27],
                  [14, 27, 37, 15, 27, 32, 32],
                  [40, 30, 21, 32, 21, 31, 20],
                  [24, 13, 25, 14],
                  [65]],
                 [4, 2, 3, 2, 2, 2, 3, 2, 2]]
                The first list represents the index of each unique token. The
                second list represents the various case index associated with
                each word. The third list is a nested list with the same number
                of elements as there are tokens in the sentence; each inner
                list represents the character index associated with each token.
                The fourth list represents the label index associated with each
                token.
    
    Returns:
        sentences:
            A python list that has the same format as the input sentences,
        except now the third inner-list how contains up to 52 zeros of padding
        and is represented as a large numpy ndarray. E.g.
        
        [[43, 59],
        [30, 17, 22, 17, 15, 32, 31]]
        
        Turns into
        
        array([[43, 59,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
                0,  0,  0,  0],
        [30, 17, 22, 17, 15, 32, 31,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0]])
           
    '''
    maxlen = 52
    for sentence in sentences:
        char = sentence[2]
        for x in char:
            maxlen = max(maxlen,len(x))
    for i,sentence in enumerate(sentences):
        # Use Keras' pad_sequences function
        sentences[i][2] = pad_sequences(sentences[i][2],52,padding='post')
    return sentences

def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing,char], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels
#Method to compute the accruarcy. Call predict_labels to get the labels for the dataset
def compute_f1(predictions, correct, idx2Label): 
    label_pred = []    
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])
        
    label_correct = []    
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])
            
    
    #print label_pred
    #print label_correct
    
    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)
    
    f1 = 0
    if (rec+prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);
        
    return prec, rec, f1

def compute_precision(guessed_sentences, correct_sentences):
    assert(len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0
    
    
    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert(len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B': #A new chunk starts
                count += 1
                
                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True
                    
                    while idx < len(guessed) and guessed[idx][0] == 'I': #Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False
                        
                        idx += 1
                    
                    if idx < len(guessed):
                        if correct[idx][0] == 'I': #The chunk in correct was longer
                            correctlyFound = False
                        
                    
                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:  
                idx += 1
    
    precision = 0
    if count > 0:    
        precision = float(correctCount) / count
        
    return precision

# =============================================================================
# Parameters
# =============================================================================
epochs = 2
# =============================================================================
# Code
# =============================================================================
trainSentences = read_conll_2003(os.path.join(DATA_PATH, "train.txt"))
devSentences = read_conll_2003(os.path.join(DATA_PATH, "valid.txt"))
testSentences = read_conll_2003(os.path.join(DATA_PATH, "test.txt"))


# Break each token into its individual characters:
# E.g. ['rejects', ['r', 'e', 'j', 'e', 'c', 't', 's'], 'O\n']
trainSentences = addCharInformation(trainSentences)
devSentences = addCharInformation(devSentences)
testSentences = addCharInformation(testSentences)

# Identify all of the unique tokens (words) contained in the text of each
# dataset, as well as all of the unique NER labels that each word could take
# on. 
labelSet = set()
words = {}

for dataset in [trainSentences, devSentences, testSentences]:
    for sentence in dataset:
        for token,char,label in sentence:
            labelSet.add(label)
            words[token.lower()] = True

# Create a map between the labels and a numeric index.
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)

# Numeric index corresponding to case lookup value
case2Idx = {'numeric': 0, 
            'allLower':1, 
            'allUpper':2, 
            'initialUpper':3, 
            'other':4, 
            'mainly_numeric':5, 
            'contains_digit': 6, 
            'PADDING_TOKEN':7}

# Represent these case embeddings as one-hot encoded vectors.
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')

# Read specified word embeddings from disk - load only those that are contained
# in the corpus into memory.
word2Idx = {}
wordEmbeddings = []

fEmbeddings = open(os.path.join(EMB_PATH, "glove.6B.50d.txt"), encoding="utf-8")

for line in fEmbeddings:
    split = line.strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
        
wordEmbeddings = np.array(wordEmbeddings)
# Map each unique character to an integer
char2Idx = {"PADDING":0, 
            "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    # Give each new character a unique integer
    char2Idx[c] = len(char2Idx)
    
print(trainSentences[0])

train_set = padding(createMatrices(trainSentences,
                                   word2Idx,  
                                   label2Idx, 
                                   case2Idx,
                                   char2Idx))
dev_set = padding(createMatrices(devSentences,
                                 word2Idx, 
                                 label2Idx, 
                                 case2Idx,
                                 char2Idx))
test_set = padding(createMatrices(testSentences, 
                                  word2Idx, 
                                  label2Idx, 
                                  case2Idx,
                                  char2Idx))

idx2Label = {v: k for k, v in label2Idx.items()}

# Create batches and get cumulative number of sentences with a particular 
# batch length
train_batch,train_batch_len = createBatches(train_set)
dev_batch,dev_batch_len = createBatches(dev_set)
test_batch,test_batch_len = createBatches(test_set)


words_input = Input(shape=(None,), dtype='int32', name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0],
                  output_dim=wordEmbeddings.shape[1],
                  weights=[wordEmbeddings], 
                  trainable=True)(words_input)
casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
casing = Embedding(output_dim=caseEmbeddings.shape[1], 
                   input_dim=caseEmbeddings.shape[0], 
                   weights=[caseEmbeddings], 
                   trainable=True)(casing_input)
character_input=Input(shape=(None,52,),name='char_input')
embed_char_out=TimeDistributed(Embedding(len(char2Idx),
                                         30,
                                         embeddings_initializer=RandomUniform(
                                                 minval=-0.5,
                                                 maxval=0.5)), 
                                         name='char_embedding')(character_input)
dropout= Dropout(0.5)(embed_char_out)
conv1d_out= TimeDistributed(Conv1D(kernel_size=3, 
                                   filters=30, 
                                   padding='same',
                                   activation='tanh', 
                                   strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)
output = concatenate([words, casing,char])
output = Bidirectional(LSTM(200, 
                            return_sequences=True, 
                            dropout=0.50, 
                            recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
model = Model(inputs=[words_input, casing_input,character_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()
# plot_model(model, to_file='model.png')


# Training model
for epoch in range(epochs):    
    print("Epoch %d/%d"%(epoch+1,epochs))
    a = Progbar(len(train_batch_len))
    for i, batch in enumerate(iterate_minibatches(train_batch,
                                                  train_batch_len)):
        labels, tokens, casing,char = batch       
        model.train_on_batch([tokens, casing,char], labels)
        a.update(i)
    # Test performance for current epoch
    predLabels, correctLabels = tag_dataset(dev_batch)        
    pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
    print(" ")
    print("Dev-Data, Epoch %d/%d: Prec: %.3f, Rec: %.3f, F1: %.3f" % (epoch+1, 
                                                                      epochs,
                                                                      pre_dev, 
                                                                      rec_dev, 
                                                                      f1_dev))
    print('*'*40)

#   Performance on dev dataset        
predLabels, correctLabels = tag_dataset(dev_batch)        
pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
print("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))
    
#   Performance on test dataset       
predLabels, correctLabels = tag_dataset(test_batch)        
pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))