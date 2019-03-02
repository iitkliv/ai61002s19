from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def normalizeWindowsString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def randomTrainingExample(pairs, labels):
    rand_key = random.choice(list(pairs))
    return pairs[rand_key], labels[rand_key]


MAX_RAW_LENGTH = 30
MAX_RNN_LENGTH = 10
all_categories = []


def stripWords2MaxLen(sent):
    allwords = sent.split(' ')
    if(len(allwords) < MAX_RNN_LENGTH):
        return sent
    sent = ' '.join(allwords[i] for i in range(MAX_RNN_LENGTH-1))
    #print(sent)
    return sent

def filterSentencePair(sent0, sent1):
    return len(sent0.split(' ')) < MAX_RAW_LENGTH and \
        len(sent1.split(' ')) < MAX_RAW_LENGTH 

def readClassificationData(lang1):
    print("Reading Lines...")

    # Read the file and split into lines
    #with open('data/dev_w_id.txt', 'rb') as f:
    #  lines = f.readlines()#.strip().split('\n')
    #print(lines[0])
    #lines = open('data/dev_w_id.txt').read().strip().split('\n')
    lines = open('data/train_w_id.txt', encoding='windows-1252').readlines()
    print(len(lines))
    
    input_lang = Lang(lang1)
    pairs = {}
    labels = {}
    # Split every line into id, query, passage, label
    for l in lines:
        [id, sent0, sent1, label] = l.strip().replace('\n','').split('\t')
        
        if(filterSentencePair(sent0, sent1)):
            sent0 = stripWords2MaxLen(normalizeString(sent0))
            sent1 = stripWords2MaxLen(normalizeString(sent1))
            #print([id, normalizeWindowsString(sent0), normalizeWindowsString(sent1), label])
            pairs[id] = ([(sent0), (sent1)])
            labels[id] = (label)
            if label not in all_categories:
                all_categories.append((label))
            input_lang.addSentence((sent0))
            input_lang.addSentence((sent1))
    return input_lang, pairs, labels

def prepareClassificationData(lang1):
    input_lang, pairs, labels = readClassificationData(lang1)
    if( len(pairs) != len(labels)):
        print("Number of sentence pairs is not equal to label count")
        # exit(0)
    print("Read %s sentence pairs and labels" % len(pairs))

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    return input_lang, pairs, labels
