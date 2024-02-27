#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:27:31 2023

@author: kaimihuang
"""

import re
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings


#create a dataset of labeled sentences. organize the data in CoNLL format
train_tag_path = 'data/ptb.2-21.tgs'
train_text_path = 'data/ptb.2-21.txt'

dev_tags_path = 'data/ptb.22.tgs' 
dev_texts_path = 'data/ptb.22.txt' 

conll_lines = []
with open('conll/train.conll', 'w', encoding='utf-8') as conll_file:
    with open (train_tag_path, 'r') as train_tag_file, open(train_text_path, 'r') as train_text_file:
        for tagString, tokenString in zip(train_tag_file, train_text_file):
            tags = re.split("\s+", tagString.rstrip())
            tokens = re.split("\s+", tokenString.rstrip())
            for i in range(len(tags)):        
                conll_line = tokens[i] + " " + tags[i] + '\n'
                if conll_line not in conll_lines:
                    conll_lines.append(conll_line)
                    
    conll_file.writelines(conll_lines)             

with open('conll/dev.conll', 'w', encoding='utf-8') as conll_file:
    with open (dev_tags_path, 'r') as dev_tag_file, open(dev_texts_path, 'r') as dev_text_file:
        for tagString, tokenString in zip(dev_tag_file, dev_text_file):
            tags = re.split("\s+", tagString.rstrip())
            tokens = re.split("\s+", tokenString.rstrip())
            for i in range(len(tags)):        
                conll_line = tokens[i] + " " + tags[i] + '\n'
                if conll_line not in conll_lines:
                    conll_lines.append(conll_line)
                    
    conll_file.writelines(conll_lines)
    
#1. load the corpus (Ontonotes does not ship with Flair, you need to download and reformat into a column format yourself)              
data_folder = 'conll'
column_format = {0: 'text', 1: 'pos'}
corpus: Corpus = ColumnCorpus(data_folder, column_format, 
                              train_file='train.conll',
                              test_file = 'dev.conll')

# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# 4. initialize each embedding we use
embedding_types = [

    # contextual string embeddings, forward
    FlairEmbeddings('news-forward'),

    # contextual string embeddings, backward
    FlairEmbeddings('news-backward'),
]

# embedding stack consists of Flair and GloVe embeddings
embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=tag_dictionary,
                        tag_type=tag_type)

# 6. initialize trainer
from flair.trainers import ModelTrainer
trainer = ModelTrainer(tagger, corpus)

# 7. run training
trainer.train('bonus_output',
              train_with_dev=False,
              train_with_test=True,
              max_epochs=150,
              mini_batch_size=16)


# load tagger
tagger = SequenceTagger.load("flair/pos-english")

# make example sentence
sentence = Sentence("I love bunnies.")

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('The following NER tags are found:')
# iterate over entities and print
for entity in sentence.get_spans('pos'):
    print(entity)