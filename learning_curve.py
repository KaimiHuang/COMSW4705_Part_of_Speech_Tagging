#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:41:12 2023

@author: kaimihuang
"""
#get 1k training data, train bi_hmm, use dev set to evaluate the hmm. increase training data by 1000 every batch, repeat
#use the development data (ptb.22.txt) to evaluate
#learning curve: x-training dataset size, range 1k-4k; y- performance measure

import sys
import tag_acc_fun
import viterbi_func
import train_hmm_fun
import matplotlib.pyplot as plt

train_tag_path = 'data/ptb.2-21.tgs'
train_text_path = 'data/ptb.2-21.txt'
dev_tags_path = 'data/ptb.22.tgs' 
dev_texts_path = 'data/ptb.22.txt' 

tag_sequences_output = 'tag_sequences_output.tgs'
text_sequences_output = 'text_sequences_output.txt'

error_rate_by_word = []
error_rate_by_sent = []
train_size = []


for i in range(1,41):
    
    k = 1000
    num = i*k
    
    if num < 39832:
        num_of_sequences = num
    else:
        num_of_sequences = 39832
    
    #extract the first i*k lines from the four opened files
    with open (train_tag_path, 'r') as train_tag_file, open(train_text_path, 'r') as train_text_file:
        tag_sequences = [next(train_tag_file) for _ in range(num_of_sequences)]
        text_sequences = [next(train_text_file) for _ in range(num_of_sequences)]

    #write them to separate new output files
    with open(tag_sequences_output, 'w') as output_file_1:
        output_file_1.writelines(tag_sequences)
    
    with open(text_sequences_output, 'w') as output_file_2:
        output_file_2.writelines(text_sequences)
        
    #train bigram hmm using the training tag dataset and training text dataset. save output hmm to a new file
    with open('output.hmm', 'w') as output_file_5:
        original_stdout = sys.stdout
        sys.stdout = output_file_5
        train_hmm_fun.train_hmm(tag_sequences_output, text_sequences_output)
        sys.stdout = original_stdout
    
    #predict tag sequences for the dev text set using the trained bigram hmm and viterbi algorithm. save predicted tag sequences to a new file
    with open('output.tgs', 'w') as output_file_6:
        original_stdout = sys.stdout
        sys.stdout = output_file_6
        viterbi_func.viterbi('output.hmm', dev_texts_path)
        sys.stdout = original_stdout
    
    #get performance by comparing predicted tag sequences against gold tag sequences
    rates = tag_acc_fun.evalaute_tag_acc(dev_tags_path, 'output.tgs')
    train_size.append(num)
    error_rate_by_word.append(rates[0])
    error_rate_by_sent.append(rates[1])        

plt.plot(error_rate_by_word, label = 'Error Rate by Word') # line plot
plt.plot(error_rate_by_sent, label = 'Error Rate by Sentence') # line plot
plt.xlabel('Training Dataset Size (unit: thousand sentences)', fontsize=13)
plt.ylabel('Error Rate', fontsize=13)
plt.title('Bigram HMM Performance vs Training Dataset Size', fontsize=15)
plt.legend(fontsize=12)
plt.show()