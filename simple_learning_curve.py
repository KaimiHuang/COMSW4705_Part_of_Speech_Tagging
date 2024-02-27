#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:41:12 2023

@author: kaimihuang
"""
#task
#get 1k training data, train bi_hmm, use dev set to evaluate the hmm. increase training data by 500 every batch, repeat
#use the development data (ptb.22.txt) to evaluate
#learning curve: x-training dataset size, range 1k-4k; y- performance measure
import sys
import tag_acc_fun
import viterbi_func
import train_hmm_fun

# =============================================================================
# bi_hmm_path = 'my.hmm'
# =============================================================================
train_tag_path = 'data/ptb.2-21.tgs'
train_text_path = 'data/ptb.2-21.txt'
dev_tags_path = 'data/ptb.22.tgs' 
dev_texts_path = 'data/ptb.22.txt' 

tag_sequences_output = 'tag_sequences_output.tgs'
text_sequences_output = 'text_sequences_output.txt'
dev_tag_sequences_output = 'dev_tag_sequences_output.tgs'
dev_text_sequences_output = 'dev_text_sequences_output.txt'

error_rate_by_word = []
error_rate_by_sent = []
train_size = []

with open (train_tag_path, 'r') as train_tag_file, open(train_text_path, 'r') as train_text_file, open(dev_tags_path, 'r') as dev_tags_file, open(dev_texts_path, 'r') as dev_texts_file:
# =============================================================================
#     train_tags = train_tag_file.readlines()
#     train_texts = train_text_file.readlines()
#     dev_tags = dev_tags_file.readlines()
#     dev_texts = dev_texts_file.readlines()
    dev_tags_sequences = [next(dev_tags_file) for _ in range(1700)]
    dev_texts_sequences = [next(dev_texts_file) for _ in range(1700)]
    
    with open(dev_tag_sequences_output, 'w') as output_file_3:
        output_file_3.writelines(dev_tags_sequences)
        
    with open(dev_text_sequences_output, 'w') as output_file_4:
        output_file_4.writelines(dev_texts_sequences)
# =============================================================================

# =============================================================================
#     for i in range(1,41):
# =============================================================================
    i = 9
    k = 1000
    num = i*k
    
    if num < 39832:
        num_of_sequences = num
    else:
        num_of_sequences = 39832
    
    #extract the first 1000 lines from the four opened files
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
        viterbi_func.viterbi('output.hmm', dev_text_sequences_output)
        sys.stdout = original_stdout
    
    #get performance by comparing predicted tag sequences against gold tag sequences, which are in the dev_tag_sequences_output.tgs file
    rates = tag_acc_fun.evalaute_tag_acc(dev_tag_sequences_output, 'output.tgs')
    train_size.append(num)
    error_rate_by_word.append(rates[0])
    error_rate_by_sent.append(rates[1])
            
        

