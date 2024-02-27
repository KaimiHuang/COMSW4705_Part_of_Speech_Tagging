#!/usr/bin/python

"""
Implement a trigrm HMM here. 
You model should output the HMM similar to `train_hmm.py`.

Usage:  python train_trigram_hmm.py tags text > hmm-file

"""

import sys
import math, collections

def train_trigram_hmm(tag_sequences, obs_sequences):
# =============================================================================
#     tags = 'data/ptb.2-21.tgs'
#     text = 'data/ptb.2-21.txt'
# =============================================================================
    trigramCounts = collections.defaultdict(lambda: 0)
    bigramCounts = collections.defaultdict(lambda: 0)
    unigramCounts = collections.defaultdict(lambda: 0)
    total = 0

    #get trigrams, bigrams, and unigrams of the tags
    for sequence in tag_sequences:
        sequence = sequence.split()
        sequence = ['init']*2 + sequence+ ['final']
        
        for i in range(2, len(sequence)):
            trigram = (sequence[i-2], sequence[i-1], sequence[i])
            trigramCounts[trigram] = trigramCounts[trigram] + 1
        
        for i in range(1, len(sequence)):
            bigram = (sequence[i-1], sequence[i])
            bigramCounts[bigram] = bigramCounts[bigram] + 1
            
        for i in range(1, len(sequence)-1):
            unigram = sequence[i]
            unigramCounts[unigram] = unigramCounts[unigram] + 1
            total +=1
    tag_size = len(unigramCounts)
    
    #construct transition probs using add-1 smoothing                             
    for trigram in trigramCounts:
        tri_count = trigramCounts[trigram]
        tri_prev = trigram[0:2]
        tri_prev_count = bigramCounts[tri_prev]
        p = (tri_count + 1)/(tri_prev_count + tag_size)
        
        output = ['trans', trigram[0], trigram[1], trigram[2], str(p)]
        print(' '.join(output))
    
    #construct emission probs
    word_tag_counts = {}    
    for sequence_num, obs in enumerate(obs_sequences):
        obs = obs.split()
        tags = tag_sequences[sequence_num].split()
        
        for index, word in enumerate(obs):
            tag = tags[index]
            word_tag_counts[word] = word_tag_counts.get(word, {})
            word_tag_counts[word][tag] = word_tag_counts[word].get(tag, 0)
            word_tag_counts[word][tag] += 1
        
    for word in word_tag_counts:
        for tag in word_tag_counts[word]:
            word_tag_count = word_tag_counts[word][tag] + 1 #add-one smoothing
            V = len(word_tag_counts)
            tag_count = unigramCounts[tag] 
            p = word_tag_count/(tag_count + V) #add-one smoothing
            
            output = ['emit', tag, word, str(p)]
            print(' '.join(output))
    
    #how to get the prob(OOV|tag)?
    word = 'OOV'
    for tag in unigramCounts:
        tag_count = unigramCounts[tag] + V
        p = 1/(tag_count + V)
        output = ['emit', tag, word, str(p)]
        print(' '.join(output))
    


if __name__ == "__main__":
    TAGS_FILE = sys.argv[1]
    TEXT_FILE = sys.argv[2]
        
    with open(TAGS_FILE) as tags_file, open(TEXT_FILE) as text_file:
        tag_sequences = tags_file.readlines()
        obs_sequences = text_file.readlines()
    
    train_trigram_hmm(tag_sequences, obs_sequences)

    









