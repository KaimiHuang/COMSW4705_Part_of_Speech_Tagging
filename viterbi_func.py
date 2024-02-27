#!/usr/bin/python

"""
Implement the Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.
Please check `viterbi.pl` for reference.

Usage:  python viterbi.py hmm-file < text > tags

special keywords:
 $init_state   (an HMM state) is the single, silent start state
 $final_state  (an HMM state) is the single, silent stop state
 $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word
"""

import sys
import math

def viterbi(hmmfile, textfile):

    A = {}
    B = {}
    vocab = {} 
    states = {}
    vocab_str = str()
    
    #hmmfile = 'my.hmm'
    with open(hmmfile, 'r') as f:
    # Read in the HMM and store the probabilities as log probabilities
        for line in f:
            if line.startswith('trans'):
                matrix, prev_state, state, p = line.strip().split()
                A[prev_state] = A.get(prev_state, {})
                A[prev_state][state] = math.log(float(p))
                states[prev_state] = 1
                states[state] = 1
                
            elif line.startswith('emit'):
                matrix, state, word, p = line.strip().split()
                B[state] = B.get(state,{})
                B[state][word] = math.log(float(p))
                states[state] = 1
                vocab[word] = 1
                if word not in vocab_str:
                    vocab_str = vocab_str + " " + word

    output_file_path = "vocab.txt"
    with open(output_file_path, 'w') as output_file:
        output_file.write(vocab_str)
# =============================================================================
#     data = []
#     for sentence in sys.stdin:
#         data.append(sentence)
# =============================================================================
    with open(textfile, 'r') as data:
        for sentence in data:       
            #print(sentence)
            w = sentence.split()
            w = [''] + w
            V = {}
            Bt = {}
            V[0] = {'init': 0.0}
            
            for t in range(1, len(w)+1):
                if t < len(w):
                    word = w[t]
                
                    if word not in vocab:
                        word = 'OOV'
                                
                    V[t] = {}
                    Bt[t] = {}
                
                    for prev_state in V[t-1]:
                        for state in B:                
                            if word in B[state] and state in A[prev_state]:
                                
                                v_prev = V[t-1][prev_state]
                                a_ij = A[prev_state][state] 
                                b_j_wt = B[state][word]
                                
                                v =  v_prev + a_ij + b_j_wt
                                if state not in V[t] or v > V[t][state]:
                                    V[t][state] = v
                                    Bt[t][state] = prev_state        
                elif t == len(w):
                    V[t] = {}
                    Bt[t] = {}
                    
                    for prev_state in V[t-1]:
                        state = 'final'
                        
                        if state in A[prev_state]:                
                            v_prev = V[t-1][prev_state]
                            a_ij = A[prev_state][state]
                            v =  v_prev + a_ij 
                            if state not in V[t] or v > V[t][state]:
                                V[t][state] = v
                                Bt[t][state] = prev_state                                           
            tags = []
            if 'final' not in Bt[len(Bt)]:
                print(' ')
            else:
                tag = Bt[len(Bt)]['final']
                tags.append(tag)
                for i in range(len(Bt)-1, 0, -1):
                    tag = Bt[i][tag]
                    tags.append(tag)
                tags = tags[::-1][1:]
                print(' '.join(tags))


     
test = viterbi('my.hmm', 'data/ptb.22.txt' )

