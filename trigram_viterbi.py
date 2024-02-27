#!/usr/bin/python

"""
Implement the trigram Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.

Usage:  python trigram_viterbi.py hmm-file < text > tags

special keywords:
 $init_state   (an HMM state) is the single, silent start state
 $final_state  (an HMM state) is the single, silent stop state
 $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word
"""

import sys
import math

def trigram_viterbi():

    A = {}
    B = {}
    vocab = {} 
    states_i = {}
    states_j = {}
    
    #hmmfile = 'my_trigram.hmm'
    with open(sys.argv[1], 'r') as f: #
    # Read in the HMM and store the probabilities as log probabilities
        for line in f:
            if line.startswith('trans'):
                line_split = line.strip().split()
                prev_state = tuple(line_split[1:3])
                state = line_split[3]
                p = line_split[4]
                A[prev_state] = A.get(prev_state, {})
                A[prev_state][state] = math.log(float(p))
                states_i[prev_state] = states_i.get(prev_state, 0)
                states_i[prev_state] += 1
                #states_j [state] = 1
            
            elif line.startswith('emit'):
                matrix, state, word, p = line.strip().split()
                B[state] = B.get(state,{})
                B[state][word] = math.log(float(p))
                states_j[state] = 1
                vocab[word] = 1

    data = []
    for sentence in sys.stdin:
        data.append(sentence)
    for sentence in data:       
# =============================================================================
#     text_file = 'data/ptb.22.txt'
#     with open(text_file, 'r') as f:
#         for sentence in f:
# =============================================================================
        #print(sentence)
        w = sentence.split()
        w = [''] + w
        V = {}
        Bt = {}
        V[0] = {('init', 'init'): 0.0}
        
        for t in range(1, len(w)+1):
            if t < len(w):
                word = w[t]
            
                if word not in vocab:
                    word = 'OOV'
                            
                V[t] = {}
                Bt[t] = {}
            
                for prev_state in V[t-1]:
                    for state in B:                
                        if word in B[state] and prev_state in A and state in A[prev_state]:
                            
                            v_prev = V[t-1][prev_state]
                            a_ij = A[prev_state][state] 
                            b_j_wt = B[state][word]
                            v =  v_prev + a_ij + b_j_wt
                            
                            state_j = tuple([prev_state[1], state])
                            if state_j not in V[t] or v > V[t][state_j]:
                                V[t][state_j] = v
                                Bt[t][state_j] = prev_state  
                                
                        elif word in B[state] and prev_state in A and state not in A[prev_state]:
                            
                            v_prev = V[t-1][prev_state]
                            
                            #add-1 smoothed trigram.
                            a_ij = math.log(1/(states_i[prev_state] + len(states_j)))
                            b_j_wt = B[state][word]
                            v =  v_prev + a_ij + b_j_wt
                            
                            state_j = tuple([prev_state[1], state])
                            
                            if state_j not in V[t] or v > V[t][state_j]:
                                V[t][state_j] = v
                                Bt[t][state_j] = prev_state
        
            elif t == len(w):
                V[t] = {}
                Bt[t] = {}
                
                for prev_state in V[t-1]:
                    state = 'final'
                    
                    if prev_state not in A:
                            continue
                    
                    elif prev_state in A and state in A[prev_state]:                
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
            state = Bt[len(Bt)]['final'][:]
            tags.append(state[-1])
            tags.append(state[0])
            for i in range(len(Bt)-1, 1, -1):
                state = Bt[i][state]
                tag = state[0]
                tags.append(tag)
            tags = tags[::-1][1:]
            print(' '.join(tags))

if __name__ == "__main__":
    trigram_viterbi()

