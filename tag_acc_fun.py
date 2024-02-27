#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 23:18:09 2023

@author: kaimihuang
"""

#!/usr/bin/python

"""
Calculates and prints out error rate (word-level and sentence-level) of a POS tagger.

Usage: python tag_acc.py gold-tags hypothesized-tags

Tags should be separated by whitespace, no leading or trailing spaces,
one sentence per line.  There's no error handling if things don't line up!
"""


import re
import sys


def evalaute_tag_acc(GOLD_FILE, HYPO_FILE):
    
    with open(GOLD_FILE) as goldFile, open(HYPO_FILE) as hypoFile:
            golds = goldFile.readlines()
            hypos = hypoFile.readlines()
            
    tag_errors = 0
    sent_errors = 0
    tag_tot = 0
    sent_tot = 0

    for g, h in zip(golds, hypos):
        g = g.strip()
        h = h.strip()

        g_toks = re.split("\s+", g)
        h_toks = re.split("\s+", h)

        error_flag = False

        for i in range(len(g_toks)):
            if i >= len(h_toks) or g_toks[i] != h_toks[i]:
                tag_errors += 1
                error_flag = True

            tag_tot += 1

        if error_flag:
            sent_errors += 1

        sent_tot += 1
        
    error_rate_by_word = tag_errors / tag_tot
    error_rate_by_sent = sent_errors / sent_tot
    return error_rate_by_word, error_rate_by_sent

# =============================================================================
# 
# if __name__ == "__main__":
#     # arguemnt
#     GOLD_FILE = sys.argv[1]
#     HYPO_FILE = sys.argv[2]
# 
#     with open(GOLD_FILE) as goldFile, open(HYPO_FILE) as hypoFile:
#         golds = goldFile.readlines()
#         hypos = hypoFile.readlines()
# 
#         if len(golds) != len(hypos):
#             raise ValueError("Length is different for two files!")
# 
#     evalaute_tag_acc(golds, hypos)
# 
# =============================================================================
