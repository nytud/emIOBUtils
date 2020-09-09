#!/usr/bin/env pyhton3
# -*- coding: utf-8, vim: expandtab:ts=4 -*-

# Usage: pyhton3 create_test_data.py > test_data.in

from itertools import permutations

# All possible labels with two groups
labels = {'B-NP', 'I-NP', 'E-NP', 'S-NP', '1-NP', 'L-NP', 'U-NP', 'NP', 'O',
          'B-VP', 'I-VP', 'E-VP', 'S-VP', '1-VP', 'L-VP', 'U-VP', 'VP'}

# Write header
print('form', 'xpos', 'NP-BIO', sep='\t')

# Combined in trigrams in a sorted order for stability
for p in sorted(permutations(labels, 3)):
    for label in p:
        print('word', 'TAG', label, sep='\t')
# Twice to wrap around the beginning and the end of the file in the middle of the data
for p in sorted(permutations(labels, 3)):
    for label in p:
        print('word', 'TAG', label, sep='\t')

print()
