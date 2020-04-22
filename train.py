import pickle
import math
import numpy as np
from collections import Counter


def train():
    file = open("wsj1-18.training", "r")
    mixed_words = file.read().strip('\n').split()
    words = []
    tags = []

    for i in range(0, len(mixed_words)):
        if i % 2 == 0:
            words.append(mixed_words[i])
        elif i % 2 == 1:
            tags.append(mixed_words[i])

    cnt = Counter(words)
    words = set(words)
    for key,val in cnt.items():
        if val < 3:
            words.remove(key)
    words = list(words)
    tags = list(set(tags))

    words_dict = {}
    tags_dict = {}
    words_dict['UNKA'] = 0

    for i in range(0, len(words)):
        words_dict[words[i]] = i + 1
    for i in range(0, len(tags)):
        tags_dict[tags[i]] = i

    num_words = len(words_dict)
    num_tags = len(tags_dict)
    A = np.array([[0 for _ in range(num_tags)] for _ in range(num_tags)], dtype='float64')
    B = np.array([[0 for _ in range(num_words)] for _ in range(num_tags)], dtype='float64')
    pi = np.array([0 for _ in range(num_tags)], dtype='float64')
    N = np.array([0 for _ in range(num_tags)], dtype='float64')  # Count numbers.

    # compute Transition Matrix A, B and pi
    for line in open("wsj1-18.training"):
        all_words_line = line.split()
        i = 0
        while i < (len(all_words_line) - 1):
            word = all_words_line[i]
            if word in words_dict:
                word_index = words_dict[word]
            else:
                word_index = 0  # Unknown words.

            tag = all_words_line[i + 1]
            N[tags_dict[tag]] += 1
            if i == 0:
                pi[tags_dict[tag]] += 1
            else:
                row = tags_dict[all_words_line[i - 1]]
                col = tags_dict[tag]
                A[row][col] += 1
            col = tags_dict[tag]
            B[col][word_index] += 1
            i += 2

    # turn A, B and pi from counts to probabilities
    for i in range(0, num_tags):
        A[i] = np.log(A[i]) - np.log(sum(A[i]))
    for i in range(0, num_tags):
        B[i] = np.log(B[i]) - np.log(sum(B[i]))
    pi = np.log(pi) - np.log(sum(pi))

    # smoothing
    # N = np.log(N) - np.log(sum(N))
    # S = np.array([0 for _ in range(0, num_tags)], dtype='float64')
    # for i in range(0, num_tags):
    #     S[i] = np.log(1 - np.exp(sum(A[i]))) - np.log(sum(np.where(A[i] == float('-inf'), N, 0)))
    # for i in range(0, num_tags):
    #     A[i] = np.where(A[i] == 0, S[i] + N, A[i])
    #
    # S_prime = np.log(1 - np.exp(sum(pi))) - np.log(sum(np.where(pi == float('-inf'), N, 0)))
    # pi = np.where(pi == 0, S_prime + N, pi)

    return A, B, pi, words_dict, tags_dict


A, B, pi, words_dict, tags_dict = train()
model = open("model.pyc", "wb")
pickle.dump((A, B, pi, words_dict, tags_dict), model)
