import pickle
import sys
import numpy as np


def viterbi(A, B, pi, obs):
    obs_v = np.array(obs)
    v = np.array([[float(0) for _ in range(obs_v.shape[0])] for _ in range(A.shape[0])], dtype='float64')
    for t in range(A.shape[0]):
        v[t][0] = pi[t] + B[t][obs_v[0]]  # log version
    for j in range(0, A.shape[0]):
        for t in range(1, obs_v.shape[0]):
            max_one = v[0][t - 1] + A[0][j]
            for i in range(1, A.shape[0]):
                max_one = max(max_one, v[i][t - 1] + A[i][j])
            v[j][t] = max_one + B[j][obs_v[t]]
    return v.transpose().argmax(axis=1)


def main(argv):
    model = open('model.pyc', 'rb')
    A, B, pi, words_dict, tags_dict = pickle.load(model)

    testing_file = argv[1]
    truth_file = argv[2]
    # testing_file = "wsj19-21.testing"
    # truth_file = "wsj19-21.truth"

    truth_tags = []
    for line in open(truth_file):
        words_a_line = line.strip('\n').split()
        for tag in words_a_line[1::2]:
            truth_tags.append(tags_dict[tag])
    truth_tags = np.array(truth_tags)

    test_tags = []
    for line in open(testing_file):
        words_a_line = line.strip('\n').split()
        obs = []
        for word in words_a_line:
            if word not in words_dict:
                obs.append(0)
            else:
                obs.append(words_dict[word])
        v = viterbi(A, B, pi, obs)
        test_tags = np.array(np.concatenate((test_tags, v)))

    equal_sum = 0
    for i in range(0, len(test_tags)):
        if test_tags[i] == truth_tags[i]:
            equal_sum += 1

    accuracy = equal_sum / test_tags.shape[0]
    print("The accuracy is %.2f%%." % (accuracy * 100))
    return


if __name__ == "__main__":
    main(sys.argv)
