from nltk import Tree
from nltk import induce_pcfg
import pickle
from nltk import Nonterminal

terminal_dict = {}
non_terminal_dict = {}
with open("TrainingTree.txt", 'r') as f:
    lines = f.readlines()
    lines = [line.replace('[','(').replace(']',')') for line in lines]
    rules = []
    for line in lines:
        t = Tree.fromstring(line)
        rules += t.productions()
    S = Nonterminal('S')
    grammar = induce_pcfg(S, rules)
    print(grammar)

with open("grammar.pkl", 'wb') as pickle_file:
    pickle.dump(grammar, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)