import pickle
import nltk
import sys
from node import Node, Rule, BackCell

# print(RuleTree(0.1, "S", "VP", "NP").print_tree())

with open('Vocabulary.txt', 'r') as f:
    vocabulary = f.read()
    vocabulary = vocabulary.split()
    # print(vocabulary)


def generate_output(choice_items, table):
    isTerminal = (choice_items.back is None)
    if isTerminal:  # Terminal.
        return '[' + choice_items.rule.parent + ' ' + choice_items.rule.left_child + ']'
    else:
        # Non-terminal.
        i1, i2 = choice_items.back.i1, choice_items.back.i2
        j1, j2 = choice_items.back.j1, choice_items.back.j2
        A1, A2 = choice_items.back.A1, choice_items.back.A2
        # No CNF modification.
        already_CNF = (choice_items.rule.parent[len(choice_items.rule.parent) - 1] != '*')
        if already_CNF:
            output = '[' + choice_items.rule.parent + ' ' + generate_output(table[i1][j1][A1],
                                                                            table) + ' ' + generate_output(
                table[i2][j2][A2], table) + ']'
        else:  # After CNF modification.
            output = generate_output(table[i1][j1][A1], table) + ' ' + generate_output(table[i2][j2][A2], table)
    return output


def CNF(parent, rhs_list, probability):
    # A -> B C D E [p]
    # B*C*D -> B C D [1]
    # B*C -> B C
    result = []
    isBC = (len(rhs_list) == 2)
    if isBC:
        result.append(Node(probability, parent, rhs_list[0], rhs_list[1]))
    else:
        B = ''
        rhs_len = len(rhs_list)
        drop_last = []
        for i in range(0, rhs_len - 1):
            drop_last.append(rhs_list[i])
        for r in drop_last:
            B = B + str(r) + '*'
        # B*C*D
        # B*C
        last_rhs = rhs_list[len(rhs_list) - 1]
        result.append(Node(probability, parent, B, last_rhs))
        # A -> B*C*D E [p]
        # B*C*D -> B*C D [1]
        isleft = (CNF(B, drop_last, 1) is not None)
        if isleft:
            # B*C*D -> B C D
            # B*C -> B C
            result.extend(CNF(B, drop_last, 1))
    return result


def CKY(words, cnf_grammar):
    for word in words:
        if word not in vocabulary:
            return None

    num_words = len(words)
    if num_words == 0:
        return None
    table = [[[] for _ in range(num_words)] for _ in range(num_words)]
    # [ [] [] [] [] [] []
    #   [] [] [] [] [] []
    #   [] [] [] [] [] []
    #   [] [] [] [] [] []
    #   [] [] [] [] [] []
    #   [] [] [] [] [] [] ]

    # Diagonal
    for idx in range(num_words):
        word = words[idx]
        for grammar in cnf_grammar:
            isword = (grammar.left_child == word)
            if isword:
                isleaf = (grammar.right_child is None)
                if isleaf:
                    table[idx][idx].append(Rule(grammar.probability, grammar))

        # Find pairs for each of the cube in the upper half table.
        for j in range(1, num_words):  # column
            for i in reversed(range(j)):  # row
                # Upper half table: always j >= i
                trace_back_list = []
                for k in range(j - i):
                    pair_A = table[i][i + k]
                    pair_B = table[i + k + 1][j]
                    for idx1, A_back in enumerate(pair_A):
                        for idx2, B_back in enumerate(pair_B):
                            for grammar in cnf_grammar:
                                # A -> BC
                                isrule = grammar.right_child is not None and \
                                         (grammar.left_child == A_back.rule.parent and
                                          grammar.right_child == B_back.rule.parent)
                                if isrule:
                                    tmp = Rule(A_back.rule.probability * B_back.rule.probability
                                               * grammar.probability, grammar,
                                               BackCell(i, i + k + 1, i + k, j, idx1, idx2))
                                    if tmp not in trace_back_list:
                                        trace_back_list.append(tmp)

                table[i][j].extend(trace_back_list)
                table[i][j] = list(set(table[i][j]))

    reference = 0
    choice_items = None
    if num_words > 0:
        for item in table[0][num_words - 1]:
            if item.rule.parent == "S" and item.rule.probability > reference:
                reference = max(item.rule.probability, reference)
            choice_items = item
        isNotSentence = (choice_items is None)
        if isNotSentence:
            print("No sentence.")
    else:
        return None
    return generate_output(choice_items, table)


def main(argv):
    TextInputFile = argv[2]
    GrammarFile = argv[3]
    TextOutputFile = argv[4]

    grammar = pickle.load(open(GrammarFile, 'rb'))

    # print(grammar.productions()[0])  # S -> Verb NP PP [0.65]
    # print(grammar.productions()[0].lhs())  # S
    # print(grammar.productions()[0].rhs())  # (Verb, NP, PP)
    # print(grammar.productions()[0].prob())  # 0.65

    old_cnf_grammar = []
    for rule in grammar.productions():
        # print(rule)
        isterminal = (type(rule.rhs()[0]) is not nltk.grammar.Nonterminal)
        if isterminal:
            tmp = Node(rule.prob(), str(rule.lhs()), str(rule.rhs()[0]))
            old_cnf_grammar.append(tmp)
        else:
            noCNF = (len(rule.rhs()) == 2)
            needCNF = (len(rule.rhs()) > 2)
            # print("noCNF is " + noCNF)
            # print("needCNF is " + needCNF)
            if noCNF:
                tmp = Node(rule.prob(), str(rule.lhs()), str(rule.rhs()[0]), str(rule.rhs()[1]))
                old_cnf_grammar.append(tmp)
            elif needCNF:
                old_cnf_grammar.extend(CNF(str(rule.lhs()), [str(item) for item in rule.rhs()], rule.prob()))

    # Delete repeated grammar rule.
    # cnf_grammar = list(set(cnf_grammar))
    delete_idx = []
    for idx1, item1 in enumerate(old_cnf_grammar):
        for idx2, item2 in enumerate(old_cnf_grammar):
            detectSame = (idx1 != idx2 and item1.parent == item2.parent and
                          item1.left_child == item2.left_child and
                          item1.right_child == item2.right_child and
                          item1.probability == item2.probability)
            if detectSame:
                delete_idx.append(max(idx1, idx2))
    cnf_grammar = []
    for i in range(len(old_cnf_grammar)):
        toKeep = (i not in set(delete_idx))
        if toKeep:
            cnf_grammar.append(old_cnf_grammar[i])

    parse_result = []

    with open(TextInputFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split()
            result = CKY(words, cnf_grammar)
            noresult = (result is None)
            if noresult:
                print("Sorry, some words are not in the vocabulary.")
                parse_result.append('')
            else:
                # print(result)
                parse_result.append(result)

    with open(TextOutputFile, 'w') as f:
        for item in parse_result:
            f.write(item)
            f.write("\n")
        print("All the results have been written into the file: " + TextOutputFile)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main(sys.argv)
