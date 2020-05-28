class Node:
    def __init__(self, probability, parent, left_child, right_child=None):
        self.probability = probability
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        # self.traceback = traceback

    # def print_tree(self):
    #     if self.right_child is not None:
    #         return str(self.parent) + " -> " + str(self.left_child) + " " + str(self.right_child) + " [" + str(self.probability) + "]"
    #     else:
    #         return str(self.parent) + " -> " + str(self.left_child) + " [" + str(self.probability) + "]"

    def __repr__(self):
        if self.right_child is not None:
            return str(self.parent) + " -> " + str(self.left_child) + " " + str(self.right_child) + " [" + str(
                self.probability) + "]"
        else:
            return str(self.parent) + " -> " + str(self.left_child) + " [" + str(self.probability) + "]"


class Rule:
    def __init__(self, probability, rule, back=None):
        self.prob = probability
        self.rule = rule
        self.back = back


class BackCell:
    def __init__(self, i1, i2, j1, j2, A1, A2):
        # i represents the row
        # j represents the column
        # A represents the index of the parent
        self.i1, self.i2 = i1, i2
        self.j1, self.j2 = j1, j2
        self.A1, self.A2 = A1, A2

