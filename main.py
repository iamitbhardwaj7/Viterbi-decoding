"""
6A Assignment
Name-Amit Kumar
Roll no. 137


This program reads data from train.txt and uses the data to predict 
the possible POS tag sequence for the strings in test.txt
"""
from collections import namedtuple
import numpy as np
from tabulate import tabulate

VERBOSE = False

class ViterbiDecoding:
    """
    Class to computer model parameters
        - pi -> initial probabilites
        - A -> Transition probabilities
        - B -> Emission probabilities
    """
    def __init__(self, input_file_name):
        self.tokens = set()
        self.poss = set()
        self.lines = self._process_file(input_file_name)
        self.pi = np.zeros((len(self.poss)))
        self.A = np.zeros((len(self.poss), len(self.poss)))
        self.B = np.zeros((len(self.tokens), len(self.poss)))
        self.tokens = list(self.tokens)
        self.poss = list(self.poss)
        self._process_data()

        # print(self.poss)
        # print(self.pi)
        # print(self.A)
        # print(self.B)
        if VERBOSE:
            self.print_parameters()    

    def _process_file(self, filename):
        file = open(filename, mode='r', encoding='utf-8')
        lines = []
        # TODO: Process tokens and pos tag pairs
        pair = namedtuple('TokenPOS', 'token pos')
        for line in file.readlines():
            line = line.replace('\n','')
            new_line = []
            for word in line.split(' '):
                token, pos = word.split('/')
                self.tokens.add(token)
                self.poss.add(pos)
                new_line.append(pair(token=token, pos=pos))
            lines.append(new_line)
        file.close()
        return lines

    def _process_data(self):
        """
        Groups the data 
        """
        for line in self.lines:
            # TODO: Update pi
            self.pi[self.poss.index(line[0].pos)] += 1
            
            # TODO: Update Transition Prob. wrt to POS tag
            for i in range(1, len(line)):
                self.A[self.poss.index(line[i-1].pos), self.poss.index(line[i].pos)] += 1
            
            # TODO: Update Emission Prob.
            for token, pos in line:
                self.B[self.tokens.index(token), self.poss.index(pos)] += 1
        
        # TODO: normalize dictionary values
        self.pi /= np.sum(self.pi, axis = 0)
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        self.B /= np.sum(self.B, axis=0, keepdims=True)

    def print_parameters(self, tokens_idx = None):
        pos_tags = self.poss
        if tokens_idx == None:
            tokens_idx = np.arange(len(self.tokens))
        tokens = [self.tokens[i] for i in tokens_idx]
        print("-----Initial Parameters-----")
        print(tabulate([list(self.pi)], headers=pos_tags, tablefmt="grid"))
        
        print("\n-----Transition Probabilities-----")
        A = [[label] + lst for label,lst in zip(pos_tags, self.A.tolist())]
        print(tabulate(A, headers=pos_tags, tablefmt="grid"))

        print("\n-----Emission Probabilities-----")
        B = [[label] + lst for label,lst in zip(tokens, self.B[tokens_idx].tolist())]
        print(tabulate(B, headers=pos_tags, tablefmt="grid"))


    def ViterbiDecoding(self, idx, words):
        # TODO: create 2-D matrix for Dynamic Programming
        print("\n#####Viterbi Decoding#####")
        dp_mat = np.zeros((len(self.poss), len(idx)))
        seq_mat = []
        dp_mat[:,0] = self.pi * self.B[idx[0],:]
        if VERBOSE:
            print("For word: ", words[0])
            print(tabulate([dp_mat[:,0].tolist()], headers=self.poss, tablefmt="grid"))
        for i in range(1,len(idx)):
            lst = np.multiply(self.A, dp_mat[:,i-1].reshape((dp_mat[:,i-1].size, 1)))
            lst2 = self.B[idx[i],:] * lst
            if VERBOSE:
                A = [[label] + lst for label,lst in zip(self.poss, lst2.tolist())]
                print("For word: ", words[i])
                print(tabulate(A, headers=self.poss, tablefmt="grid"))
            dp_mat[:,i] = np.max(lst2, axis=0)
            seq_mat.append(np.argmax(lst2, axis=0))

        # TODO: Print Forward Matrix

        #print(dp_mat)
        alpha_mat = [[label] + lst for label,lst in zip(self.poss, dp_mat.tolist())]
        print("\n-----Alpha Propagation-----")
        print(tabulate(alpha_mat, headers=words, tablefmt="grid"))
        
        # TODO: Print sequence
        l = np.argmax(dp_mat[:,-1])
        sequence = []
        for seq in reversed(seq_mat):
            sequence.append(self.poss[l])
            l = seq[l]
        sequence.append(self.poss[l])
        sequence.reverse()
        print("For Words: ", words)
        print("Most Probable Sequence: ", sequence)

    def test(self, input_file_name):
        file = open(input_file_name, mode='r', encoding='utf-8')
        lines = file.readlines()
        for line in lines:
            line = line.replace('\n','')
            idx = [self.tokens.index(word) for word in line.split(' ')]
            # print(self.B[idx,:])
            print("Processing...")
            print(line)
            self.print_parameters(tokens_idx=idx)
            self.ViterbiDecoding(idx, line.split(' '))
        file.close()


if __name__ == "__main__":
    mp = ViterbiDecoding("data/train1.txt")
    mp.test("data/test1.txt")

    # mp = ViterbiDecoding("data/train3a.txt")
    # mp.test("data/test3.txt")

    # mp = ViterbiDecoding("data/train3b.txt")
    # mp.test("data/test3.txt")


    """Output
   # "C:\python 3.7\python.exe" "C:/Users/91809/Desktop/SEM 5/AI/NLP-ViterbiDecoding-master/NLP-ViterbiDecoding-master/main.py"
   //# Processing...
    time
    flies
    like
    an
    arrow
    -----Initial
    Parameters - ----
    +------+------+------+------+
    | DT | VB | IN | NN |
    += == == = += == == = += == == = += == == =+
    | 0 | 0.2 | 0 | 0.8 |
    +------+------+------+------+

    -----Transition
    Probabilities - ----
    +----+------+------+------+------+
    | | DT | VB | IN | NN |
    += == = += == == = += == == = += == == = += == == =+
    | DT | 0 | 0 | 0 | 1 |
    +----+------+------+------+------+
    | VB | 0.5 | 0 | 0.2 | 0.3 |
    +----+------+------+------+------+
    | IN | 0.25 | 0 | 0 | 0.75 |
    +----+------+------+------+------+
    | NN | 0 | 0.4 | 0.1 | 0.5 |
    +----+------+------+------+------+

    -----Emission
    Probabilities - ----
    +-------+------+------+------+------+
    | | DT | VB | IN | NN |
    += == == == += == == = += == == = += == == = += == == =+
    | time | 0 | 0.1 | 0 | 0.1 |
    +-------+------+------+------+------+
    | flies | 0 | 0.2 | 0 | 0.1 |
    +-------+------+------+------+------+
    | like | 0 | 0.2 | 0.25 | 0 |
    +-------+------+------+------+------+
    | an | 0.5 | 0 | 0 | 0 |
    +-------+------+------+------+------+
    | arrow | 0 | 0 | 0 | 0.1 |
    +-------+------+------+------+------+

    #####Viterbi Decoding#####

    -----Alpha
    Propagation - ----
    +----+--------+---------+---------+-------+---------+
    | | time | flies | like | an | arrow |
    += == = += == == == = += == == == == += == == == == += == == == += == == == == +
    | DT | 0 | 0 | 0 | 8e-05 | 0 |
    +----+--------+---------+---------+-------+---------+
    | VB | 0.02 | 0.0064 | 0.00032 | 0 | 0 |
    +----+--------+---------+---------+-------+---------+
    | IN | 0 | 0 | 0.00032 | 0 | 0 |
    +----+--------+---------+---------+-------+---------+
    | NN | 0.08 | 0.004 | 0 | 0 | 8e-06 |
    +----+--------+---------+---------+-------+---------+
    For
    Words: ['time', 'flies', 'like', 'an', 'arrow']
    Most
    Probable
    Sequence: ['NN', 'NN', 'VB', 'DT', 'NN']

    Process
    finished
    with exit code 0
"""
