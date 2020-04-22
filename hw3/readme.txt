*************************
* EECS 595 Assignment 3 *
*************************
In order to run the file, you should use the command line:

$python3 try.py node.py TestingRaw.txt grammar.pkl TextOutput.txt

in which you can change the name of the testing raw file, grammar pickle file and the text output file. Then the program will parse sentences in the TestingRaw.txt based on the grammar in grammar.pkl file and output the results in the TextOutput.txt file.

For the grammar.pkl file, I automatically extract grammar rules from the training file. Please make sure that Vocabulary.txt and grammar.pkl are under the same path with python files (all files needed should be under the same path).