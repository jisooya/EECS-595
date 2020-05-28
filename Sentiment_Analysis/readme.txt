***************************
* Readme For Assignment 2 *
***************************
File Path:
I have submitted the dan.py, dan.torch, rnn.py and rnn.torch. You have to put the dan.torch and rnn.torch file in the same directory as dan.py and rnn.py respectively in order to run them successfully.

Also, in order to load training, testing and validation files, you have to put neg and pos files in the same directory as python files because I have set the file path as:
NEG_TRAINING_FILE = './training/neg'
POS_TRAINING_FILE = './training/pos'

NEG_VALIDATION_FILE = './validation/neg'
POS_VALIDATION_FILE = './validation/pos'

NEG_TESTING_FILE = './testing/neg'
POS_TESTING_FILE = './testing/pos'

The embedding file I use is glove.6B.300d.txt for my final result, which is the pre-trained model. So you should also put glove.6B.300d.txt in the same directory as python files.

If you want to generate the plots I use in the report, you can cancel the annotations as is described in the code.

Run the code by calling respectively:
$ ​python3 dan.py
$ ​python3 rnn.py

The final test accuracy will be printed:
******************************
* The test accuracy is: xx%. *
******************************