import numpy as np
import pickle, os, sys
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import codecs
import matplotlib.pyplot as plt

DIM = 300
LR = 0.0001
ITERS = 50
K_RANGE = 20
BATCH_SIZE = 50
PADDING_SIZE = 500
DR = 0.2

RNN_FILE = "rnn.torch"
GLOVE_FILE = 'glove.6B.' + str(DIM) + 'd.txt'

NEG_TRAINING_FILE = './training/neg'
POS_TRAINING_FILE = './training/pos'

NEG_VALIDATION_FILE = './validation/neg'
POS_VALIDATION_FILE = './validation/pos'

NEG_TESTING_FILE = './testing/neg'
POS_TESTING_FILE = './testing/pos'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=DIM, hidden_size=45, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(90, 180)
        self.linear2 = nn.Linear(180, 180)
        self.linear3 = nn.Linear(180, 2)
        self.dropout = nn.Dropout(DR)
        self.init_weights()

    def init_weights(self):
        for m in [self.linear1]:
            nn.init.kaiming_uniform_(m.weight.data)
        for m in [self.linear2]:
            nn.init.kaiming_uniform_(m.weight.data)
        for m in [self.linear2]:
            nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, seq, hidden=None):
        _, output = self.rnn(seq)
        output = torch.cat((output[0][0], output[1][1]), 1)
        output = F.relu(self.linear1(output))
        output = self.dropout(output)
        output = F.relu(self.linear2(output))
        output = self.dropout(output)
        output = self.linear3(output)
        return output

#########################
# Load the glove model. #
#########################
def loadGloveModel(gloveFile):
    outname = 'glove.pkl'
    if os.path.isfile(outname):
        glove_model = pickle.load(open(outname,'rb'))
    else:
        f = open(gloveFile,'r')
        glove_model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            glove_model[word] = embedding
        pickle.dump(glove_model, open(outname,'wb'))
        print("All glove words and vectors are dumped to the file: "+outname)
    return glove_model


###############################
# Turn the files into matrix. #
###############################
def file_embedding(model_name, neg_filename, pos_filename):
    if neg_filename==NEG_TRAINING_FILE and pos_filename==POS_TRAINING_FILE:
        outname = 'rnn_training.pkl'
    elif neg_filename==NEG_TESTING_FILE and pos_filename==POS_TESTING_FILE:
        outname = 'rnn_testing.pkl'
    elif neg_filename==NEG_VALIDATION_FILE and pos_filename==POS_VALIDATION_FILE:
        outname = 'rnn_validation.pkl'
    else:
        print("wrong file")
        return

    tags = []
    txt_embeddings = []
    glove_model = model_name

    if os.path.isfile(outname):
        data = pickle.load(open(outname,'rb'))
    else:
        for file in os.listdir(neg_filename):
            f = codecs.open(neg_filename + "/" + file, "r", encoding="ISO-8859-1")
            text = f.read()
            words_in_text = text.split()
            tags.append(0)
            embeddings = np.zeros((PADDING_SIZE, DIM))

            # Long file: take the first PADDING_SIZE words.
            if len(words_in_text) >= PADDING_SIZE:
                words_in_text = words_in_text[0:PADDING_SIZE]

                for i, word in enumerate(words_in_text):
                    if word in glove_model.keys():
                        one_embedding = glove_model[word]
                    else:
                        one_embedding = np.zeros(DIM)
                    embeddings[i] = one_embedding

            # Short file: padding.
            else:
                for i, word in enumerate(words_in_text):
                    if word in glove_model.keys():
                        one_embedding = glove_model[word]
                    else:
                        one_embedding = np.zeros(DIM)
                    embeddings[i] = one_embedding
                # Padding: Add 1s.
                for i in range(len(words_in_text), PADDING_SIZE):
                    one_embedding = np.ones(DIM)
                    embeddings[i] = one_embedding

            txt_embeddings.append(embeddings)
            # for word in words_in_text:
            #     if word in glove_model.keys():
            #         embeddings = glove_model[word]
            #     else:
            #         continue
            #     sum_embedding += embeddings
            #     length += 1

        for file in os.listdir(pos_filename):
            f = codecs.open(pos_filename + "/" + file, "r", encoding="ISO-8859-1")
            text = f.read()
            words_in_text = text.split()
            tags.append(1)
            embeddings = np.zeros((PADDING_SIZE, DIM))

            # Long file: take the first PADDING_SIZE words.
            if len(words_in_text) >= PADDING_SIZE:
                words_in_text = words_in_text[0:PADDING_SIZE]

                for i, word in enumerate(words_in_text):
                    if word in glove_model.keys():
                        one_embedding = glove_model[word]
                    else:
                        one_embedding = np.zeros(DIM)
                    embeddings[i] = one_embedding

            # Short file: padding.
            else:
                for i, word in enumerate(words_in_text):
                    if word in glove_model.keys():
                        one_embedding = glove_model[word]
                    else:
                        one_embedding = np.zeros(DIM)
                    embeddings[i] = one_embedding
                for i in range(len(words_in_text), PADDING_SIZE):
                    one_embedding = np.ones(DIM)
                    embeddings[i] = one_embedding

            txt_embeddings.append(embeddings)
            # for word in words_in_text:
            #     if word in glove_model.keys():
            #         embeddings = glove_model[word]
            #     else:
            #         continue
            #     sum_embedding += embeddings
            #     length += 1

        tags = np.array(tags)
        txt_embeddings = np.array(txt_embeddings)
        data = {}
        data['tags'] = tags
        data['mean_embeddings'] = txt_embeddings
        pickle.dump(data, open(outname,'wb'))
    return data


def main(argv):
    # Load glove model: glove_model[word] = embedding vector
    glove_model = loadGloveModel(GLOVE_FILE)

    # Load 6 files.
    train_data = file_embedding(glove_model, NEG_TRAINING_FILE, POS_TRAINING_FILE)
    train_tags = train_data['tags']
    train_embeddings = train_data['mean_embeddings']

    test_data = file_embedding(glove_model, NEG_TESTING_FILE, POS_TESTING_FILE)
    test_tags = test_data['tags']
    test_embeddings = test_data['mean_embeddings']

    validation_data = file_embedding(glove_model, NEG_VALIDATION_FILE, POS_VALIDATION_FILE)
    validation_tags = validation_data['tags']
    validation_embeddings = validation_data['mean_embeddings']

    rnn = RNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=LR)

    # For plotting.
    x_axis = []
    loss_y_axis = []
    train_y_axis = []
    validation_y_axis = []

    # Training
    optimal = 0
    for epoch in range(ITERS):
        x_axis.append(epoch+1)
        if os.path.isfile(RNN_FILE):
            rnn = torch.load(RNN_FILE)

        # Shuffle embedding vectors.
        idx = np.random.permutation(range(len(train_tags)))
        train_tags = train_tags[idx]
        train_embeddings = train_embeddings[idx]

        losses = 0.0
        trained_num = 0
        train_correct = 0
        validation_correct = 0
        for k in range(K_RANGE):
            rnn.zero_grad()

            # Batch input
            trained_num += BATCH_SIZE
            idxes = [idx % len(train_tags) for idx in range(k * BATCH_SIZE, (k + 1) * BATCH_SIZE)]
            # Turn into tensor.
            X_batch = train_embeddings[idxes]
            X_batch = torch.tensor(X_batch).float()
            Y_batch = train_tags[idxes]
            Y_batch = torch.tensor(Y_batch).long()
            # Put into rnn.
            output = rnn(X_batch)
            loss = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()

            output = np.array([np.where(x == np.max(x)) for x in output.detach().numpy()])
            Y_batch = np.squeeze(Y_batch.detach().numpy())
            for output_y, target_y in zip(output, Y_batch):
                if output_y == target_y:
                    train_correct += 1
            losses += loss.item()

        training_accuracy = 100*train_correct/trained_num

        rnn.eval()
        X_batch = torch.tensor(validation_embeddings).float()
        output = np.array([np.where(x == np.max(x)) for x in rnn(X_batch).detach().numpy()])
        for output_y, target_y in zip(output, validation_tags):
            if output_y == target_y:
                validation_correct += 1
        validation_accuracy = 100*validation_correct/len(validation_tags)
        loss_y_axis.append(losses/K_RANGE)
        train_y_axis.append(training_accuracy)
        validation_y_axis.append(validation_accuracy)

        # Print the average loss and accuracy.
        print("**************")
        print("* Epoch [%d] *" % (epoch+1))
        print("**************")
        print("Training loss is: %.3f." % (losses/K_RANGE))
        print('Training accuracy is: %.5f%%.' % training_accuracy)
        print('Validation accuracy is: %.3f%%.' % validation_accuracy)

        # save the network
        optimal = max(optimal,validation_accuracy)
        if validation_accuracy == optimal:
            torch.save(rnn, "FINAL_RNN.torch")

    test_rnn = torch.load("FINAL_RNN.torch")
    torch.save(test_rnn, RNN_FILE)
    test_rnn.eval()
    X_batch = torch.tensor(test_embeddings).float()
    output = np.array([np.where(x == np.max(x)) for x in test_rnn(X_batch).detach().numpy()])

    validation_correct = 0
    for output_y, target_y in zip(output, test_tags):
        if output_y == target_y:
            validation_correct += 1

    test_accuracy =  100*validation_correct/len(validation_tags)
    print("\n")
    print("**********************************")
    print('  The test accuracy is: %.3f%%.' % test_accuracy)
    print("**********************************")

    # fp = open("relu_loss_plot_file", 'w+')
    # for element in loss_y_axis:
    #     fp.write(str(element) + '\n')
    #
    # fp = open("relu_train_plot_file", 'w+')
    # for element in train_y_axis:
    #     fp.write(str(element) + '\n')
    #
    # fp = open("relu_val_plot_file", 'w+')
    # for element in validation_y_axis:
    #     fp.write(str(element) + '\n')


# x_axis = []
# loss_y_tanh = []
# loss_y_relu = []
# for i in range(0, 300):
#     x_axis.append(i+1)
# with open("tanh_loss_plot_file",'r+') as f:
#     for line in f.readlines():
#         loss_y_tanh.append(line)
#
# with open("relu_loss_plot_file",'r+') as f:
#     for line in f.readlines():
#         loss_y_relu.append(line)

# plt.plot(x_axis, loss_y_tanh, color='green', label='loss (tanh)')
# plt.plot(x_axis, loss_y_relu, color='blue', label='loss (relu)')
# plt.xlabel('epoch')
# plt.ylabel('training loss (%)')
# plt.legend()
# plt.show()

    # Use this part to generate plots.
    # plt.plot(x_axis, loss_y_axis)
    # plt.xlabel('epoch')
    # plt.ylabel('training loss (%)')
    # plt.show()
    #
    # plt.plot(x_axis, train_y_axis)
    # plt.xlabel('epoch')
    # plt.ylabel('training accuracy (%)')
    # plt.show()
    #
    # plt.plot(x_axis, validation_y_axis)
    # plt.xlabel('epoch')
    # plt.ylabel('validation accuracy (%)')
    # plt.show()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main(sys.argv)