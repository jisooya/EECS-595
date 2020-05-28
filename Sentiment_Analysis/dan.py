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
ITERS = 300
K_RANGE = 20
BATCH_SIZE = 50
PADDING_SIZE = 500
DR = 0.2

DAN_FILE = "dan.torch"
GLOVE_FILE = 'glove.6B.' + str(DIM) + 'd.txt'

NEG_TRAINING_FILE = './training/neg'
POS_TRAINING_FILE = './training/pos'

NEG_VALIDATION_FILE = './validation/neg'
POS_VALIDATION_FILE = './validation/pos'

NEG_TESTING_FILE = './testing/neg'
POS_TESTING_FILE = './testing/pos'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DAN(nn.Module):
    def __init__(self):
        super(DAN, self).__init__()
        self.linear1 = nn.Linear(DIM, 600)
        self.linear2 = nn.Linear(600, 2)
        self.dropout = nn.Dropout(DR)
        self._init_weights()

    def _init_weights(self):
        for m in [self.linear1]:
            nn.init.kaiming_uniform_(m.weight.data)
        for m in [self.linear2]:
            nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, seq):
        output = F.tanh(self.linear1(seq))
        output = self.dropout(output)
        output = self.linear2(output)
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
        outname = 'dan_training.pkl'
    elif neg_filename==NEG_TESTING_FILE and pos_filename==POS_TESTING_FILE:
        outname = 'dan_testing.pkl'
    elif neg_filename==NEG_VALIDATION_FILE and pos_filename==POS_VALIDATION_FILE:
        outname = 'dan_validation.pkl'
    else:
        print("wrong file")
        return

    tags = []
    mean_embeddings = []
    glove_model = model_name

    if os.path.isfile(outname):
        data = pickle.load(open(outname,'rb'))
    else:
        for file in os.listdir(neg_filename):
            f = codecs.open(neg_filename + "/" + file, "r", encoding="ISO-8859-1")
            text = f.read()
            words_in_text = text.split()
            tags.append(0)
            sum_embedding = np.zeros(DIM)
            length = 0

            # Long file: take the first PADDING_SIZE words.
            if len(words_in_text) >= PADDING_SIZE:
                words_in_text = words_in_text[0:PADDING_SIZE]

                for word in words_in_text:
                    if word in glove_model.keys():
                        embeddings = glove_model[word]
                        sum_embedding += embeddings
                    length += 1
            # Short file: padding.
            else:
                for word in words_in_text:
                    if word in glove_model.keys():
                        embeddings = glove_model[word]
                        sum_embedding += embeddings
                    length += 1
                for i in range(len(words_in_text), PADDING_SIZE):
                    embeddings = np.ones(DIM)
                    sum_embedding += embeddings
                    length += 1

            # for word in words_in_text:
            #     if word in glove_model.keys():
            #         embeddings = glove_model[word]
            #     else:
            #         continue
            #     sum_embedding += embeddings
            #     length += 1

            mean_embeddings.append(sum_embedding/length)

        for file in os.listdir(pos_filename):
            f = codecs.open(pos_filename + "/" + file, "r", encoding="ISO-8859-1")
            text = f.read()
            words_in_text = text.split()
            tags.append(1)
            sum_embedding = np.zeros(DIM)
            length = 0

            # Long file: take the first PADDING_SIZE words.
            if len(words_in_text) >= PADDING_SIZE:
                words_in_text = words_in_text[0:PADDING_SIZE]

                for word in words_in_text:
                    if word in glove_model.keys():
                        embeddings = glove_model[word]
                        sum_embedding += embeddings
                    length += 1
            # Short file: padding.
            else:
                for word in words_in_text:
                    if word in glove_model.keys():
                        embeddings = glove_model[word]
                        sum_embedding += embeddings
                    length += 1
                for i in range(len(words_in_text), PADDING_SIZE):
                    embeddings = np.ones(DIM)
                    sum_embedding += embeddings
                    length += 1

            # for word in words_in_text:
            #     if word in glove_model.keys():
            #         embeddings = glove_model[word]
            #     else:
            #         continue
            #     sum_embedding += embeddings
            #     length += 1

            mean_embeddings.append(sum_embedding/length)

        tags = np.array(tags)
        mean_embeddings = np.array(mean_embeddings)
        data = {}
        data['tags'] = tags
        data['mean_embeddings'] = mean_embeddings
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

    dan = DAN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dan.parameters(), lr=LR)

    # For plotting.
    x_axis = []
    loss_y_axis = []
    train_y_axis = []
    validation_y_axis = []

    # Training
    optimal = 0
    for epoch in range(ITERS):
        x_axis.append(epoch+1)
        if os.path.isfile(DAN_FILE):
            dan = torch.load(DAN_FILE)

        # Shuffle embedding vectors.
        idx = np.random.permutation(range(len(train_tags)))
        train_tags = train_tags[idx]
        train_embeddings = train_embeddings[idx]

        losses = 0.0
        trained_num = 0
        train_correct = 0
        validation_correct = 0
        for k in range(K_RANGE):
            dan.zero_grad()

            # Batch input
            trained_num += BATCH_SIZE
            idxes = [idx % len(train_tags) for idx in range(k * BATCH_SIZE, (k + 1) * BATCH_SIZE)]
            X_batch = train_embeddings[idxes]
            X_batch = torch.tensor(X_batch).float()
            Y_batch = train_tags[idxes]
            Y_batch = torch.tensor(Y_batch).long()

            output = dan(X_batch)
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

        dan.eval()
        X_batch = torch.tensor(validation_embeddings).float()
        output = np.array([np.where(x == np.max(x)) for x in dan(X_batch).detach().numpy()])
        for output_y, target_y in zip(output, validation_tags):
            if output_y == target_y:
                validation_correct += 1
        validation_accuracy = 100*validation_correct/len(validation_tags)
        loss_y_axis.append(losses/K_RANGE)
        train_y_axis.append(training_accuracy)
        validation_y_axis.append(validation_accuracy)

        # Print the average loss and accuracy.
        # print("**************")
        # print("* Epoch [%d] *" % (epoch+1))
        # print("**************")
        # print("Training loss is: %.3f." % (losses/K_RANGE))
        # print('Training accuracy is: %.5f%%.' % training_accuracy)
        # print('Validation accuracy is: %.3f%%.' % validation_accuracy)

        # save the network
        optimal = max(optimal,validation_accuracy)
        if validation_accuracy == optimal:
            torch.save(dan, "FINAL_DAN.torch")

    test_dan = torch.load("FINAL_DAN.torch")
    torch.save(test_dan, DAN_FILE)
    test_dan.eval()
    X_batch = torch.tensor(test_embeddings).float()
    output = np.array([np.where(x == np.max(x)) for x in test_dan(X_batch).detach().numpy()])

    validation_correct = 0
    for output_y, target_y in zip(output, test_tags):
        if output_y == target_y:
            validation_correct += 1
    test_accuracy = 100*validation_correct/len(validation_tags)
    print("\n")
    print("**********************************")
    print('  The test accuracy is: %.3f%%.' % test_accuracy)
    print("**********************************")

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