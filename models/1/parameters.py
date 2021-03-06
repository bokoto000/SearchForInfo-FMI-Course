import torch

corpusFileName = 'corpusPoems'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'

device = torch.device("cuda:0")
#device = torch.device("cpu")

batchSize = 64
char_emb_size = 116

hid_size = 512
lstm_layers = 2
dropout = 0.4

epochs = 10
learning_rate = 0.001

defaultTemperature = 0.4
