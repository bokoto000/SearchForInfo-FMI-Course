import torch

corpusFileName = 'corpusPoems'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'

device = torch.device("cuda:0")
#device = torch.device("cpu")

batchSize = 32
char_emb_size = 116

hid_size = 1024
lstm_layers = 2
dropout = 0.8

epochs = 10
learning_rate = 0.005

defaultTemperature = 0.4
