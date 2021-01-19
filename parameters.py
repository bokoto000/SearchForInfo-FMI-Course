import torch

corpusFileName = 'corpusPoems'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'

device = torch.device("cuda:0")
#device = torch.device("cpu")

batchSize = 8
char_emb_size = 8

hid_size = 32
lstm_layers = 2
dropout = 0.2

epochs = 3
learning_rate = 0.005

defaultTemperature = 0.4
