#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch

def generateText(model, char2id, startSentence, limit=1000, temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ
    #############################################################################
    ###  Тук следва да се имплементира генерацията на текста
    #############################################################################
    #### Начало на Вашия код.
    int2char = dict(enumerate(char2id))
    train_on_gpu = torch.cuda.is_available()
    if(train_on_gpu):
        print('Training on GPU!')
    else: 
        print('No GPU available, training on CPU; consider making n_epochs very small.')


    def init_hidden(model, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(model.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(2, batch_size, model.hidden_size).zero_().cuda(),
                  weight.new(2, batch_size, model.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(model.lstm_layers, batch_size, model.hidden_size).zero_(),
                      weight.new(model.lstm_layers, batch_size, model.hidden_size).zero_())
        #print(hidden)
        return hidden
    
    
    def one_hot_encode(arr, n_labels):
        
        # Initialize the the encoded array
        #print(arr.size)
        one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
        
        # Fill the appropriate elements with ones
        #print("one_hot", one_hot)
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
        
        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, n_labels))
        
        return one_hot


    def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = np.array([[char2id[char]]])
        #print(x)
        x = one_hot_encode(x, len(char2id))
        #print(x)
        inputs = torch.from_numpy(x)
        print("inputs",inputs)
        #print(len(inputs[0][0]))
        if(train_on_gpu):
            inputs = inputs.cuda()
        # detach hidden state from history
        #(net.embed_size,net.hidden_size, net.lstm_layers)
        h = tuple([each.data for each in h])
        #print(len(h))
        # get the output of the model
        out, h = net.lstm(inputs, h)
        # get the character probabilities
        #print(out[0][0][0].item())
        #print(len(out[0][0]))
        p = torch.nn.functional.softmax(out, dim=2).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
        print(p)
        print(p.topk(512))
        # get top characters
        pe = p.topk(512)
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
        # select the likely next character with some element of randomness
        #print(p[0][0][0].item())
        p = p.numpy().squeeze()#numpy()#.squeeze()
        #print(top_ch, p)
        char = np.random.choice(top_ch)
        # return the encoded value of the predicted char and the hidden state
        #print(char)
        return int2char[char], h

    #for x in char2id:
        #print(x)

    if(train_on_gpu):
        model.cuda()
    else:
        model.cpu()
    model.eval()

    # First off, run through the prime characters
    # First off, run through the prime characters
    chars = [ch for ch in startSentence]
    h = init_hidden(model, 1)
    for ch in startSentence:
        char, h = predict(model, ch, h, top_k=2)
    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(1000):
        char, h = predict(model, chars[-1], h,top_k=2)
        chars.append(char)
        print (char)
    return ''.join(chars)

    #### Край на Вашия код
    #############################################################################
    return chars
