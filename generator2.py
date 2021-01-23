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
    def one_hot_encode(indices, dict_size):
        ''' Define one hot encode matrix for our sequences'''
        # Creating a multi-dimensional array with the desired output shape
        # Encode every integer with its one hot representation
        features = np.eye(dict_size, dtype=np.float32)[indices.flatten()]
        
        # Finally reshape it to get back to the original array
        features = features.reshape((*indices.shape, dict_size))
                
        return features

    def predict_probs(model, hidden, character, vocab, device):
    # One-hot encoding our input to fit into the model
        character = np.array([[char2id[c] for c in char2id]])
        character = one_hot_encode(character, len(vocab))
        character = torch.from_numpy(character)
        character = character.to(device)
        
        with torch.no_grad():
            # Forward pass through the model
            out, hidden = model(character, hidden)
        # Return the logits
        prob = torch.nn.functional.softmax(out[-1], dim=0).data

        return prob, hidden
    def predict_probs(model, hidden, character, vocab, device):
        # One-hot encoding our input to fit into the model
        character = np.array([[vocab[c] for c in character]])
        character = one_hot_encode(character, len(vocab))
        character = torch.from_numpy(character)
        character = character.to(device)
        print(character)
        with torch.no_grad():
            # Forward pass through the model
            out = model(character)
        # Return the logits
        prob = nn.functional.softmax(out[-1], dim=0).data

        return prob, hidden

    def init_state(model, device, batch_size=1):
            """
            initialises rnn states.
            """
            return (torch.zeros(model.lstm_layer, batch_size, model.hidden_size).to(device),
                    torch.zeros(model.lstm_layer, batch_size, model.hidden_size).to(device))

    def predict_fn(input_data, model):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model.word2ind is None:
            raise Exception('Model has not been loaded properly, no word_dict.')
        
        # Extract the input data and the desired length
        start = input_data


        model.eval() # eval mode
        start = start
        # Clean the text as the text used in training 
        # First off, run through the starting characters
        chars = [ch for ch in start]
        size = 100
        # Init the hidden state
        state = init_state(model, device, 1)

        # Warm up the initial state, predicting on the initial string
        for ch in chars:
            #char, state = predict(model, ch, state, top_n=top_k)
            probs, state = predict_probs(model, state, ch, model.word2ind, device)
            next_index = sample_from_probs(probs, 5)

        # Include the last char predicted to the predicted output
        chars.append(model.int2char[next_index.data[0]])   
        # Now pass in the previous characters and get a new one
        for ii in range(size-1):
            #char, h = predict_char(model, chars, vocab)
            probs, state = predict_probs(model, state, chars[-1], model.word2ind, device)
            next_index = sample_from_probs(probs, 5)
            # append to sequence
            chars.append(model.int2char[next_index.data[0]])

        # Join all the chars    
        #chars = chars.decode('utf-8')
        return ''.join(chars)
    
    #### Край на Вашия код
    #############################################################################
    return predict_fn(startSentence, model)
