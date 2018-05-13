import codecs
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import csv
import re
import random
import matplotlib.pyplot as plt
import torch.autograd as autograd
import string
import numpy as np
from os import walk
from nltk.corpus import wordnet
def training_loop_batch(model, loss, optimizer, num_train_steps):
    step = 0
    for i in range(num_train_steps):
        model.train()
        vectors, labels = get_sorted_batch(next(training_iter), "bitcoin has crash", glove_model)
        vectors = Variable(torch.stack(vectors).squeeze().cuda())
        #labels = Variable(torch.stack(labels).squeeze())
        labels = Variable(torch.LongTensor([labels]).cuda())
        #labels = Variable(torch.LongTensor(torch.stack(labels).squeeze()[0]))

                
        model.zero_grad()
        model.hidden = model.init_hidden()
        output = model(vectors)
        
        lossy = loss(output, labels)
        lossy.backward()
        optimizer.step()
        '''
        if step % 100 == 0 :
            print( "Step %i; Loss %f; Train acc: %f; Dev acc %f" 
                %(step, lossy.data[0], evaluate(model, train_eval_iter), evaluate(model, dev_iter)))
        '''
        if step % 100 == 0 :        
            print( "Step %i; Loss %f;" 
                %(step, lossy.data[0]))

        step += 1
cnn_lstm_model = TextCNNLSTM(input_size, embedding_dim, 
                                window_size, n_filters, cnn_out_size, dropout_prob, hidden_dim, num_labels)
cnn_lstm_model.cuda()
# Loss and Optimizer
loss = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(cnn_lstm_model.parameters(), lr=learning_rate)

# Train the model
for i in range(epoch_num):
    print("Start epoch: %i" % (i + 1))
    train_set = epoch_generate(train_inc_set, train_dec_set, train_cos_set, 30)
    training_iter = data_iter(train_set, 1)
    #train_eval_iter = eval_iter(train_set[0:500], batch_size)
    # dev_iter = eval_iter(train_set[0:500], batch_size)
    num_train_steps = len(train_set)
    training_loop_batch(cnn_lstm_model, loss, optimizer, num_train_steps)
def evaluate(model, data_iter):
    model.eval()
    correct = 0
    total = 0
    for i in range(len(data_iter)):
        vectors, labels = get_sorted_batch(data_iter[i], "bitcoin has crash", glove_model)
        vectors = Variable(torch.stack(vectors).squeeze().cuda())
        labels = torch.LongTensor([labels]).cuda()
        output = model(vectors)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return correct / float(total)
# Test the model
dev_set = epoch_generate(dev_inc_set, dev_dec_set, dev_cos_set, 30)
dev_iter = eval_iter(dev_set, 1)
dev_acc = evaluate(cnn_lstm_model, dev_iter)
print('Accuracy of the CNN-LSTM on the validation data: %f' % (dev_acc))