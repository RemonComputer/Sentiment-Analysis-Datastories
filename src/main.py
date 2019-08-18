import os
import sys
sys.path.insert(0, '..')
from Utils import loadData, splitData

from model import M1, M2

from dataset import loadEmbeddings, SentimentSentencesDataset, IMDBORSSTDatasetWrapper
from preprocess import preprocess_sentence_ekphrasis

import torch
from torch import device
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


from tqdm import tqdm
import yaml
from sklearn.metrics import f1_score, accuracy_score

from torchtext import datasets
from torchtext import data




def runExp1():
    expDevice = device('cuda:0')
    data_files = ['../../Data/twitter-2013train.txt',
                        '../../Data/twitter-2015train.txt',
                        '../../Data/twitter-2016train.txt']
    sentences, labels = loadData(data_files)
    train_sentences, test_sentences, train_labels, test_labels = splitData(sentences, labels, val_size=0.2)
    wordToEmbeddingIdx, embeddingMatrix = loadEmbeddings('embeddings/datastories.twitter.50d.txt')
    preprocessor_function = lambda sentence: preprocess_sentence_ekphrasis(sentence)
    sequence_length = 50
    train_dataset = SentimentSentencesDataset(train_sentences, train_labels, wordToEmbeddingIdx, preprocessor_function, sequence_length = sequence_length)
    test_dataset = SentimentSentencesDataset(test_sentences, test_labels, wordToEmbeddingIdx, preprocessor_function,
                                             sequence_length=sequence_length)
    batch_size = 64
    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=num_workers)
    n_epochs = 50
    embeddingMatrix = torch.tensor(embeddingMatrix, dtype=torch.float32)
    embeddingMatrix = embeddingMatrix.to(expDevice)
    model = M1(embeddingMatrix, n_lstm_layers=2, lstm_layer_size=150, dropout_propability=0.01, bidirectional=True, sequence_length=sequence_length)
    model = model.to(expDevice)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch_idx in range(n_epochs):
        running_loss = 0
        print('Training Epoch {}...'.format(epoch_idx + 1))
        model.train()
        for train_x, train_y in tqdm(train_loader):
            train_x = train_x.to(expDevice)
            train_y = train_y.to(expDevice)
            optimizer.zero_grad()
            pred_y = model(train_x)
            loss_val = loss_function(pred_y, train_y)
            running_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()
        running_loss /= len(train_dataset)
        print('Avg Training loss: {}'.format(running_loss))
        with torch.no_grad():
            running_loss = 0
            print('Testing ...')
            model.eval()
            total = 0
            correct = 0
            for test_x, test_y in tqdm(test_loader):
                test_x = test_x.to(expDevice)
                test_y = test_y.to(expDevice)
                pred_y = model(test_x)
                loss_val = loss_function(pred_y, test_y)
                running_loss += loss_val.item()
                _, predicted_labels = torch.max(pred_y.data, 1)
                total += pred_y.size(0)
                correct += (predicted_labels == test_y).sum().item()
            accuracy = float(correct) / total
            print('Accuracy at epoch: {} is {}'.format(epoch_idx + 1, accuracy))

#def run_on_dataset_and_compute_metrics(model, dataset, device, isTraining=False)

def train_model(model_name, model, data_files, device_name, batch_size, n_epochs, wordToEmbeddingIdx, max_sequence_length, learning_rate, num_workers, test_split=0.2, useExtraTrainingData=True):
    expDevice = device(device_name)
    sentences, labels = loadData(data_files)
    train_sentences, test_sentences, train_labels, test_labels = splitData(sentences, labels, val_size=test_split)
    preprocessor_function = lambda sentence: preprocess_sentence_ekphrasis(sentence)
    train_dataset = SentimentSentencesDataset(train_sentences, train_labels, wordToEmbeddingIdx, preprocessor_function,
                                              sequence_length=max_sequence_length)
    labels_count = train_dataset.labels_count
    if useExtraTrainingData:
        TEXT = data.Field(lower=False, include_lengths=False, batch_first=True)
        LABEL = data.Field(sequential=False)
        print('Loading IMDB From Disk...')
        train_IMDB, test_IMDB = datasets.IMDB.splits(TEXT, LABEL)
        imdb_dataset=IMDBORSSTDatasetWrapper((train_IMDB, test_IMDB) ,wordToEmbeddingIdx, preprocessor_function, sequence_length=max_sequence_length)
        labels_count += imdb_dataset.labels_count
        print('Loading SST From Disk...')
        train_SST, val_SST, test_SST = datasets.SST.splits(TEXT, LABEL)
        sst_dataset = IMDBORSSTDatasetWrapper((train_SST, val_SST, test_SST) ,wordToEmbeddingIdx, preprocessor_function, sequence_length=max_sequence_length)
        labels_count += sst_dataset.labels_count
        train_dataset = ConcatDataset([train_dataset, imdb_dataset, sst_dataset])
    balancing_weights = torch.max(labels_count) / labels_count
    test_dataset = SentimentSentencesDataset(test_sentences, test_labels, wordToEmbeddingIdx, preprocessor_function,
                                             sequence_length=max_sequence_length)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=num_workers)
    model = model.to(expDevice)
    loss_function = torch.nn.CrossEntropyLoss() #weight=balancing_weights.to(expDevice)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    max_f1_test_score = -1
    best_model_file_path = os.path.join('models','{}_best.pth'.format(model_name))
    for epoch_idx in range(n_epochs):
        running_loss = 0
        print('Starting Epoch: {}'.format(epoch_idx + 1))
        print('Training...')
        model.train()
        total_targets = []
        total_predicted = []
        for train_x, train_y in tqdm(train_loader):
            train_x = train_x.to(expDevice)
            train_y = train_y.to(expDevice)
            optimizer.zero_grad()
            pred_y = model(train_x)
            loss_val = loss_function(pred_y, train_y)
            _, predicted_labels = torch.max(pred_y.data, 1)
            running_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()
            total_targets.extend(train_y.cpu().numpy())
            total_predicted.extend(predicted_labels.cpu().numpy())
        accuracy = accuracy_score(total_targets, total_predicted)
        print('accuracy: {}'.format(accuracy))
        fscore = f1_score(total_targets, total_predicted, average='macro')
        print('Macro f1 score: {}'.format(fscore))
        running_loss /= len(train_dataset)
        print('Avg loss: {}'.format(running_loss))
        with torch.no_grad():
            running_loss = 0
            print('Testing ...')
            model.eval()
            #total = 0
            #correct = 0
            total_targets = []
            total_predicted = []
            for test_x, test_y in tqdm(test_loader):
                test_x = test_x.to(expDevice)
                test_y = test_y.to(expDevice)
                pred_y = model(test_x)
                loss_val = loss_function(pred_y, test_y)
                running_loss += loss_val.item()
                _, predicted_labels = torch.max(pred_y.data, 1)
                #total += pred_y.size(0)
                #correct += (predicted_labels == test_y).sum().item()
                total_targets.extend(test_y.cpu().numpy())
                total_predicted.extend(predicted_labels.cpu().numpy())
            accuracy = accuracy_score(total_targets, total_predicted)
            print('accuracy: {}'.format(accuracy))
            fscore = f1_score(total_targets, total_predicted, average='macro')
            print('Macro f1 score: {}'.format(fscore))
            running_loss /= len(test_dataset)
            print('Avg loss: {}'.format(running_loss))
            if fscore > max_f1_test_score:
                max_f1_test_score = fscore
                print('Got max f1 score of: {}'.format(fscore))
                print('Saving model to {}'.format(best_model_file_path))
                torch.save(model, best_model_file_path)

            #accuracy = float(correct) / total
            # compute and print f-score
            #print('Accuracy at epoch: {} is {}'.format(epoch_idx + 1, accuracy))

def runExp2(config_file_path):
    with open(config_file_path, mode='r') as config_file_handler:
        config = yaml.load(config_file_handler)
    model_config = config['model']
    training_config = config['training']
    wordToEmbeddingIdx, embeddingMatrix = loadEmbeddings(model_config['embedding_file_path'])
    embeddingMatrix = torch.tensor(embeddingMatrix, dtype=torch.float32)
    #device = torch.device(training_config['device_name'])
    model_config.pop('embedding_file_path')
    model = M2(embeddingMatrix=embeddingMatrix, **model_config)
    #model = model.to(device)
    train_model(model_name='M2',model=model, wordToEmbeddingIdx=wordToEmbeddingIdx, **training_config)
    model_state_dictionary = model.state_dict()
    torch.save(model_state_dictionary, 'models/exp2.pth')

if __name__ == '__main__':
    # runExp1()
    runExp2('configs/config1.yaml')