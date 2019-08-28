import os
import sys

from Utils import loadData, splitData

from model import DataStoriesTaskA

from dataset import loadEmbeddings, SentimentSentencesDataset, IMDBORSSTDatasetWrapper
from preprocess import preprocess_sentence_ekphrasis

import torch
from torch import device
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


from tqdm import tqdm
import yaml
from sklearn.metrics import f1_score, accuracy_score
import argparse

from torchtext import datasets
from torchtext import data


def train_model(best_model_file_path, model, train_data_files, dev_data_files, test_data_files, device_name, batch_size, n_epochs, wordToEmbeddingIdx, max_sequence_length, learning_rate, num_workers, useExtraTrainingData):
    expDevice = device(device_name)
    train_sentences, train_labels = loadData(train_data_files)
    dev_sentences, dev_labels = loadData(dev_data_files)
    test_sentences, test_labels = loadData(test_data_files)
    #train_sentences, test_sentences, train_labels, test_labels = splitData(sentences, labels, val_size=test_split)
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
    dev_dataset = SentimentSentencesDataset(dev_sentences, dev_labels, wordToEmbeddingIdx, preprocessor_function,
                                             sequence_length=max_sequence_length)
    test_dataset = SentimentSentencesDataset(test_sentences, test_labels, wordToEmbeddingIdx, preprocessor_function,
                                             sequence_length=max_sequence_length)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=num_workers)
    model = model.to(expDevice)
    loss_function = torch.nn.CrossEntropyLoss(weight=balancing_weights)
    loss_function = loss_function.to(expDevice)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    max_f1_test_score = -1
    #best_model_file_path = os.path.join('models','{}_best.pth'.format(model_name))
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
            for dataset_name, loader in [('dev', dev_loader), ('test', test_loader)]:
                for test_x, test_y in tqdm(loader):
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
                print('{} accuracy: {}'.format(dataset_name, accuracy))
                fscore = f1_score(total_targets, total_predicted, average='macro')
                print('{} Macro f1 score: {}'.format(dataset_name, fscore))
                running_loss /= len(test_dataset)
                print('{} Avg loss: {}'.format(dataset_name, running_loss))
                if fscore > max_f1_test_score and dataset_name == 'dev':
                    max_f1_test_score = fscore
                    print('Got max f1 score of: {}'.format(fscore))
                    print('Saving model to {}'.format(best_model_file_path))
                    torch.save(model, best_model_file_path)

            #accuracy = float(correct) / total
            # compute and print f-score
            #print('Accuracy at epoch: {} is {}'.format(epoch_idx + 1, accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DataStories Training')
    parser.add_argument('--config-file', default='configs/config.yaml', help='The config file to use')
    args = parser.parse_args()
    config_file_path = args.config_file
    with open(config_file_path, mode='r') as config_file_handler:
        config = yaml.load(config_file_handler)
    model_config = config['model']
    training_config = config['training']
    wordToEmbeddingIdx, embeddingMatrix = loadEmbeddings(model_config['embedding_file_path'])
    embeddingMatrix = torch.tensor(embeddingMatrix, dtype=torch.float32)
    model_config.pop('embedding_file_path')
    model = DataStoriesTaskA(embeddingMatrix=embeddingMatrix, **model_config)
    train_model(model=model, wordToEmbeddingIdx=wordToEmbeddingIdx, **training_config)
    model_state_dictionary = model.state_dict()
    torch.save(model_state_dictionary, training_config['best_model_file_path'])
