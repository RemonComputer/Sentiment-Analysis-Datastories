from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def loadEmbeddings(embeddingsFilePath):
    # dataFrame = pd.read_csv(embeddingsFilePath, sep=' ', header=None, index_col=0)
    # words = list(dataFrame.index)
    # wordIndices = np.arange(len(words))
    # wordToEmbeddingIdx = {word: idx for idx, word in zip(wordIndices, words)}
    # embeddingMatrix = dataFrame.as_matrix()
    word_counter = 0
    wordToEmbeddingIdx = {}
    embeddingMatrix = []
    with open(embeddingsFilePath, mode='r') as f:
        #lines = f.readlines()
        for line in f:
            tokens = line.split(' ')
            word = tokens[0]
            features = [float(num) for num in tokens[1:]]
            embeddingMatrix.append(features)
            wordToEmbeddingIdx[word] = word_counter
            word_counter += 1
    embeddingMatrix = np.array(embeddingMatrix, dtype=np.float)
    return wordToEmbeddingIdx, embeddingMatrix


class SentimentSentencesDataset(Dataset):
    def __init__(self, sentences, labels_or_ids, word_to_index_dict, preprocessor_function, sequence_length=50, is_submission_data=False):
        self.labels_count = torch.zeros((3,), dtype=torch.float32)
        self.label_to_index = {
            'negative':2,
            'neutral':0,
            'positive':1
        }
        self.sequence_vs_label = []
        unk_index = word_to_index_dict['<unk>']
        pad_index = word_to_index_dict['<pad>']
        print('Making dataset...')
        for sentence, label in tqdm(zip(sentences, labels_or_ids)):
            tokens = preprocessor_function(sentence)
            indices = list(map(lambda word: word_to_index_dict.get(word, unk_index), tokens))
            indices = indices[:sequence_length]
            indices += [pad_index] * (sequence_length - len(indices))
            if not is_submission_data:
                label_idx = self.label_to_index[label]
                self.labels_count[label_idx] += 1
            else:
                label_idx = label
            self.sequence_vs_label.append((torch.tensor(indices), torch.tensor(label_idx)))

    def __len__(self):
        return len(self.sequence_vs_label)

    def __getitem__(self, idx):
        return self.sequence_vs_label[idx]

class IMDBORSSTDatasetWrapper(Dataset):
    def __init__(self, IMDBdatasets, word_to_index_dict, preprocessor_function, sequence_length=50):
        self.labels_count = torch.zeros((3, ), dtype=torch.float32)
        self.label_to_index = {
            'pos':1,
            'neg':2,
            'positive':1,
            'negative':2,
            'neutral':0
        }
        self.sequence_vs_label = []
        unk_index = word_to_index_dict['<unk>']
        pad_index = word_to_index_dict['<pad>']
        print('Making IMDB OR SST Dataset..')
        for dataset in IMDBdatasets:
            for example in tqdm(dataset):
                label = example.label
                label_idx = self.label_to_index[label]
                tokens = example.text
                sentence = ' '.join(tokens)
                tokens = preprocessor_function(sentence)
                indices = list(map(lambda word: word_to_index_dict.get(word, unk_index), tokens))
                indices = indices[:sequence_length]
                indices += [pad_index] * (sequence_length - len(indices))
                label_idx = self.label_to_index[label]
                self.labels_count[label_idx] += 1
                self.sequence_vs_label.append((torch.tensor(indices), torch.tensor(label_idx)))

    def __len__(self):
        return len(self.sequence_vs_label)

    def __getitem__(self, idx):
        return self.sequence_vs_label[idx]