import os
import sys
sys.path.insert(0, '..')
from Utils import loadData, splitData

from model import M1, M2

from dataset import loadEmbeddings, SentimentSentencesDataset
from preprocess import preprocess_sentence_ekphrasis

import torch
from torch import device
from torch.utils.data import DataLoader


from tqdm import tqdm
import yaml
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd



def generateSubmissionFile(test_file_path, output_submission_file_path, model_file_path, embedding_file_path):
    test_data_frame = pd.read_csv(test_file_path, header=0, index_col=False)
    test_sentences = list(test_data_frame['tweet'])
    test_sentences_ids = list(test_data_frame['id'])
    wordToEmbeddingIdx, embeddingMatrix = loadEmbeddings(embedding_file_path)
    preprocessor_function = lambda sentence: preprocess_sentence_ekphrasis(sentence)
    test_dataset = SentimentSentencesDataset(test_sentences, test_sentences_ids, wordToEmbeddingIdx, preprocessor_function,
                                             sequence_length=50, is_submission_data=True)
    test_loader = DataLoader(test_dataset, 128, shuffle=False, num_workers=8)
    model = torch.load(model_file_path)
    ids = []
    predictions = []
    with torch.no_grad():
        model.eval()
        for test_x, test_id in tqdm(test_loader):
            test_x = test_x.to(next(model.parameters()).device)
            pred_y_batch_probabilities = model(test_x)
            _, pred_y_batch_labels = torch.max(pred_y_batch_probabilities.data, 1)
            predictions.extend(pred_y_batch_labels.cpu().numpy())
            ids.extend(test_id.cpu().numpy())
    submission_df = pd.DataFrame({'id': ids, 'label': predictions})
    submission_df.to_csv(output_submission_file_path, index=False)


if __name__ == '__main__':
    generateSubmissionFile('../../Data/test.csv', 'submission_datastores_after_droput_biased.csv', 'models/M2_best_after_dropout_accuracy_0.6578_f1_0.5966.pth', 'embeddings/datastories.twitter.50d.txt')