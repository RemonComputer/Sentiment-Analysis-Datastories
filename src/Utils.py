import pandas as pd
from sklearn.model_selection import train_test_split

def loadData(dataFilesPaths=['../Data/train-dev-test/twitter-2013train-A.txt',
                             '../Data/train-dev-test/twitter-2015train-A.txt',
                             '../Data/train-dev-test/twitter-2016train-A.txt']):
    sentences = []
    labels = []
    for dataFilePath in dataFilesPaths:
        dataframe = pd.read_csv(dataFilePath, sep='\t',
                                header=None,
                                names=['ID', 'Label', 'Sentence'],
                                lineterminator='\n',
                                escapechar='\\',
                                error_bad_lines=True)
        is_sentence_good = ~dataframe['Sentence'].str.contains('not available', case=False)
        sentences.extend(list(dataframe['Sentence'][is_sentence_good]))
        labels.extend(list(dataframe['Label'][is_sentence_good]))
    return sentences, labels

def loadStopWords(filePath='../Resources/stopwords.txt'):
    dataFrame = pd.read_csv(filePath,
                            names=['word'],
                            comment='#')
    return list(dataFrame['word'])

def loadLexicons(positiveLexiconFilePath='../Resources/opinion-lexicon-English-Bing_liu/positive-words.txt',
                 negativeLexiconFilePath='../Resources/opinion-lexicon-English-Bing_liu/negative-words.txt'):
    wordsList = []
    for filePath in [positiveLexiconFilePath, negativeLexiconFilePath]:
        dataframe = pd.read_csv(filePath,
                                comment=';',
                                names=['word'],
                                encoding = 'latin-1')
        words = list(dataframe['word'])
        wordsList.append(words)
    return wordsList


def unitTestLoadData():
    sentences, labels = loadData()
    assert len(sentences) == len(labels), 'lengths of the sentences and labels aren\'t equal'
    for idx, label in enumerate(labels):
        assert label in ['positive', 'negative', 'neutral'], \
            'label {} is not a good label it is {}'.format(idx, label)

def unitTestLoadStopWords():
    words = loadStopWords()
    for idx, word in enumerate(words):
        assert ' ' not in word, 'word {}: -{}- has space'.format(idx, word)

def unitTestLoadLexicons():
    wordLists = loadLexicons()
    for wordlist in wordLists:
        for idx, word in enumerate(wordlist):
            assert ' ' not in word, 'word {}: -{}- has space'.format(idx, word)

def splitData(sentences, labels, val_size=0.2, random_seed=7):
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=val_size, random_state=random_seed)
    return train_sentences, test_sentences, train_labels, test_labels

def generateSubmissionFile(test_file_path, output_submission_file_path, sklearnEstimator, classifer_output_dict={'neutral':'NEUTRAL', 'positive':'POSITIVE', 'negative':'NEGATIVE'}):
    test_data_frame = pd.read_csv(test_file_path, header=0, index_col=False)
    sentences = list(test_data_frame['tweet'])
    predicted_labels_by_estimator = sklearnEstimator.predict(sentences)
    competition_submission_dict = {
        'NEUTRAL': 0,
        'POSITIVE': 1,
        'NEGATIVE': 2
    }
    predicted_labels_for_submission = list(map(lambda estimator_label: competition_submission_dict[classifer_output_dict[estimator_label]], predicted_labels_by_estimator))
    test_sentences_ids = list(test_data_frame['id'])
    submission_df = pd.DataFrame({'id':test_sentences_ids, 'label':predicted_labels_for_submission})
    submission_df.to_csv(output_submission_file_path, index=False)

# def generateSubmissionFileFromPytorchModel(test_file_path, output_submission_file_path, pytorchModel, classifer_output_dict={'neutral':'NEUTRAL', 'positive':'POSITIVE', 'negative':'NEGATIVE'}):
#     test_data_frame = pd.read_csv(test_file_path, header=0, index_col=False)
#     sentences = list(test_data_frame['tweet'])
#     predicted_labels_by_estimator = sklearnEstimator.predict(sentences)
#     competition_submission_dict = {
#         'NEUTRAL': 0,
#         'POSITIVE': 1,
#         'NEGATIVE': 2
#     }
#     predicted_labels_for_submission = list(map(lambda estimator_label: competition_submission_dict[classifer_output_dict[estimator_label]], predicted_labels_by_estimator))
#     test_sentences_ids = list(test_data_frame['id'])
#     submission_df = pd.DataFrame({'id':test_sentences_ids, 'label':predicted_labels_for_submission})
#     submission_df.to_csv(output_submission_file_path, index=False)

if __name__ == '__main__':
    unitTestLoadData()
    unitTestLoadStopWords()
    unitTestLoadLexicons()