import torch
import torch.nn as nn

class M1(nn.Module):
    def __init__(self, embeddingMatrix, n_lstm_layers, lstm_layer_size, dropout_propability, bidirectional, sequence_length=50):
        super(M1, self).__init__()
        number_of_tokens_in_embedding = embeddingMatrix.shape[0]
        number_of_features_per_token = embeddingMatrix.shape[1]
        # make the embedding_layer here
        self.embeding = nn.Embedding(num_embeddings=number_of_tokens_in_embedding,
                                     embedding_dim=number_of_features_per_token,
                                     _weight=embeddingMatrix)
        self.n_lstm_layer = n_lstm_layers
        self.lstm = nn.LSTM(input_size=number_of_features_per_token,
                            hidden_size=lstm_layer_size, num_layers=n_lstm_layers,
                            batch_first=True, dropout=dropout_propability,
                            bidirectional=bidirectional)# Be ware that the last layer doesn't contain dropout, don't know if batch_first should be true or false
        self.size_input_for_output = sequence_length * lstm_layer_size * (1 + int(bidirectional==True))
        self.outputLayer = nn.Linear(in_features=self.size_input_for_output, out_features=3)

    def forward(self, X):
        X = self.embeding(X)
        X, _ = self.lstm(X) # require to see dims for debugging
        #X = X.permute(1, 0, 2)
        X = X.contiguous().view(-1, self.size_input_for_output)
        X = self.outputLayer(X)
        return X

class AttentionModel(nn.Module):
    def __init__(self, features_size):
        super(AttentionModel, self).__init__()
        self.linear = nn.Linear(in_features=features_size, out_features=1)
        self.activation_function = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, X):
        X1 = self.linear(X)
        X1 = self.activation_function(X1)
        #X1 = torch.squeeze(X1, dim=-1)
        X1 = self.softmax(X1)
        weighted_features = X * X1 # check the computation and dimention
        output = torch.sum(weighted_features, dim=-2)
        return output

class M2(nn.Module):
    def __init__(self, embeddingMatrix, n_lstm_layers, lstm_layer_size, dropout_propability_embeddings, dropout_propability_lstm, bidirectional, embedding_trainable):
        super(M2, self).__init__()
        number_of_tokens_in_embedding = embeddingMatrix.shape[0]
        number_of_features_per_token = embeddingMatrix.shape[1]
        # make the embedding_layer here
        self.embedding = nn.Embedding(num_embeddings=number_of_tokens_in_embedding,
                                     embedding_dim=number_of_features_per_token,
                                     _weight=embeddingMatrix)
        self.embedding.weight.requires_grad=embedding_trainable
        self.dropoutEmbeddings = nn.Dropout(p=dropout_propability_embeddings)
        self.n_lstm_layer = n_lstm_layers
        self.lstm = nn.LSTM(input_size=number_of_features_per_token,
                            hidden_size=lstm_layer_size, num_layers=n_lstm_layers,
                            batch_first=True, dropout=dropout_propability_lstm,
                            bidirectional=bidirectional)  # Be ware that the last layer doesn't contain dropout, don't know if batch_first should be true or false
        self.dropoutLSTM = nn.Dropout(p=dropout_propability_lstm)
        words_last_feature_size = lstm_layer_size * (1 + int(bidirectional))
        self.attention = AttentionModel(words_last_feature_size)
        self.outputLayer = nn.Linear(in_features=words_last_feature_size, out_features=3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        embeddings = self.embedding(X)
        embeddings = self.dropoutEmbeddings(embeddings)
        hidden_states, _ = self.lstm(embeddings)
        hidden_states = self.dropoutLSTM(hidden_states)
        weighted_attended_word_vectors = self.attention(hidden_states)
        output_probability_unnormalized = self.outputLayer(weighted_attended_word_vectors)
        output_probabilities = self.softmax(output_probability_unnormalized)
        return  output_probabilities
