import torch
from torch import nn
import numpy as np
from torch.nn import functional as F



class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        '''
        Trained before the model, and will be load when the model is trained.
        '''
        super(Embedding, self).__init__()
        self.embedding = nn.Linear(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)
    
class output(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(output, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        '''
        Transformer is the core of our model.
        But due to performance and memory limitation, 
        parameters can't be enormous. Thus it may need extra help.
        '''
        super(Model, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)

        self.transformer = nn.Sequential( 
            nn.TransformerDecoderLayer(embedding_dim, hidden_dim, batch_first=True),
            nn.TransformerDecoderLayer(embedding_dim, hidden_dim, batch_first=True),
            nn.TransformerDecoderLayer(embedding_dim, hidden_dim, batch_first=True)
        )
        self.output = output(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.output(x)
        return x
    
    def load_embeddings(self, embeddings):
        self.embedding.embedding.weight.data.copy_(torch.from_numpy(embeddings))

class Decider(nn.Module):
    def __init__(self, input_dim, output_dim):
        '''
        when training, there must be lots of 'natural' data, which should not 
        be used to train the model.
        Decider is used to decide whether the data is 'natural' or not.
        If the data is unuseful, just skip it and start next training.
        This is supposed to decrease much calculations, as Decider is desinged
        light weight.
        It may also be used in the game to decide whether should the model 
        reason the structure if there's no useful prompts.
        '''
        super(Decider, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
