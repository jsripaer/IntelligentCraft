import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

LEARN_RATE = 0.0005
BATCH_SIZE = 1#just for test, real batch size should be larger

class EndofBlock(Exception):
    pass

class Embedding(nn.Module):
    def __init__(self):
        '''
        Trained before the model, and will be load when the model is trained.
        basic encodding has been done in analizer.py, this embedding layer is used to
        enlarge the encoding dimension, making it sparse and having more complex linear features
        '''
        super(Embedding, self).__init__()
        self.embedding = nn.Linear(18, 128)
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=LEARN_RATE*2)

    def forward(self, x):
        '''
        return a new tensor, in which each row is the embedding of the input row
        x:(4096, 18)
        return:(4096, 128)
        '''
        return self.embedding(x)
    
class Output(nn.Module):
    def __init__(self):
        super(Output, self).__init__()
        self.linear = nn.Linear(128, 18)

    def forward(self, x):
        return self.linear(x)
    

class Decider(nn.Module):
    def __init__(self):
        #input:(4096, 18)
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
        self.squeeze = nn.Parameter(torch.randn((1,4096),dtype=torch.float32))
        
        self.linear = nn.Sequential(
            nn.Linear(18, 512),
            nn.ReLU(),#4096*512
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),#4096*128
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=LEARN_RATE/2)

    def forward(self, x):
        if torch.all(x == 0):
            return torch.tensor(0.0)#empty section
        x =  self.linear(x)#4096*1
        return torch.sigmoid(self.squeeze @ x)#1*1

class DecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self):
        super(DecoderLayer, self).__init__(d_model=128, nhead=4, dim_feedforward=512, dropout=0.05, activation='relu')
    def forward(self, x, me):
        x = super().forward(x, me, memory_is_causal=True)#use look-ahead mask by default
        return x

class Model(nn.Module):
    def __init__(self):
        #input:(4096, 18)
        '''
        Transformer is the core of our model.
        But due to performance and memory limitation, 
        parameters can't be enormous. Thus it may need extra help.
        position has been encoded in the input data, maybe we can add more positional encoding later?
        '''
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.device("cuda")
        super(Model, self).__init__()
        self.decider = Decider()
        self.embedding = Embedding()
        #HOW ON EARTH do I need a position encoding here???

        self.transformerdecodelayer = DecoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0.05, activation='relu')
        self.transformer = nn.TransformerDecoder(num_layers=8, decoder_layer=self.transformerdecodelayer, norm=nn.LayerNorm(128))

        self.output = Output()

        self.memorystack = torch.zeros((8,4096,18),dtype=torch.float32) #a simple memory stack
        self.lossfunc = nn.MSELoss()
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=LEARN_RATE)
    
    def forward(self, src):
        '''
        src are both tensor with shape (4096, 18)
        with every src, the model should predict for 4096 times
        and update its memory stack with the new prediction
        return: tensor with shape (1, 18)
        '''

        if self.decider(src).item() < 0.2:
            return None #skip the useless data
        self.memory_update(src)
        out = torch.zeros(4096,18)

        for i in range(4096): 
            src = self.cap_memory(out)#(8192,18)
            self.input = src
            src = self.embedding(src)#(8192,128)
            src = self.transformer((1,src, src))#(1,8192,128)
            src = self.output(src)#(1,18)
        return src
    
    def load_embeddings(self, embeddings):
        self.embedding.embedding.weight.data.copy_(torch.from_numpy(embeddings))

    def memory_update(self, new_memory):
        '''
        update the memory stack with new memory
        if the stack is full, pop the oldest memory
        new_memory: tensor with shape (4096,)
        '''
        self.memorystack = torch.roll(self.memorystack, shifts=-1, dims=0)
        self.memorystack[-1] = new_memory

    def cap_memory(self, x):
        '''
        cap the input with memory stack
        记忆栈会被以下方式加权平均：
        由栈内记忆数量计算一次softmax函数,以记忆的下标倒置为输入
        例如,栈内有4条记忆,则输入为[4,3,2,1],输出为softmax([4,3,2,1])
        这样可以让较新的记忆拥有更高的权重
        同时减小计算量
        //translate, not good at english sorry for that
        The memory stack will be weighted averaged as follows:
        Calculate a softmax function based on the number of memories in the stack,
        with the index of the memory reversed as input.
        For example, if there are 4 memories in the stack, the input is [4
        ,3,2,1], and the output is softmax([4,3,2,1]).
        This allows newer memories to have higher weights
        while reducing the amount of computation.
        x: tensor with shape (4096, 18)
        return: tensor with shape (8192, 18)
        '''
        #omit the empty memory
        memory_size = 0
        for i in range(self.memorystack.shape[0]):
            if torch.all(self.memorystack[i] == 0):
                continue
            memory_size += 1
        if memory_size == 0:
            return torch.cat([x,torch.zeros((4096, 18), dtype=torch.float32)],dim=0)#(8192,18)
        
        weights = F.softmax(torch.arange(memory_size,0,-1).float(), dim=0)#(1, memory_size)
        weights = weights.view(-1,1,1)
        weighted_memory = torch.sum(self.memorystack[:memory_size] * weights.unsqueeze(2), dim=0)#(4096,18)
        return torch.cat((weighted_memory, x), dim=0)#(8192,18)

