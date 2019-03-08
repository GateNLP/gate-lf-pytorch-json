from gatelfpytorchjson import CustomModule
from gatelfpytorchjson import EmbeddingsModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)



class TextClassBiLstmCnnSingle(CustomModule):
    def __init__(self, dataset, config={}, maxSentLen=100, kernel_dim=64, lstm_dim=64, dropout=0.2, bn_momentum=0.2):
        super().__init__(config=config)
        #super(TextClassBiLstmCnnSingle, self).__init__()
        self.maxSentLen=maxSentLen
        self.n_classes = dataset.get_info()["nClasses"]

        feature = dataset.get_indexlist_features()[0]
        vocab = feature.vocab
        vocab_size = vocab.n
        logger.debug("Initializing module TextClassCNNsingle for classes: %s and vocab %s" %
                     (self.n_classes, vocab_size, ))   

        self.embedding = EmbeddingsModule(vocab)
        embedding_dim = self.embedding.emb_dims
        self.lstm1 = nn.LSTM(embedding_dim, lstm_dim, batch_first=True, bidirectional=True)

        self.conv2 = nn.Conv2d(1, kernel_dim, (2,lstm_dim*2))
        self.conv2_bn = nn.BatchNorm1d(kernel_dim, momentum=bn_momentum)

        self.conv3 = nn.Conv2d(1, kernel_dim, (3,lstm_dim*2))
        self.conv3_bn = nn.BatchNorm1d(kernel_dim, momentum=bn_momentum)

        self.conv4 = nn.Conv2d(1, kernel_dim, (4,lstm_dim*2))
        self.conv4_bn = nn.BatchNorm1d(kernel_dim, momentum=bn_momentum)

        self.conv5 = nn.Conv2d(1, kernel_dim, (5,lstm_dim*2))
        self.conv5_bn = nn.BatchNorm1d(kernel_dim, momentum=bn_momentum)

        self.conv6 = nn.Conv2d(1, kernel_dim, (6,lstm_dim*2))
        self.conv6_bn = nn.BatchNorm1d(kernel_dim, momentum=bn_momentum)

        self.fc_hidden1 = nn.Linear(kernel_dim*5, 16)

        #self.fc = nn.Linear(kernel_dim*3,output_size)
        self.fc = nn.Linear(16,self.n_classes)
        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, batch):
        #print("checking shapes")
        #print(x.shape) # [batch size, sentence length]
        batch = torch.LongTensor(batch[0])
        sent_len = batch.shape[1]
        batchsize = batch.shape[0] 

        if sent_len > self.maxSentLen:
            batch = batch[:,:self.maxSentLen]
        elif sent_len < self.maxSentLen:
            zero_pad = torch.zeros(batchsize, self.maxSentLen-sent_len, dtype=torch.long)
            batch = torch.cat((batch, zero_pad),dim=1)


        if self.on_cuda():
            batch = batch.type(torch.cuda.LongTensor)
            batch.cuda()
        #batchsize = batch.size()[0]    

        #logger.info(batch)
        embedded = self.embedding(batch)
        #print(embedded.shape)
        #print(x.shape) # [batch size, sentence length, embedding dim]
        lstmed, hidden = self.lstm1(embedded)
        #print(lstmed.shape)
        lstmed = lstmed.unsqueeze(1) # add 1 channel to cnn
        #print(lstmed.shape)
        #print(x.shape) # [batch size, channle, sentence length, embedding dim]
        conved_2 = F.relu(self.conv2(lstmed)).squeeze(3) # conv over embedding size, left 1 in last
        conved_2 = self.conv2_bn(conved_2)
        conved_3 = F.relu(self.conv3(lstmed)).squeeze(3)
        conved_3 = self.conv3_bn(conved_3)
        conved_4 = F.relu(self.conv4(lstmed)).squeeze(3) 
        conved_4 = self.conv4_bn(conved_4)
        conved_5 = F.relu(self.conv5(lstmed)).squeeze(3)
        conved_5 = self.conv5_bn(conved_5)
        conved_6 = F.relu(self.conv6(lstmed)).squeeze(3)
        conved_6 = self.conv6_bn(conved_6)

        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        pooled_4 = F.max_pool1d(conved_4, conved_4.shape[2]).squeeze(2)
        pooled_5 = F.max_pool1d(conved_5, conved_5.shape[2]).squeeze(2)
        pooled_6 = F.max_pool1d(conved_6, conved_6.shape[2]).squeeze(2)

        concat = torch.cat((pooled_2, pooled_3, pooled_4, pooled_5, pooled_6), dim=1)
        concat = self.dropout(concat)
        hidden1 = F.leaky_relu(self.fc_hidden1(concat) )
        #hidden1 = torch.sigmoid(self.fc_hidden1(concat))
        out = self.fc(hidden1)
        out = self.logsoftmax(out)
        return out


    def get_lossfunction(self, config={}):
        # IMPORTANT: for the target indices, we use -1 for padding by default!
        return torch.nn.NLLLoss(ignore_index=-1)

    def get_optimizer(self, config={}):
        parms = filter(lambda p: p.requires_grad, self.parameters())
        # optimizer = torch.optim.SGD(parms, lr=0.01, momentum=0.9)
        # optimizer = torch.optim.SGD(parms, lr=0.01, momentum=0.9, weight_decay=0.05)
        optimizer = torch.optim.Adam(parms, lr=0.015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        return optimizer

