import torch
import torch.nn as nn
import numpy as np
from main.CNN import *
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np

num_tags=2

class ET(nn.Module):
    def __init__(self, config1, config2, config3, config4, configc1, configc2):
        super(ET, self).__init__()
        self.cnn = TextCNN(config1)
        self.cnn2 = TextCNN(config2)
        self.cnn3 = TextCNN(config3)
        self.cnn4 = TextCNN(config4)
        bn_size = int(config1["feature_size"] + config1['embedding_size'] + config2['feature_size']
                      + config3["feature_size"] + config4["feature_size"]
                      )
        print(bn_size)
        self.bn = nn.BatchNorm1d(bn_size)
        self.esm = nn.Sequential(
            nn.Linear(bn_size, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 32)
        )


        self.cnn_conf = CNNBLK(configc1)

        self.cnn2_conf = CNNBLK(configc2)

        bn_size2 = int(configc2["feature_size"] + configc2['embedding_size'])
        # no residual layer
        #bn_size2 = int(config4["feature_size"])

        #self.bn2 = nn.BatchNorm1d(bn_size2)
        self.conf = nn.Sequential(
            nn.Linear(bn_size2, 64),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(64,32)
        )


        self.embedding = nn.Linear(32+32, num_tags)
        #self.embedding = nn.Linear(32, num_tags)
        self.label = nn.Linear(num_tags, num_tags)
        #self.crf = CRF(num_tags)
        self.act = nn.Softmax(dim=-1)
    def forward(self, esm,pldd):
        esm_copy = torch.clone(esm)
        esm_copy2 = self.cnn(esm)
        esm_copy3 = self.cnn3(esm)
        esm_copy4 = self.cnn4(esm)
        esm = self.cnn2(esm)
        esm = torch.cat((esm, esm_copy, esm_copy2, esm_copy3, esm_copy4), dim=-1)
        esm = esm.permute(0, 2, 1)
        esm = self.bn(esm)
        esm = esm.permute(0, 2, 1)
        esm = self.esm(esm)

        '''
        pldd_copy = torch.clone(pldd)
        pldd_copy2 = self.cnn_conf(pldd)
        pldd = self.cnn2_conf(pldd)
        pldd = torch.cat((pldd, pldd_copy, pldd_copy2), dim=-1)
        pldd = pldd.permute(0, 2, 1)
        pldd = self.bn2(pldd)
        pldd = pldd.permute(0, 2, 1)
        '''
        pldd = self.cnn_conf(pldd)
        pldd = self.cnn2_conf(pldd)
        pldd = self.conf(pldd)
        input = torch.cat((esm, pldd), dim=-1)
        #input = input + pldd
        model = self.embedding(input)
        return model, esm, pldd