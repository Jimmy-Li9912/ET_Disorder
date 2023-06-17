import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import Parameter, init

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.is_training = True
        self.dropout_rate = config['dropout_rate']
        self.config = config
        if (config["activation_function_type"]== "ReLU"):
            p = nn.ReLU()
        elif (config["activation_function_type"]== "Sigmoid"):
            p=nn.Sigmoid()
        else:
            p=nn.Tanh()
        self.conv = nn.Sequential(nn.Conv1d(in_channels=config['embedding_size'],
                                    out_channels=config['feature_size'],
                                    kernel_size=config["kernel_size"], padding=math.floor(config["kernel_size"]/2)),

            #NSP no batchnorm
            #nn.BatchNorm1d(num_features=config['feature_size']),
            # 原文声明activation统一使用ReLU
            p)


    def forward(self, x):
        # x shape=(batch_size, max_text_len, 148)

        embed_x = x
        # embed_x shape=(batch_size, max_text_len,  embedding_dim=148)
        embed_x = embed_x.permute(0, 2, 1)

        # embed_x shape=(batch_size, embedding_dim=148, max_text_len)
        # conv in_channels=148, out_channel=40/20

        # out shape=(batch_size, out_channel=40/20, max_text_len)
        out = self.conv(embed_x.float())

        # out shape=(batch_size, max_text_len, out_channel=40/20)
        out=out.permute(0 ,2, 1)
        return out
class CNNBLK(nn.Module):
    def __init__(self, config):
        super(CNNBLK, self).__init__()
        c1 = config.copy()
        self.cnn1=TextCNN(c1)

        c2 = config.copy()
        c2['embedding_size'] = c1['feature_size']
        self.cnn2=TextCNN(c2)

        self.bn=nn.BatchNorm1d(int(c2['feature_size']+c1['embedding_size']))

    def forward(self, x):
        x_cp = torch.clone(x)
        x_cp=self.cnn1(x_cp)
        x_cp=self.cnn2(x_cp)
        x=torch.cat((x,x_cp),dim=-1)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        return x

class CNN2D(nn.Module):
    def __init__(self, config):
        super(CNN2D, self).__init__()
        self.is_training = True
        self.dropout_rate = config['dropout_rate']
        self.config = config
        if (config["activation_function_type"]== "ReLU"):
            p = nn.ReLU()
        elif (config["activation_function_type"]== "Sigmoid"):
            p=nn.Sigmoid()
        else:
            p=nn.Tanh()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=config['embedding_size'],
                                    out_channels=config['feature_size'],
                                    kernel_size=config["kernel_size"], padding=tuple(map(self.devide_by_2,config["kernel_size"]))),


            nn.BatchNorm2d(num_features=config['feature_size']),
            # 原文声明activation统一使用ReLU
            p)

    def devide_by_2(self,x):
        return math.floor(x/2)
    def forward(self, x):
        # x shape=(batch_size, max_text_len, 148)

        embed_x = x
        # embed_x shape=(batch_size, max_text_len,  embedding_dim=148)
        embed_x = embed_x.unsqueeze(1)

        # embed_x shape=(batch_size, embedding_dim=148, max_text_len)
        # conv in_channels=148, out_channel=40/20

        # out shape=(batch_size, out_channel=40/20, max_text_len)
        out = self.conv(embed_x.float())


        return out

if __name__ == '__main__':
    # The code below is used for test!


    # Convolution
    # 第一个CNN Module：kernel_size = 5
    kernel_size=5
    feature_size=40


    x=np.random.random((128, 3000, 148))
    x=torch.tensor(x)
    config={'dropout_rate':0.5, 'kernel_size':kernel_size, 'embedding_size':148,
            'feature_size':feature_size, 'activation_function_type':"Sigmoid"}

    model=TextCNN(config)
    y=model(x)

