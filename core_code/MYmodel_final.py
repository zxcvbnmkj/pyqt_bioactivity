import torch
from torch import nn
from torch.nn import Flatten, Linear,LSTM

# 三个维度
class theModel_final(torch.nn.Module):
    def __init__(self,dropout=0):
        super(theModel_final, self).__init__()
        self.lstm_layer=LSTM(input_size=300, hidden_size=200, batch_first=True,bidirectional=True,dropout=dropout)
        self.flatten=Flatten()
        # lstm_layer输出是维度200，有3个200，又因为是BI-LSTM,所以*2
        self.leaner=Linear(3*200*2,2)
        self.d3_leaner = Linear(32, 300)

    #graphs是（bs,32），input是(bs，2，300)。input是拼接好的一维、二维的张量。
    def forward(self, input,graphs):
        graphs=graphs.tolist()
        graphs=torch.tensor(graphs)
        #把graph变为（bs,300）
        graphs = self.d3_leaner(graphs)
        #graphs变为（bs,1,300）
        graphs = graphs.unsqueeze(1)
        input=torch.cat((graphs,input),1)
        output, (ht, ct)=self.lstm_layer(input,None)
        output=self.flatten(output)
        output = self.leaner(output)
        return output

# 2个维度
# class Model_d2d3(torch.nn.Module):
#     def __init__(self,dropout=0.1):
#         super(Model_d2d3, self).__init__()
#         self.lstm_layer=LSTM(input_size=300, hidden_size=200, batch_first=True,bidirectional=True,dropout=dropout)
#         self.flatten=Flatten()
#         # self.leaner=Linear(2*200*2,400)
#         # self.leaner2=Linear(400,100)
#         # self.leaner3 = Linear(100, 2)
#         self.leaner=Linear(2*200*2,2)
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(200, 50),  # 添加中间层
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(50, 2)
#         )
#
#
#     def forward(self, input):
#         output, (ht, ct)=self.lstm_layer(input,None)
#         output=self.flatten(output)
#         output = self.leaner(output)
#         # output = self.leaner2(output)
#         # output = self.leaner3(output)
#         return output


class Model_d2d3(torch.nn.Module):
    def __init__(self,dropout=0.15):
        super(Model_d2d3, self).__init__()
        self.lstm_layer1=LSTM(input_size=300, hidden_size=200, batch_first=True,bidirectional=True,dropout=dropout)
        self.lstm_layer2=LSTM(input_size=300, hidden_size=200, batch_first=True,bidirectional=True,dropout=dropout)
        self.flatten=Flatten()
        self.leaner1=Linear(1*200*2,50)
        self.leaner2 = Linear(1* 200 * 2, 50)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(100, 50),  # 添加中间层
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(50, 2)
        )


    def forward(self, molvec,gram):
        molvec, (ht, ct) = self.lstm_layer1(molvec, None)
        molvec = self.flatten(molvec)
        molvec = self.leaner1(molvec)

        gram, (ht, ct) = self.lstm_layer2(gram, None)
        gram = self.flatten(gram)
        gram = self.leaner2(gram)

        data = torch.cat((molvec, gram), 1)
        output=self.classifier(data)
        return output