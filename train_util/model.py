import torch
from torch.nn import Flatten, Linear,LSTM


class theModel_final(torch.nn.Module):
    def __init__(self,dropout=0.25):
        super(theModel_final, self).__init__()
        self.lstm_layer1=LSTM(input_size=32, hidden_size=200, batch_first=True,bidirectional=False,dropout=dropout)
        self.lstm_layer2=LSTM(input_size=300, hidden_size=200, batch_first=True,bidirectional=False,dropout=dropout)
        self.lstm_layer3=LSTM(input_size=300, hidden_size=200, batch_first=True,bidirectional=False,dropout=dropout)
        self.flatten=Flatten()
        self.leaner=Linear(1*200*1,2)
        self.leaner2 = Linear(1* 200 * 1, 2)
        self.leaner3 = Linear(1* 200 * 1, 2)

    def forward(self, gram,input,graphs):
        input, (ht, ct) = self.lstm_layer2(input, None)
        input = self.flatten(input)
        input = self.leaner2(input)
        gram, (ht, ct) = self.lstm_layer3(gram, None)
        gram = self.flatten(gram)
        gram = self.leaner3(gram)
        graphs = graphs.unsqueeze(1)
        graphs, (ht, ct) = self.lstm_layer1(graphs, None)
        graphs = self.flatten(graphs)
        graphs = self.leaner(graphs)
        d1_energy = torch.log(torch.sum(torch.exp(input), dim=1))
        d3_energy = torch.log(torch.sum(torch.exp(graphs), dim=1))
        gram_energy = torch.log(torch.sum(torch.exp(gram), dim=1))
        d1_conf = d1_energy / 10
        d3_conf = d3_energy / 10
        gram_conf = gram_energy / 10
        d1_conf = torch.reshape(d1_conf, (-1, 1))
        d3_conf = torch.reshape(d3_conf, (-1, 1))
        gram_conf = torch.reshape(gram_conf, (-1, 1))
        fusion_out = (input * d1_conf.detach() + graphs * d3_conf.detach()+ gram*gram_conf.detach())
        return fusion_out, input, graphs,gram, d1_conf, d3_conf,gram_conf
