import torch
import torch.nn as nn
from torch.autograd import Variable


class LocalAttention(nn.Module):
    def __init__(self, input_size, embed_size, win_size, out_channels):
        super(LocalAttention, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.win_size = win_size
        self.out_channels = out_channels

        self.attention_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(self.win_size, self.embed_size)),
            nn.Sigmoid())

        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(1, self.embed_size)),
            nn.Tanh(),
            nn.MaxPool2d((self.input_size, 1)))

    def forward(self, x):
        padding = Variable(torch.zeros(x.size(0), (self.win_size - 1) / 2, self.embed_size))
        padding = padding.cuda()
        x_pad = torch.cat((padding, x, padding), 1)

        x_pad = x_pad.unsqueeze(1)
        scores = self.attention_layer(x_pad)

        scores = scores.squeeze(1)

        out = torch.mul(x, scores)

        out = out.unsqueeze(1)
        out = self.cnn(out)

        return out


class GlobalAttention(nn.Module):
    def __init__(self, input_size, embed_size, out_channels):
        super(GlobalAttention, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.out_channels = out_channels

        self.attention_layer = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(self.input_size, self.embed_size)),
            nn.Sigmoid())

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(2, self.embed_size)),
            nn.Tanh(),
            nn.MaxPool2d((self.input_size - 2 + 1, 1)))

        self.cnn_2 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(3, self.embed_size)),
            nn.Tanh(),
            nn.MaxPool2d((self.input_size - 3 + 1, 1)))

        self.cnn_3 = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=(4, self.embed_size)),
            nn.Tanh(),
            nn.MaxPool2d((self.input_size - 4 + 1, 1)))

    def forward(self, x):
        x = x.unsqueeze(1)
        score = self.attention_layer(x)
        out = torch.mul(x, score)
        out_1 = self.cnn_1(out)
        out_2 = self.cnn_2(out)
        out_3 = self.cnn_3(out)
        return (out_1, out_2, out_3)


class CNNDLGA(nn.Module):

    def __init__(self, input_size, embed_size=100, win_size=5, channels_local=200, channels_global=100,
                 fc_input_size=500, hidden_size=500, output_size=50):
        super(CNNDLGA, self).__init__()

        self.localAttentionLayer_user = LocalAttention(input_size, embed_size, win_size, channels_local)
        self.globalAttentionLayer_user = GlobalAttention(input_size, embed_size, channels_global)
        self.localAttentionLayer_item = LocalAttention(input_size, embed_size, win_size, channels_local)
        self.globalAttentionLayer_item = GlobalAttention(input_size, embed_size, channels_global)
        self.fcLayer = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x_user, x_item):
        # user
        local_user = self.localAttentionLayer_user(x_user)
        global1_user, global2_user, global3_user = self.globalAttentionLayer_user(x_user)
        out_user = torch.cat((local_user, global1_user, global2_user, global3_user), 1)
        out_user = out_user.view(out_user.size(0), -1)
        out_user = self.fcLayer(out_user)

        # item
        local_item = self.localAttentionLayer_item(x_item)
        global1_item, global2_item, global3_item = self.globalAttentionLayer_item(x_item)
        out_item = torch.cat((local_item, global1_item, global2_item, global3_item), 1)
        out_item = out_item.view(out_item.size(0), -1)
        out_item = self.fcLayer(out_item)

        out = torch.sum(torch.mul(out_user, out_item), 1)

        return out
