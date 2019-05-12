import torchvision
from torch import nn
import torch


class PassModule(nn.Module):
    def __init__(self):
        super(PassModule, self).__init__()

    def forward(self, input):
        return input


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        embedding_size = 300
        self.CNN = torchvision.models.resnet18(pretrained=True)
        self.CNN.eval()
        for params in self.CNN.parameters():
            params.requires_grad = False
        self.CNN.fc = PassModule()
        self.RNN = nn.LSTM(embedding_size, 512, batch_first=True, bidirectional=True)
        self.Linear = nn.Linear(512, 1000)

    def forward(self, image, question):
        image_out = self.CNN(image)
        _, (question_out, _) = self.RNN(question)
        question_out = torch.squeeze(question_out, 0)
        #rnn_out = question_out[:, -1, :]
        print(question_out.shape)
        rnn_out = question_out[-1, :, :512] + question_out[-1, :, 512:]#question_out.view(question_out.shape[0], question_out.shape[2])
        out = image_out + rnn_out
        return self.Linear(out)

