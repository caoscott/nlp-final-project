import torchvision
from torch import nn


class PassModule(nn.Module):
    def __init__(self):
        super(PassModule, self).__init__()

    def forward(self, input):
        return input


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        embedding_size = 512
        self.CNN = torchvision.models.resnet18(pretrained=True)
        self.CNN.fc = PassModule()
        self.RNN = nn.LSTM(embedding_size, embedding_size, batch_first=True)
        self.Linear = nn.Linear(embedding_size, 1000)

    def forward(self, image, question):
        image_out = self.CNN(image)
        _, (question_out, _) = self.RNN(question)
        question_out = question_out.view(question_out.shape[0], question_out.shape[2])
        out = image_out + question_out
        return self.Linear(out)

