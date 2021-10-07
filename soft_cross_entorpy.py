import pdb

from torch import nn
import torch


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_soft_max = nn.LogSoftmax(dim=1)
        self.label = None

    def load_means(self, checkpoint: str):
        self.label = torch.load(checkpoint).cuda()

    def forward(self, prediction: torch.tensor, target: torch.tensor) -> torch.tensor:
        if self.label is not None:
            target = self.label[target]
        return -torch.sum(target * self.log_soft_max(prediction), dim=1).mean()


def get_mean(features: torch.tensor, labels: torch.tensor) -> torch.tensor:
    return features[labels.sort()[1]].view((labels.max() + 1, -1) + features.shape[1:]).mean(dim=1)


def main():
    pred = torch.randn(size=(1, 1000))
    label = torch.LongTensor([12])
    loss1 = SoftCrossEntropy()
    loss1.load_means('mean_prob.pt')
    loss2 = nn.CrossEntropyLoss()
    l1 = loss1(pred, label)
    l2 = loss2(pred, label)
    print(f'SXent {l1.item()}\t  Xent {l2.item()}')


if __name__ == '__main__':
    main()
