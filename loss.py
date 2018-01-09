#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
from torch import nn
class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss,self).__init__()
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.MarginRankingLoss()

    def forward(self,probOut,label,posScoreVec,negScoreVec,y):
        lossAns = self.criterion1(probOut,label) + self.criterion2(posScoreVec,negScoreVec,y)
        return lossAns