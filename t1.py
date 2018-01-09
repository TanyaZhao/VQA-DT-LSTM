#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time
# 定义超参数
import sets



def exclude(ans):
    if ans.find("_") != -1 or ans.find(',') != -1 or ans.find(" ")!= -1:
        return True
    else:
        return False
file1 = "qa.37.raw.reduced.train.txt"
ftrain = open(file1,'r')
dictTrain = {}
strain = set()
idx = 0
for ans in ftrain.readlines():
    idx += 1
    if idx % 2 == 0:
        ans = ans.strip()
        if exclude(ans):
            continue
        strain.add(ans)
        if ans in dictTrain:
            dictTrain[ans] += 1
        else:   dictTrain[ans] = 1

print len(strain)
print "-"*100

file2 = "qa.37.raw.reduced.test.txt"
ftest = open(file2,'r')
stest = set()
idx = 0
for ans in ftest.readlines():
    idx += 1
    if idx %2 == 0:
        ans = ans.strip()
        # if exclude(ans):
        #     continue
        stest.add(ans)

print len(stest)

print '-'*100
b = strain & stest
print len(b)
print '-'*100
for ans in  stest:
    print ans

# resDict ={}
# for ans in b:
#     resDict[ans] = dictTrain[ans]
#
# resDict = sorted(resDict.items(), lambda x, y: cmp(x[1], y[1]),reverse=True)
# idx = 0
# for item in resDict:
#     idx += 1
#     print idx , item


