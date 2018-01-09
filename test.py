#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from vocab_classes import VocabClasses
from dataset import VQADataSet
import torch.nn.functional as F
import os



# 定义 Logistic Regression 模型
class AttTreeLSTM(nn.Module):
    def __init__(self, vocab_size, ebd_dim,in_dim, n_class):
        super(AttTreeLSTM, self).__init__()
        self.Vlinear1 = nn.Linear(in_dim,300)
        self.Vlinear2 = nn.Linear(300,100)

        self.emb = nn.Embedding(vocab_size, ebd_dim)
        self.lstm = nn.LSTM(ebd_dim,100,batch_first=True)

        self.logstic = nn.Linear(200, n_class)


    def forward(self, img,question,tree):
        img = img.view(img.size(0), -1)

        vlinear1 = self.Vlinear1(img)
        vlinear2 = self.Vlinear2(vlinear1)

        qinputs = self.emb(question)
        state,hidden = self.lstm(qinputs)

        contact = torch.cat([vlinear2,state[:,-1,:]],1)
        out = self.logstic(contact)
        return F.softmax(out)

# Full_DAQUAR
# base_dir = "dealed_data/DAQUAR/Full_DAQUAR/"
#Reduced_DAQUAR
base_dir = "dealed_data/DAQUAR/Reduced_DAQUAR/"

sick_dir = os.path.join(base_dir, "sick")

sick_vocab_file = os.path.join(sick_dir,"vocab.txt")
sick_classes_file = os.path.join(sick_dir,"classes.txt")

train_dir = os.path.join(sick_dir,"train/")
test_dir = os.path.join(sick_dir,"test/")
img_dir = "original_data/DAQUAR/nyu_depth_images"

# 记录语料中单词和类别的一个类
vocab_classes = VocabClasses(vocab_file=sick_vocab_file,classes_file=sick_classes_file)
#
train_dataset = VQADataSet(img_dir,train_dir,vocab_classes)
train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=1
)

# train_dataset = datasets.MNIST(
#     root='./data', train=True, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


# 定义超参数
vocab_size  = vocab_classes.vocab_size()        # 词汇数量
num_classes = vocab_classes.classes_size()      # 分类数量（不同answer个数）
ebd_dim = 100                                   # wordEmdedding 词向量维度
mem_dim = 150                                   # TreeLSTM 输出维度
att_hidd_dim = 200                              # attention时中间转换维度
learning_rate = 0.01                            # 学习率
batchsize = 32                                   # batchsize
epochs =  100                                    # epoch 数量

model = AttTreeLSTM(vocab_size,ebd_dim,3 * 28 * 28,10)

use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 开始训练
for epoch in range(epochs):
    print('*' * 10)
    print('epoch {}'.format(epoch + 1))
    running_loss = 0.0
    running_acc = 0.0
    model.eval()
    optimizer.zero_grad()
    time = 0
    for i, data in enumerate(train_loader, 1):
        img,question,answer,tree = data
        label = answer.squeeze(0)
        # #
        # img, label = data
        label[0] = label[0] % 10

        if use_gpu:
            img = Variable(img).cuda()
            question = Variable(question).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            question = Variable(question)
            label = Variable(label)
        # 向前传播
        out = model(img,question,tree)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]
        # 向后传播

        loss.backward()

        time += 1
        if time % batchsize == 0:
            optimizer.step()
            optimizer.zero_grad()


        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, epochs, running_loss / (time),
                running_acc / (time)))

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))
