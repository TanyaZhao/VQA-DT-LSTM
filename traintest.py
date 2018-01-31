#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
from tqdm import tqdm
from tree import Tree
import torch
from torch.autograd import Variable
import time
class TrainTest(object):

    def __init__(self,model,criterion,optimizer,batchsize,num_classes,epochs):
        super(TrainTest,self).__init__()
        self.model  = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batchsize = batchsize
        self.num_classes = num_classes
        self.use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
        self.max_acc = 0
        self.epochs = epochs
        self.epoch = 0

    def read_tree(self, parents):
        # 传入进来的parents是tensor,转化为list
        parents = parents.numpy().tolist()
        # parents = list(map(int,line.split()))
        trees = dict()
        root = None
        for i in range(1,len(parents)+1):
            if i-1 not in trees.keys() and parents[i-1]!=-1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx-1] = tree
                    tree.idx = idx-1
                    if parent-1 in trees.keys():
                        trees[parent-1].add_child(tree)
                        break
                    elif parent==0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def train(self,train_loader):
        print('*' * 100)
        print('epoch {}'.format(self.epoch + 1))
        running_loss = 0.0
        running_acc = 0.0
        self.model.train()
        self.optimizer.zero_grad()
        idx = 0
        since = time.time()
        for i, data in enumerate(train_loader, 1):
            img,question,answer,treeTensor,negImg = data
            treeTensor = treeTensor.squeeze(0)
            label = answer.squeeze(0)
            tree = self.read_tree(treeTensor)

            y = torch.ones(1,1)
            if self.use_gpu:
                img = Variable(img).cuda()
                question = Variable(question).cuda()
                label = Variable(label).cuda()
                negImg = Variable(negImg).cuda()
                y = Variable(y).cuda()
            else:
                img = Variable(img)
                question = Variable(question)
                label = Variable(label)
                negImg = Variable(negImg)
                y = Variable(y)

            probOut,posScoreVec,negScoreVec = self.model(img,question,tree,negImg)
            # loss = criterion(probOut, label,posScoreVec,negScoreVec,y)
            loss = self.criterion(probOut,label,posScoreVec,negScoreVec,y)

            running_loss += loss.data[0] * label.size(0)
            _, pred = torch.max(probOut, 1)
            num_correct = (pred == label).sum()
            running_acc += num_correct.data[0]

            # 向后传播
            loss.backward()

            idx += 1

            if idx % self.batchsize == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                # print '{}, Time:{:.1f} s'.format(idx, time.time() - since)

            if i % 300 == 0:
                print('[{}/{}] Train Loss: {:.6f}, Train Acc: {:.6f}, Time:{:.1f} s'.format(
                    self.epoch + 1, self.epochs, running_loss / (idx),
                    running_acc / (idx), time.time() - since)
                )

        self.epoch += 1
        all_loss = 1.0 * running_loss / (len(train_loader))
        all_acc = 1.0 * running_acc / (len(train_loader))

        return all_loss, all_acc

    def test(self,test_loader):
        self.model.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        for data in test_loader:
            img,question,answer,treeTensor,negImg = data
            treeTensor = treeTensor.squeeze(0)
            label = answer.squeeze(0)
            tree = self.read_tree(treeTensor)
            y = torch.ones(1,1)

            if self.use_gpu:
                img = Variable(img, volatile=True).cuda()
                question = Variable(question, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
                negImg = Variable(negImg, volatile=True).cuda()
                y = Variable(y,volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                question = Variable(question, volatile=True)
                label = Variable(label, volatile=True)
                negImg = Variable(negImg,volatile=True)
                y = Variable(y,volatile=True)


            probOut,posScoreVec,negScoreVec = self.model(img,question,tree,negImg)
            loss = self.criterion(probOut,label,posScoreVec,negScoreVec,y)

            eval_loss += loss.data[0] * label.size(0)
            _,pred = torch.max(probOut,1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.data[0]

        eval_loss = 1.0 * eval_loss /len(test_loader)
        eval_acc = 1.0 * eval_acc /len(test_loader)
        if eval_acc > self.max_acc:
            self.max_acc = eval_acc

        return eval_loss, eval_acc, self.max_acc