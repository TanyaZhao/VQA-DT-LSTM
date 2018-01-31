#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable as Var
import Constants
# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3*self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3*self.mem_dim)
        self.fx = nn.Linear(self.in_dim,self.mem_dim)
        self.fh = nn.Linear(self.mem_dim,self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):

        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1)//3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
                self.fh(child_h) +
                self.fx(inputs).repeat(len(child_h), 1)
            )
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]

        if tree.num_children==0:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))

        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)

        return tree.state

# attention 在DT-LSTM之前
# putting the whole model together
class ImgWordAtt(nn.Module):
    def __init__(self, vocab_size, ebd_dim, mem_dim, dropout_p, att_hidd_dim, freeze_emb):
        super(ImgWordAtt, self).__init__()
        self.use_cuda = torch.torch.cuda.is_available()
        self.ebd_dim = ebd_dim
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p

        self.vgg = models.vgg19(pretrained=True)
        # 删除vgg19特征最后一层
        # self.vgg.features = nn.Sequential(*list(self.vgg.features.children())[:-1])
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.emb = nn.Embedding(vocab_size, ebd_dim, padding_idx=Constants.PAD)
        if freeze_emb:
            self.emb.weight.requires_grad = False

        self.img_feature = nn.Linear(512,ebd_dim)   #512是vgg图片特征抽取之后的维度
        self.dropout = nn.Dropout(self.dropout_p)
        self.att_img = nn.Linear(ebd_dim,att_hidd_dim,bias=False)    #49是图片区域数量7x7
        self.att_word = nn.Linear(ebd_dim,att_hidd_dim,bias=True)
        self.att = nn.Linear(att_hidd_dim,1,bias=True)

        self.childsumtreelstm = ChildSumTreeLSTM( 2 * ebd_dim, mem_dim)


    def forward(self, img, question, tree):

        img_feature = self.vgg.features(img)   # 1x512x7x7
        img_feature = img_feature.squeeze(0)     # 512x7x7
        img_feature = torch.t(img_feature.view(img_feature.size(0),-1)) # 49 x 512

        img_feature = F.relu(self.img_feature(img_feature))     #49 x word_embedding
        img_feature = self.dropout(img_feature)

        question = question.squeeze(0)
        # img Attention后和question拼接后作为ChildSumTreeLSTM输入
        attedInput = torch.Tensor(question.size(0), 2*self.ebd_dim).normal_(-0.05,0.05)
        attedInput = Var(attedInput)
        if self.use_cuda:
            attedInput = attedInput.cuda()

        attImg = self.att_img(img_feature)       # 49 x att_hidd_dim
        for idx in range(question.size(0)):
            word_emd = self.emb(question[idx])
            attWord = self.att_word(word_emd)    # 1 x att_hidd_dim

            addImgWord = torch.zeros(attImg.size(0),attImg.size(1))    # 49 x att_hidd_dim
            addImgWord = Var(addImgWord)

            new_img_feature = Var(torch.zeros(self.ebd_dim))
            if self.use_cuda:
               addImgWord = addImgWord.cuda()
               new_img_feature = new_img_feature.cuda()

            for i in range(attImg.size(0)):
                addImgWord[i] = F.tanh(attImg[i] + attWord)
            att_weight = self.att(addImgWord).squeeze(1)     # 49 x 1
            att_weight = F.softmax(att_weight)

            for i in range(att_weight.size(0)):
                vec = att_weight.data[i] * img_feature.data[i]
                new_img_feature.data = new_img_feature.data + vec

            new_img_feature = new_img_feature / att_weight.size(0)
            new_img_feature = new_img_feature.view(1,-1)
            attedInput[idx] = torch.cat([new_img_feature,word_emd],1)

        attedInput = self.dropout(attedInput)
        state, hidden = self.childsumtreelstm(tree, attedInput)
        return state

class SiameseNetWork(nn.Module):
    def __init__(self,vocab_size, num_classes, ebd_dim, mem_dim,dropout_p,
                 att_hidd_dim,mid_dim,mlp1_dim,mlp2_dim, freeze_emb):
        super(SiameseNetWork,self).__init__()
        self.attTreeLstm = ImgWordAtt(vocab_size, ebd_dim, mem_dim, dropout_p, att_hidd_dim, freeze_emb)
        self.dropout = nn.Dropout(dropout_p)
        self.mlp1 = nn.Linear(mem_dim,mlp1_dim)
        self.mlp2 = nn.Linear(mlp1_dim,mlp2_dim)

        self.linear = nn.Linear(mem_dim,mid_dim)
        self.classifyProb = nn.Linear(mid_dim,num_classes)

    def forward(self, img, question, tree, neg_img):
        posPairOut = self.dropout(self.attTreeLstm(img, question, tree))
        negPairOut = self.dropout(self.attTreeLstm(neg_img,question,tree))

        posScoreVec = F.relu(self.mlp1(posPairOut))
        posScoreVec = F.relu(self.mlp2(posScoreVec))

        negScoreVec = F.relu(self.mlp1(negPairOut))
        negScoreVec = F.relu(self.mlp2(negScoreVec))

        probOut = F.relu(self.linear(posPairOut))
        probOut = F.softmax(self.classifyProb(probOut))

        return probOut,posScoreVec,negScoreVec

# attention 在 DT-LSTM之后
# class ImgWordAtt(nn.Module):
#     def __init__(self, vocab_size, ebd_dim, mem_dim,att_hidd_dim):
#         super(ImgWordAtt, self).__init__()
#         self.ebd_dim = ebd_dim
#         self.mem_dim = mem_dim
#         self.vocab_size = vocab_size
#         self.emb = nn.Embedding(vocab_size, ebd_dim,padding_idx=Constants.PAD)
#
#         self.img_feature = nn.Linear(512,mem_dim)   #512是vgg19图片特征抽取之后的维度
#
#         self.att_img = nn.Linear(mem_dim,att_hidd_dim,bias=False)    #49是图片区域数量7x7
#         self.att_word = nn.Linear(mem_dim,att_hidd_dim,bias=True)
#         self.att = nn.Linear(att_hidd_dim,1,bias=True)
#
#         self.childsumtreelstm = ChildSumTreeLSTM( ebd_dim, mem_dim)
#         # self.childsumtreelstm = ChildSumTreeLSTM_(self.vocab_size, ebd_dim, mem_dim)
#
#         self.lastLayer = nn.Linear(2 * mem_dim, mem_dim)
#
#     def forward(self, img_feature, question, tree):
#
#         img_feature = F.relu(self.img_feature(img_feature))     #49 x mem_dim
#         question = question.squeeze(0)
#         wordEmds = self.emb(question)
#         state,hidden = self.childsumtreelstm(tree,wordEmds)
#
#         attImg = self.att_img(img_feature)       # 49 x att_hidd_dim
#         attWords = self.att_word(state)
#
#         addImgWord = torch.zeros(attImg.size(0),attImg.size(1))    # 49 x att_hidd_dim
#         addImgWord = Var(addImgWord,requires_grad=True)
#
#         new_img_feature = Var(torch.zeros(self.mem_dim),requires_grad=True)
#         if torch.cuda.is_available():
#            addImgWord = addImgWord.cuda()
#            new_img_feature = new_img_feature.cuda()
#
#         for i in range(attImg.size(0)):
#             addImgWord[i] = F.tanh(attImg[i] + attWords)
#         att_weight = self.att(addImgWord).squeeze(1)     # 49 x 1
#         att_weight = F.softmax(att_weight)
#
#         for i in range(att_weight.size(0)):
#             vec = att_weight.data[i] * img_feature.data[i]
#             new_img_feature.data = new_img_feature.data + vec
#         new_img_feature = new_img_feature.view(1,-1)
#         contact_feature = torch.cat([new_img_feature,state],1)
#
#         c2mem_dim = F.relu(self.lastLayer(contact_feature))
#         return c2mem_dim

