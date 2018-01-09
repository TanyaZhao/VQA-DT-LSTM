#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
from torch import optim
from torch.utils.data import DataLoader
from vocab_classes import VocabClasses
from dataset import VQADataSet
from model import SiameseNetWork,ImgWordAtt
from traintest import TrainTest
from loss import ModelLoss
import os
import time


if __name__ =="__main__":

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
        train_dataset, batch_size=1, shuffle=True, num_workers=6
    )

    test_dataset = VQADataSet(img_dir,test_dir,vocab_classes)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=6
    )

    # 定义超参数wo
    learning_rate = 0.002                            # 学习速率
    batchsize = 32                                  # batchsize
    epochs =  300                                   # epoch 数量

    vocab_size  = vocab_classes.vocab_size()        # 词汇数量
    num_classes = vocab_classes.classes_size()      # 分类数量（不同answer个数）
    ebd_dim = 150                                   # wordEmdedding 词向量维度
    mem_dim = 200                                   # TreeLSTM 输出维度
    dropout_p = 0.2                                 # Dropout系数
    att_hidd_dim = 400                              # attention时中间转换维度
    mid_dim = 200                                   # 概率分类前一Linear层维度
    mlp1_dim = 150                                  # Siamese网络最后mlp1维度
    mlp2_dim = 100                                  # Siamese网络最后mlp2维度

    model = SiameseNetWork(vocab_size,num_classes,ebd_dim, mem_dim, dropout_p,
            att_hidd_dim,mid_dim,mlp1_dim,mlp2_dim
    )

    # vgg19 = models.vgg19(pretrained=True)
    #带有batch normalization vgg19_bn
    vgg19 = models.vgg19(pretrained=True)
    # 删除vgg19特征最后一层
    # vgg19.features = nn.Sequential(*list(vgg19.features.children())[:-1])
    # print vgg19.features
    for param in vgg19.parameters():
        param.requires_grad = False
    if  torch.cuda.is_available(): #判断是否有GPU加速
        model = model.cuda()
        vgg19 = vgg19.cuda()

    # 定义loss和optimizer
    criterion = ModelLoss()
    # weight_decay 一般设置 1e-8
    optimizer = optim.RMSprop(model.parameters(),lr=learning_rate,alpha=0.8)
    # optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(0.9,0.99))

    trainTester = TrainTest(model,vgg19,criterion,optimizer,batchsize,num_classes,epochs)

    # 开始训练
    for epoch in range(epochs):
        since = time.time()

        all_loss, all_acc = trainTester.train(train_loader)
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}, Time:{:.1f} s'.format(
            epoch + 1, all_loss, all_acc, time.time() - since)
        )

        eval_loss, eval_acc, max_acc = trainTester.test(test_loader)
        print('Test Loss: {:.6f}, Test Acc: {:.6f}, Time:{:.1f} s'.format(
            eval_loss, eval_acc, time.time() - since)
        )
        print ("max Acc: {:.6f}".format(max_acc))