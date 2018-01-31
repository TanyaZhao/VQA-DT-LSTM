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
from GloveModel import Glove

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
    learning_rate = 0.001                            # 学习速率
    batchsize = 64                                  # batchsize
    epochs =  300                                   # epoch 数量

    vocab_size  = vocab_classes.vocab_size()        # 词汇数量
    num_classes = vocab_classes.classes_size()      # 分类数量（不同answer个数）
    ebd_dim = 100                                   # wordEmdedding 词向量维度
    mem_dim = 300                                   # TreeLSTM 输出维度
    dropout_p = 0.4                                 # Dropout系数
    att_hidd_dim = 400                              # attention时中间转换维度
    mid_dim = 200                                   # 概率分类前一Linear层维度
    mlp1_dim = 150                                  # Siamese网络最后mlp1维度
    mlp2_dim = 100                                  # Siamese网络最后mlp2维度
    freeze_emb = True                              # embeding 是否在训练中调整参数, True不调整, False调整

    model = SiameseNetWork(vocab_size,num_classes,ebd_dim, mem_dim, dropout_p,
            att_hidd_dim,mid_dim,mlp1_dim,mlp2_dim,freeze_emb
    )
    # 定义loss和optimizer
    criterion = ModelLoss()
    # weight_decay 一般设置 1e-8
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=learning_rate, alpha=0.8, weight_decay=1e-4)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=learning_rate,betas=(0.9,0.99))

    emb_file = os.path.join("dealed_data", "emb_%sd_file.pth" % str(ebd_dim))
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        glove = Glove(ebd_dim)
        emb = torch.Tensor(vocab_size, ebd_dim).normal_(-0.05,0.05)
        for word in vocab_classes.wordToIdx.keys():
            emb[vocab_classes.get_vocab_index(word)] = torch.Tensor(glove.getVec(word))
        torch.save(emb, emb_file)

    if  torch.cuda.is_available(): #判断是否有GPU加速
        model = model.cuda()
        criterion = criterion.cuda()
        emb = emb.cuda()
    model.attTreeLstm.emb.weight.data.copy_(emb)

    trainTester = TrainTest(model,criterion,optimizer,batchsize,num_classes,epochs)

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