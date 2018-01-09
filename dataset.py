#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import os
import torch
import torch.utils.data as data
from torchvision import transforms
from tree import Tree
from PIL import Image
import random
class VQADataSet(data.Dataset):
    def __init__(self, img_dir, path, vocab_classes):
        super(VQADataSet, self).__init__()

        self.vocab_class = vocab_classes
        self.img_dir = img_dir
        self.transform = self.img_transform()

        self.imgs = self.read_images(os.path.join(path,'img.txt'))
        self.questions = self.read_questions(os.path.join(path,'question.toks'))
        self.answers = self.read_answers(os.path.join(path,'answer.txt'))
        self.trees = self.read_trees(os.path.join(path,'question.parents'))

        self.size = len(self.answers)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        randInt = random.randint(0,self.size-1)
        while self.imgs[index] == self.imgs[randInt]:
            randInt = random.randint(0,self.size-1)

        img = self.read_image(self.imgs[index])
        negImg = self.read_image(self.imgs[randInt])
        question = self.read_question(self.questions[index])
        answer = self.read_answer(self.answers[index])
        treeTensor = torch.IntTensor(map(int,self.trees[index].split()))

        return (img,question,answer,treeTensor,negImg)

    def img_transform(self):
        transform = transforms.Compose([
            transforms.Scale((224,224)),
            # transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)) #bring images to (-1,1)
            transforms.Normalize(mean = (0.485, 0.456, 0.406), std =(0.229, 0.224, 0.225))  #bring images to (0,1)
        ])
        return transform

    def read_images(self,filename):
        with open(filename,'r') as f:
            imgs = [line.strip() for line in f.readlines()]
        return imgs

    def read_image(self,imgName):

        img_path = os.path.join(self.img_dir, imgName + ".png")
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img

    def read_questions(self, filename):
        with open(filename,'r') as f:
            questions = [line.strip() for line in f.readlines()]
        return questions

    def read_question(self, line):
        indices = self.vocab_class.convertToIdx(line.split())
        return torch.LongTensor(indices)

    def read_answers(self, filename):
        with open(filename,'r') as f:
            answers = [line.strip() for line in f.readlines()]
        return answers

    def read_answer(self, ans):
        indice = [self.vocab_class.get_classes_index(ans)]
        return torch.LongTensor(indice)

    def read_trees(self, filename):
        with open(filename,'r') as f:
            trees = [line.strip() for line in f.readlines()]
        return trees

    def read_tree(self, line):
        parents = list(map(int,line.split()))
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