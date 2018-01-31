#__author__ = 'Administrator'
# -*- coding: utf-8 -*-

import gensim
import os
import numpy as np
class Glove(object):
    def __init__(self, dim):
        self.dim = dim
        self.glove_file = "/home/liuyun/glove/glove.6B.%sd.txt" % str(dim)
        self.glove = self.readglove(self.glove_file)

    def readglove(self,filename):
        f = open(filename, 'r')
        glove = {}
        for line in f:
            list = line.split(" ")
            word = list[0].strip()
            vec = map(float,list[1:])
            glove[word] = vec
        return glove

    def getVec(self, word):
        try:
            return self.glove[word]
        except KeyError:
            return map(float, list(np.zeros(self.dim)))    #返回0向量

if __name__ == "__main__":
    glove = Glove(200)
    print glove.getVec("the")
    print '-'*100
    print glove.getVec(".")