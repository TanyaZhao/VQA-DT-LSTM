#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
"""
数据处理，将VQADATA 以 img question answer 的格式存入文本中，同时统计信息
"""
import sets

trainClassSet = set()
testClassSet = set()

classSet =  set()

def addTrainclass(answer):
    answer = answer.strip()
    global trainClassSet
    if answer.find(",") ==  -1 :
        cls = answer
        if cls not in trainClassSet:
            trainClassSet.add(cls)
        return True
    return False

def addTestclass(answer):
    answer = answer.strip()
    global testClassSet
    if answer.find(",") ==  -1 :
        cls = answer
        if cls not in testClassSet:
            testClassSet.add(cls)
        return True
    return False

def inClassSet(answer):
    answer = answer.strip()
    if answer.find(",") == -1 :
        cls = answer
        if cls in classSet:
            return True
    return False

def dealData(train_test_file_path,train_file_path,test_file_fath,
             deal_train_path,deal_test_path):
    ftrain = open(deal_train_path,'w')
    ftest = open(deal_test_path,'w')


    file = open(test_file_fath,'r')
    file_content = file.readlines()
    for idx in range(len(file_content)):
        line = file_content[idx].strip()
        if idx % 2 == 1:
            answer = line
            addTrainclass(answer)

    file = open(train_file_path,'r')
    file_content = file.readlines()
    for idx in range(len(file_content)):
        line = file_content[idx].strip()
        if idx % 2 == 1:
            answer = line
            addTestclass(answer)

    global classSet
    classSet = trainClassSet & testClassSet

    test_count = 0
    file = open(test_file_fath,'r')
    file_content = file.readlines()
    test_pair_count = len(file_content)/2
    for idx in range(len(file_content)):
        line = file_content[idx].strip()
        if idx % 2 == 0:
            question = line
        if idx % 2 == 1:
            answer = line
            if inClassSet(answer):
                test_count += 1
                img = question.strip().split(" ")[-2]
                ftest.write(img + '\t'+ question + '\t' + answer + '\n')

    train_count = 0
    file = open(train_file_path,'r')
    file_content = file.readlines()
    train_pair_count = len(file_content)/2
    question = ""
    for idx in range(len(file_content)):
        line = file_content[idx].strip()
        if idx % 2 == 0:
            question = line
        if idx % 2 == 1:
            answer = line
            if inClassSet(answer):
                train_count += 1
                img = question.strip().split(" ")[-2]
                ftrain.write(img + '\t'+ question + '\t' + answer + '\n')


    all_pair_count = train_pair_count + test_pair_count
    file.close()
    ftrain.close()
    ftest.close()

    print 'original all question answer pair count: ',all_pair_count
    print 'original train question answer pair count: ',train_pair_count
    print 'original test question answer pair count: ',test_pair_count

    print 'train count: ', train_count
    print 'test count: ', test_count

    print 'classes count:', len(classSet)

    # original all question answer pair count:  12468
    # original train question answer pair count:  6795
    # original test question answer pair count:  5673
    # train count:  6149
    # test count:  4949
    # classes count: 459


if __name__ =="__main__":
    train_test_file_path = 'original_data/DAQUAR/Full_DAQUAR/Question answer pairs - train + test.txt'
    train_file_path = 'original_data/DAQUAR/Full_DAQUAR/Question answer pairs - train.txt'
    test_file_fath = 'original_data/DAQUAR/Full_DAQUAR/Question answer pairs - test.txt'

    deal_train_path = 'dealed_data/DAQUAR/Full_DAQUAR/qa_train.txt'
    deal_test_path = 'dealed_data/DAQUAR/Full_DAQUAR/qa_test.txt'


    # train_test_file_path = 'original_data/DAQUAR/Reduced_DAQUAR/Question answer pairs - train + test.txt'
    # train_file_path = 'original_data/DAQUAR/Reduced_DAQUAR/Question answer pairs - train.txt'
    # test_file_fath = 'original_data/DAQUAR/Reduced_DAQUAR/Question answer pairs - test.txt'
    #
    # deal_train_path = 'dealed_data/DAQUAR/Reduced_DAQUAR/qa_train.txt'
    # deal_test_path = 'dealed_data/DAQUAR/Reduced_DAQUAR/qa_test.txt'

    dealData(train_test_file_path,train_file_path,test_file_fath,
             deal_train_path,deal_test_path)

