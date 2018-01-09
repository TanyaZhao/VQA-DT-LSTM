#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
"""
利用stanford parse解析语料，并将解析后的树形结构存储起来
这个文件不能单独执行，需要在linux 下执行./process.sh文件
"""

import os
import glob

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath =  os.path.join(dirpath, filepre + '.rels')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'img.txt'), 'w') as img_file, \
         open(os.path.join(dst_dir, 'question.txt'), 'w') as question_file,  \
         open(os.path.join(dst_dir, 'answer.txt'), 'w') as answer_file:
            # datafile.readline()
            for line in datafile:
                img, question, answer = line.strip().split('\t')
                img_file.write(img + '\n')
                question_file.write(question + '\n')
                answer_file.write(answer + '\n')
    img_file.close()
    question_file.close()
    answer_file.close()

def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'question.txt'), cp=cp, tokenize=True)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing VQA dataset')
    print('=' * 80)
    # Full_DAQUAR
    # base_dir = "dealed_data/DAQUAR/Full_DAQUAR"
    # Reduced_DAQUAR
    base_dir = "dealed_data/DAQUAR/Reduced_DAQUAR"
    lib_dir =  "lib/"

    sick_dir = os.path.join(base_dir, 'sick')


    train_dir = os.path.join(sick_dir, 'train')
    test_dir = os.path.join(sick_dir, 'test')
    make_dirs([sick_dir,train_dir, test_dir])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')]
    )

    # split into separate files
    split(os.path.join(base_dir, 'qa_train.txt'), train_dir)
    split(os.path.join(base_dir, 'qa_test.txt'), test_dir)

    # parse sentences
    parse(train_dir, cp=classpath)
    parse(test_dir, cp=classpath)


    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(sick_dir, '*/*.toks')),
        os.path.join(sick_dir, 'vocab.txt'))

    build_vocab(
        glob.glob(os.path.join(sick_dir, '*/*.toks')),
        os.path.join(sick_dir, 'vocab-cased.txt'),
        lowercase=False)
    # get classes
    build_vocab(
        glob.glob(os.path.join(train_dir,'answer.txt')),
        os.path.join(sick_dir, 'classes.txt'))