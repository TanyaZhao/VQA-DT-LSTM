#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
# vocab object from harvardnlp/opennmt-py
"""
# 记录语料中单词和类别的一个类
"""
class VocabClasses(object):
    def __init__(self, vocab_file=None, classes_file=None,sqecial_data=None, lower=True):
        self.idxToWord = {}
        self.wordToIdx = {}
        self.idxToClass = {}
        self.classToIdx = {}

        self.lower = lower
        self.unkWord = "unkWord"
        # Special entries will not be pruned.
        self.special = []

        if sqecial_data is not None:
            self.addSpecials(sqecial_data)

        if vocab_file  is not None:
            self.load_vocab_file(vocab_file)
        self.add_vocab(self.unkWord)

        if classes_file is not None:
            self.load_classes_file(classes_file)

    def vocab_size(self):
        return len(self.idxToWord)

    def classes_size(self):
        return len(self.idxToClass)

    # Load entries from a file.
    def load_vocab_file(self, filename):
        for line in open(filename):
            token = line.rstrip('\n')
            self.add_vocab(token)

    def load_classes_file(self, filename):
        for line in open(filename):
            token = line.rstrip('\n')
            self.add_class(token)

    def get_vocab_index(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.wordToIdx[key]
        except KeyError:
            return default

    def get_vocab_word(self, idx, default=None):
        try:
            return self.idxToWord[idx]
        except KeyError:
            return default

    def get_classes_index(self, key, default=None):
        try:
            return self.classToIdx[key]
        except KeyError:
            return default

    def get_classes_word(self, idx, default=None):
        try:
            return self.idxToClass[idx]
        except KeyError:
            return default

    # Mark this `label` and `idx` as special
    def addSpecial(self, word, idx=None):
        idx = self.add_vocab(word)
        self.special += [idx]

    # Mark all labels in `labels` as specials
    def addSpecials(self, words):
        for word in words:
            self.addSpecial(word)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add_vocab(self, word):
        word = word.lower() if self.lower else word
        if word in self.wordToIdx:
            idx = self.wordToIdx[word]
        else:
            idx = len(self.idxToWord)
            self.idxToWord[idx] = word
            self.wordToIdx[word] = idx
        return idx

    def add_class(self, cls):
        if cls in self.classToIdx:
            idx = self.classToIdx[cls]
        else:
            idx = len(self.idxToClass)
            self.idxToClass[idx] = cls
            self.classToIdx[cls] = idx
        return idx

    # Convert `wordsList` to indices. Use `unkWord` if not found.
    def convertToIdx(self, wordsList):
        vec = []
        unk = self.get_vocab_index(self.unkWord)
        vec += [self.get_vocab_index(word, default=unk) for word in wordsList]
        return vec

    # Convert `idx` to labels.
    def convertToWords(self, idx):
        wordsList = []
        for i in idx:
            wordsList += [self.get_vocab_word(i)]
        return wordsList


