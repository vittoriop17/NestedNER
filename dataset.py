from datetime import datetime
import argparse
import random
import pickle
import codecs
import json
import os
import nltk
import torch
import numpy as np
from pprint import pprint
from datasets import load_metric
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from terminaltables import AsciiTable


# ==================== Datasets ==================== #

# Mappings between symbols and integers, and vice versa.
# They are global for all datasets.
source_w2i = {}
source_i2w = []
target_w2i = {}
target_i2w = []

# The padding symbol will be used to ensure that all tensors in a batch
# have equal length.
PADDING_SYMBOL = ' '
source_w2i[PADDING_SYMBOL] = 0
source_i2w.append(PADDING_SYMBOL)
target_w2i[PADDING_SYMBOL] = 0
target_i2w.append(PADDING_SYMBOL)

# The special symbols to be added at the end of strings
START_SYMBOL = '<start>'
END_SYMBOL = '<end>'
UNK_SYMBOL = '<unk>'
source_w2i[START_SYMBOL] = 1
source_i2w.append(START_SYMBOL)
target_w2i[START_SYMBOL] = 1
target_i2w.append(START_SYMBOL)
source_w2i[END_SYMBOL] = 2
source_i2w.append(END_SYMBOL)
target_w2i[END_SYMBOL] = 2
target_i2w.append(END_SYMBOL)
source_w2i[UNK_SYMBOL] = 3
source_i2w.append(UNK_SYMBOL)
target_w2i[UNK_SYMBOL] = 3
target_i2w.append(UNK_SYMBOL)

SPECIAL_CHAR_IDX = [source_w2i[PADDING_SYMBOL],
                    source_w2i[START_SYMBOL],
                    source_w2i[END_SYMBOL],
                    source_w2i[UNK_SYMBOL]]

# Max number of words to be predicted if <END> symbol is not reached
MAX_PREDICTIONS = 128


def load_glove_embeddings(embedding_file):
    """
    Reads pre-made embeddings from a file
    """
    N = len(source_w2i)
    embeddings = [0] * N
    with codecs.open(embedding_file, 'r', 'utf-8') as f:
        for line in f:
            data = line.split()
            word = data[0].lower()
            if word not in source_w2i:
                source_w2i[word] = N
                source_i2w.append(word)
                N += 1
                embeddings.append(0)
            vec = [float(x) for x in data[1:]]
            D = len(vec)
            embeddings[source_w2i[word]] = vec
    # Add a '0' embedding for the padding symbol
    embeddings[0] = [0] * D
    # Check if there are words that did not have a ready-made Glove embedding
    # For these words, add a random vector
    for word in source_w2i:
        index = source_w2i[word]
        if embeddings[index] == 0:
            embeddings[index] = (np.random.random(D) - 0.5).tolist()
    return D, embeddings


class SummarizationDataset(Dataset):
    """
    A dataset with source sentences and their respective translations
    into the target language.

    Each sentence is represented as a list of word IDs.
    """

    def __init__(self, filename, max_seq_len=1024, record_symbols=True):
        try:
            nltk.word_tokenize("hi there.")
        except LookupError:
            nltk.download('punkt')
        self.source_list = []
        self.target_list = []
        self.max_seq_len = max_seq_len
        # Read the datafile
        cont_lines = -1
        with codecs.open(filename, 'r', 'utf-8') as f:
            lines = f.read().split('\n')
            for line in lines:
                cont_lines += 1
                if cont_lines == 0:
                    continue
                if len(line.strip()) == 0:
                    continue
                s, t = line.split('","')
                source_sentence = []
                for w in nltk.word_tokenize(s):
                    if w not in source_i2w and record_symbols:
                        source_w2i[w] = len(source_i2w)
                        source_i2w.append(w)
                    source_sentence.append(source_w2i.get(w, source_w2i[UNK_SYMBOL]))
                source_sentence.append(source_w2i[END_SYMBOL])
                self.source_list.append(source_sentence)
                target_sentence = []
                for w in nltk.word_tokenize(t):
                    if w not in target_i2w and record_symbols:
                        target_w2i[w] = len(target_i2w)
                        target_i2w.append(w)
                    target_sentence.append(target_w2i.get(w, target_w2i[UNK_SYMBOL]))
                target_sentence.append(target_w2i[END_SYMBOL])
                self.target_list.append(target_sentence)

    def __len__(self):
        return len(self.source_list)

    def __getitem__(self, idx):
        return self.source_list[idx], self.target_list[idx]


class PadSequence:
    """
    A callable used to merge a list of samples to form a padded mini-batch of Tensor
    """

    def __call__(self, batch, pad_source=source_w2i[PADDING_SYMBOL], pad_target=target_w2i[PADDING_SYMBOL]):
        source, target = zip(*batch)
        max_source_len = max(map(len, source))
        max_target_len = max(map(len, target))
        padded_source = [[b[i] if i < len(b) else pad_source for i in range(max_source_len)] for b in source]
        padded_target = [[l[i] if i < len(l) else pad_target for i in range(max_target_len)] for l in target]
        return padded_source, padded_target




# import torch
# import numpy as np
# import os
# from torch.utils.data import Dataset
# from torchtext.data.utils import get_tokenizer
# from collections import defaultdict
# from torchtext.vocab import Vocab
# from torchtext.utils import download_from_url, extract_archive
# import io
# from torch.nn.utils.rnn import pad_sequence
#
# SPECIALS = {
#     '<unk>': 0,
#     '<pad>': 1,
#     '<bos>': 2,
#     '<eos>': 3
# }
#
#
# class WikiHowDataset(Dataset):
#     """
#     A dataset with source articles and their respective summary
#
#     """
#     def __init__(self, titles_path, articles_path=None):
#         self.titles_path = titles_path
#         self.articles_path = articles_path if articles_path else os.path.join(os.path.dirname(titles_path), "articles")
#         self.filenames = None
#         self._read_file_names()
#         self.article_vocab, self.summary_vocab = self._build_vocab()
#
#     def _read_file_names(self):
#         self.filenames = np.loadtxt(self.titles_path, dtype=str, encoding='utf-8')
#
#     def _build_vocab(self):
#         summary_vocab = LanguageModel()
#         article_vocab = LanguageModel()
#         for filename in self.filenames:
#             filepath = os.path.join(self.articles_path, filename+".txt")
#             try:
#                 with io.open(filepath, encoding="utf8") as f:
#                     for string_ in f:
#                         if string_.startswith("@summary"):
#                             append_to_summary = True
#                         elif string_.startswith("@article"):
#                             append_to_summary = False
#                         elif append_to_summary:
#                             summary_vocab.add_sentence(string_)
#                         else:
#                             article_vocab.add_sentence(string_)
#             except IOError as e:
#                 print(f"Failed to read the current file: {filepath}")
#                 print("The execution will continue anyway")
#         return article_vocab, summary_vocab
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, item):
#         """
#         :return:  str, str: text_article, text_summary
#         """
#         filename = self.filenames[item]
#         summary, text_summary = [], ""
#         article, text_article = [], ""
#         append_to_summary = False
#         filepath = os.path.join(self.articles_path, filename + ".txt")
#         with open(filepath, "r", encoding='utf-8') as fin:
#             for line in fin.readlines():
#                 if line.startswith("@summary"):
#                     append_to_summary = True
#                 elif line.startswith("@article"):
#                     append_to_summary = False
#                 elif append_to_summary:
#                     summary.extend([self.summary_vocab.w2i[token]
#                                     if token in self.summary_vocab.w2i.keys()
#                                     else self.summary_vocab.w2i['<unk>']
#                                     for token in self.summary_vocab.tokenizer(line)])
#                     text_summary += line + ". "
#                 else:
#                     article.extend([self.article_vocab.w2i[token]
#                                     if token in self.article_vocab.w2i.keys()
#                                     else self.article_vocab.w2i['<unk>']
#                                     for token in self.article_vocab.tokenizer(line)])
#                     text_article += line + ".  "
#         # return torch.tensor(article, dtype=torch.long), torch.tensor(summary, dtype=torch.long)
#         return text_article, text_summary
#
#
# class LanguageModel:
#     def __init__(self):
#         self.w2i = {}
#         self.i2w = {}
#         self.w2count = defaultdict(lambda: 0)
#         self.n_words = 0
#         self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
#         self._add_specials()
#
#     def _add_specials(self):
#         for special, idx in SPECIALS.items():
#             self.w2i[special] = idx
#             self.i2w[idx] = special
#             self.n_words += 1
#
#     def add_sentence(self, sentence):
#         for token in self.tokenizer(sentence):
#             self._add_word(token)
#
#     def _add_word(self, word):
#         if word not in self.w2i.keys():
#             self.w2i[word] = self.n_words
#             self.i2w[self.n_words] = word
#             self.n_words += 1
#             self.w2count[word] = 1
#         self.w2count[word] += 1
#
#
# def generate_batch(data_batch):
#     articles_batch, summaries_batch = [], []
#     PAD_IDX, BOS_IDX, EOS_IDX = SPECIALS['<pad>'], SPECIALS['<bos>'], SPECIALS['<eos>']
#     for (article, summary) in data_batch:
#         articles_batch.append(torch.cat([torch.tensor([BOS_IDX]), article, torch.tensor([EOS_IDX])], dim=0))
#         summaries_batch.append(torch.cat([torch.tensor([BOS_IDX]), summary, torch.tensor([EOS_IDX])], dim=0))
#     articles_batch = pad_sequence(articles_batch, padding_value=PAD_IDX)
#     summaries_batch = pad_sequence(summaries_batch, padding_value=PAD_IDX)
#     return articles_batch, summaries_batch
