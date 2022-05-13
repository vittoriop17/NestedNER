import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from collections import defaultdict
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io
from torch.nn.utils.rnn import pad_sequence

SPECIALS = {
    '<unk>': 0,
    '<pad>': 1,
    '<bos>': 2,
    '<eos>': 3
}


class WikiHowDataset(Dataset):
    """
    A dataset with source articles and their respective summary

    """
    def __init__(self, titles_path, articles_path=None):
        self.titles_path = titles_path
        self.articles_path = articles_path if articles_path else os.path.join(os.path.dirname(titles_path), "articles")
        self.filenames = None
        self._read_file_names()
        self.article_vocab, self.summary_vocab = self._build_vocab()

    def _read_file_names(self):
        self.filenames = np.loadtxt(self.titles_path, dtype=str, encoding='utf-8')

    def _build_vocab(self):
        summary_vocab = LanguageModel()
        article_vocab = LanguageModel()
        for filename in self.filenames:
            filepath = os.path.join(self.articles_path, filename+".txt")
            try:
                with io.open(filepath, encoding="utf8") as f:
                    for string_ in f:
                        if string_.startswith("@summary"):
                            append_to_summary = True
                        elif string_.startswith("@article"):
                            append_to_summary = False
                        elif append_to_summary:
                            summary_vocab.add_sentence(string_)
                        else:
                            article_vocab.add_sentence(string_)
            except IOError as e:
                print(f"Failed to read the current file: {filepath}")
                print("The execution will continue anyway")
        return article_vocab, summary_vocab

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        """
        :return:  str, str: text_article, text_summary
        """
        filename = self.filenames[item]
        summary, text_summary = [], ""
        article, text_article = [], ""
        append_to_summary = False
        filepath = os.path.join(self.articles_path, filename + ".txt")
        with open(filepath, "r", encoding='utf-8') as fin:
            for line in fin.readlines():
                if line.startswith("@summary"):
                    append_to_summary = True
                elif line.startswith("@article"):
                    append_to_summary = False
                elif append_to_summary:
                    summary.extend([self.summary_vocab.w2i[token]
                                    if token in self.summary_vocab.w2i.keys()
                                    else self.summary_vocab.w2i['<unk>']
                                    for token in self.summary_vocab.tokenizer(line)])
                    text_summary += line + ". "
                else:
                    article.extend([self.article_vocab.w2i[token]
                                    if token in self.article_vocab.w2i.keys()
                                    else self.article_vocab.w2i['<unk>']
                                    for token in self.article_vocab.tokenizer(line)])
                    text_article += line + ".  "
        # return torch.tensor(article, dtype=torch.long), torch.tensor(summary, dtype=torch.long)
        return text_article, text_summary


class LanguageModel:
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.w2count = defaultdict(lambda: 0)
        self.n_words = 0
        self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self._add_specials()

    def _add_specials(self):
        for special, idx in SPECIALS.items():
            self.w2i[special] = idx
            self.i2w[idx] = special
            self.n_words += 1

    def add_sentence(self, sentence):
        for token in self.tokenizer(sentence):
            self._add_word(token)

    def _add_word(self, word):
        if word not in self.w2i.keys():
            self.w2i[word] = self.n_words
            self.i2w[self.n_words] = word
            self.n_words += 1
            self.w2count[word] = 1
        self.w2count[word] += 1


def generate_batch(data_batch):
    articles_batch, summaries_batch = [], []
    PAD_IDX, BOS_IDX, EOS_IDX = SPECIALS['<pad>'], SPECIALS['<bos>'], SPECIALS['<eos>']
    for (article, summary) in data_batch:
        articles_batch.append(torch.cat([torch.tensor([BOS_IDX]), article, torch.tensor([EOS_IDX])], dim=0))
        summaries_batch.append(torch.cat([torch.tensor([BOS_IDX]), summary, torch.tensor([EOS_IDX])], dim=0))
    articles_batch = pad_sequence(articles_batch, padding_value=PAD_IDX)
    summaries_batch = pad_sequence(summaries_batch, padding_value=PAD_IDX)
    return articles_batch, summaries_batch
