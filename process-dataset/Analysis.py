
import argparse
import string
import seaborn as sn
import codecs
import csv
from tqdm import tqdm
from terminaltables import AsciiTable
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from statistics import mean





class LoadDataset(Dataset):
    """
    A class loading 
     NER dataset from a CSV file to be used as an input to PyTorch DataLoader.
    """
    def __init__(self, filename):
        reader = csv.reader(codecs.open(filename, encoding='ascii', errors='ignore'), delimiter=',')
        
        self.sentences = []
        self.labels = []
        
        sentence, labels = [], []
        for row in reader:
                self.sentences.append(row[0])
                self.labels.append(row[1])

    def analysis(self):
        no_of_tokens=dict()
        temp=list()

        line_to_text=list()
        line_to_summary=list()

        for line in self.sentences[1:]:
            new_string = line.translate(str.maketrans('', '', string.punctuation))
            no_of_tokens[(len(new_string.split()))]=no_of_tokens.get((len(new_string.split())),0)+1
            line_to_text.append(len(new_string.split()))
            

        

        print(len(no_of_tokens.values()))

        
        tokens = list(no_of_tokens.keys())
        texts = list(no_of_tokens.values())

     

        plt.bar(tokens, texts,width=20)
        #plt.xticks((1000,2000,3000,4000,5000), fontsize = 18)
        plt.xticks((0,100,200,300,400,500,600,700,800), fontsize = 18)
        plt.xlabel('tokens')
        plt.ylabel('texts')
        plt.show()

    
        
    
    
        for line in self.labels[1:] :
            new_string = line.translate(str.maketrans('', '', string.punctuation))
            no_of_tokens[(len(new_string.split()))]=no_of_tokens.get((len(new_string.split())),0)+1
            line_to_summary.append((len(new_string.split())))

        

        

        
        tokens = list(no_of_tokens.keys())
        texts = list(no_of_tokens.values())

        plt.bar(range(len(tokens)), texts,width=2)
        plt.xticks((0,100,200,300,400,500,600,700,800), fontsize = 18)
        plt.xlabel('tokens')
        plt.ylabel('summaries')
        plt.show()

        res = [i / j for i, j in zip(line_to_summary, line_to_text)]

        plt.hist(res, density=True, bins=30)
        plt.show()
        print(mean(res))

        #plt.bar( range(len(res)),height=res)
        #plt.show()








                    
                
                







if __name__ == '__main__':
    training_data = LoadDataset('test__text_summary.csv')
    training_data.analysis()