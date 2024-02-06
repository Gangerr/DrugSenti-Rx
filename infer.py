# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from data_utils import ABSADatesetReader, ABSADataset, Tokenizer, build_embedding_matrix
from bucket_iterator import BucketIterator
from models import LSTM, AIGCN, CNN
from dependency_graph import dependency_adj_matrix

class Inferer:
    def __init__(self, opt):
        self.opt = opt
        fname = {
            'drug': {
                'train': './datasets/UCI-ml/train.raw',
                'test': './datasets/UCI-ml/test.raw'
            },
        }
        if os.path.exists(opt.dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(opt.dataset))
            with open(opt.dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 self.tokenizer = Tokenizer(word2idx=word2idx)
        else:
            print("reading {0} dataset...".format(opt.dataset))
            
            text = ABSADatesetReader.__read_text__([fname[opt.dataset]['train'], fname[opt.dataset]['test']])
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_text(text)
            with open(opt.dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(self.tokenizer.word2idx, f)
        embedding_matrix = build_embedding_matrix(self.tokenizer.word2idx, opt.embed_dim, opt.dataset)
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_text, aspect):
        text_seqs = [self.tokenizer.text_to_sequence(raw_text.lower())]
        aspect_seqs = [self.tokenizer.text_to_sequence(aspect.lower())]
        left_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().split(aspect.lower())[0])]
        text_indices = torch.tensor(text_seqs, dtype=torch.int64)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64)
        left_indices = torch.tensor(left_seqs, dtype=torch.int64)
        dependency_graph = torch.tensor(np.array([dependency_adj_matrix(raw_text.lower())]), dtype=torch.float32)
        data = {
              'text_indices': text_indices, 
              'aspect_indices': aspect_indices,
              'left_indices': left_indices, 
              'dependency_graph': dependency_graph
          }
        t_inputs = [data[col].to(opt.device) for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs


if __name__ == '__main__':
    dataset = 'drug'
    model_state_dict_paths = {
        'lstm': 'state_dict/lstm_'+dataset+'.pkl',
        'cnn': 'state_dict/cnn_'+dataset+'.pkl',
        'aigcn': 'state_dict/aigcn_'+dataset+'.pkl',
    }
    model_classes = {
        'lstm': LSTM,
        'cnn': CNN,
        'aigcn': AIGCN,
    }
    input_colses = {
        'lstm': ['text_indices'],
        'cnn': ['text_indices', 'aspect_indices', 'left_indices'],
        'aigcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
    }
    class Option(object): pass
    opt = Option()
    opt.model_name = 'aigcn'
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.dataset = dataset
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.polarities_dim = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)

    # Get user input for sentence and aspect
    raw_text = input("Enter the sentence: ")
    aspect = input("Enter the aspect: ")

    # Perform sentiment analysis
    t_probs = inf.evaluate(raw_text, aspect)

    # Print the predicted sentiment label
    print("Predicted Sentiment Label:", t_probs.argmax(axis=-1)[0])
