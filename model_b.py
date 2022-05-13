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


class TranslationDataset(Dataset):
    """
    A dataset with source sentences and their respective translations
    into the target language.

    Each sentence is represented as a list of word IDs.
    """

    def __init__(self, filename, record_symbols=True):
        try:
            nltk.word_tokenize("hi there.")
        except LookupError:
            nltk.download('punkt')
        self.source_list = []
        self.target_list = []
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


# ==================== Encoder ==================== #

class EncoderRNN(nn.Module):
    """
    Encodes a batch of source sentences.
    """

    def __init__(self, no_of_input_symbols, embeddings=None, embedding_size=16, hidden_size=25,
                 encoder_bidirectional=False, device='cpu', use_gru=False, tune_embeddings=False, **kwargs):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.is_bidirectional = encoder_bidirectional
        self.embedding = nn.Embedding(no_of_input_symbols, embedding_size)
        if embeddings != None:
            self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float),
                                                 requires_grad=tune_embeddings)
        if use_gru:
            self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=self.is_bidirectional)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True, bidirectional=self.is_bidirectional)
        self.device = device
        self.to(device)

    def set_embeddings(self, embeddings):
        self.embedding.weight = torch.tensor(embeddings, dtype=torch.float)

    def forward(self, x):
        """
        x is a list of lists of size (batch_size,max_seq_length)
        Each inner list contains word IDs and represents one sentence.
        The whole list-of-lists represents a batch of sentences.

        Returns:
        the output from the encoder RNN: a pair of two tensors, one containing all hidden states, and one
        containing the last hidden state (see https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        """
        B, T = len(x), len(x[0])
        word_embeddings = self.embedding(torch.tensor(x).to(self.device)).view(B, T, self.embedding_size)
        output, h_n = self.rnn(word_embeddings)
        return output, h_n


# ==================== Decoder ==================== #

class DecoderRNN(nn.Module):

    def __init__(self, no_of_output_symbols, embedding_size=16, hidden_size=25, use_attention=True,
                 display_attention=False, device='cpu', use_gru=False):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(no_of_output_symbols, embedding_size)
        self.no_of_output_symbols = no_of_output_symbols
        self.W = nn.Parameter(torch.rand(hidden_size, hidden_size) - 0.5)
        self.U = nn.Parameter(torch.rand(hidden_size, hidden_size) - 0.5)
        self.v = nn.Parameter(torch.rand(hidden_size, 1) - 0.5)
        self.use_attention = use_attention
        self.display_attention = display_attention
        if use_gru:
            self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, no_of_output_symbols)
        self.device = device
        self.to(device)

    def forward(self, inp, hidden, encoder_outputs):
        """
        'input' is a list of length batch_size, containing the current word
        of each sentence in the batch

        'hidden' is a tensor containing the last hidden state of the decoder,
        for each sequence in the batch
        hidden.shape = (1, batch_size, hidden_size)

        'encoder_outputs' is a tensor containing all hidden states from the
        encoder (used in problem c)
        encoder_outputs.shape = (batch_size, max_seq_length, hidden_size)

        Note that 'max_seq_length' above refers to the max_seq_length
        of the encoded sequence (not the decoded sequence).

        Returns:
        If use_attention and display_attention are both True (task (c)), return a triple
        (logits for the predicted next word, hidden state, attention weights alpha)

        Otherwise (task (b)), return a pair
        (logits for the predicted next word, hidden state).
        """
        B = len(inp)
        word_embeddings = self.embedding(torch.tensor(inp).to(self.device)).view(B, 1, -1)
        if self.use_attention:
            context, alpha = self._compute_context(encoder_outputs, hidden)
            context = context.view(1, B, -1)
            rnn_output, h_n = self.rnn(word_embeddings, context)
        else:
            rnn_output, h_n = self.rnn(word_embeddings, hidden)
        if self.use_attention and self.display_attention:
            return self.output(rnn_output), h_n, alpha
        else:
            return self.output(rnn_output), h_n

    def _compute_context(self, encoder_hidden_states, last_state):
        # compute e_ij = v' * tanh(WH + Us)
        # actually the coeffs e_ij are computed all at the same time. For the whole batch
        B, MAX_SEQ_LEN, ENCODER_HIDDEN_SIZE = encoder_hidden_states.shape
        # encoder_hidden_states shape: B, MAX_SEQ_LEN, HIDDEN_SIZE * 2
        # TODO - e.shape: B, HIDDEN_SIZE, MAX_SEQ_LEN
        e = torch.bmm(self.v.T.repeat(B, 1, 1),
                      torch.tanh((torch.bmm(self.W.repeat(B, 1, 1), encoder_hidden_states.permute(0, 2, 1)) +
                                  (torch.bmm(self.U.repeat(B, 1, 1), last_state.permute(1, 2, 0))))))
        # e = (torch.tanh(((encoder_hidden_states @ self.W.T) + (last_state @ self.U.T).permute(1, 0, 2))) @ self.v).permute(0, 2, 1)
        # compute alphas
        # TODO - alpha.shape: B, 1, MAX_SEQ_LEN
        # TODO - encoder_hidden_states.shape: B, MAX_SEQ_LEN, HIDDEN_SIZE * 2
        exp_e = torch.exp(e)
        alpha = exp_e / torch.sum(exp_e, dim=-1, keepdim=True)
        c = (alpha * encoder_hidden_states.permute(0, 2, 1)).sum(dim=-1)
        return c, alpha


# ======================================== #

def evaluate(ds, encoder, decoder):
    confusion = [[0 for a in target_i2w] for b in target_i2w]
    correct_sentences, incorrect_sentences = 0, 0
    for x, y in ds:
        predicted_sentence = []
        outputs, hidden = encoder([x])
        if encoder.is_bidirectional:
            hidden = hidden.permute((1, 0, 2)).reshape(1, -1).unsqueeze(0)
        predicted_symbol = target_w2i[START_SYMBOL]
        for correct in y:
            predictions, hidden = decoder([predicted_symbol], hidden, outputs)
            _, predicted_tensor = predictions.topk(1)
            predicted_symbol = predicted_tensor.detach().item()
            confusion[int(predicted_symbol)][int(correct)] += 1
            predicted_sentence.append(predicted_symbol)
        if predicted_sentence == y:
            correct_sentences += 1
        else:
            incorrect_sentences += 1
    correct_symbols = sum([confusion[i][i] for i in range(len(confusion))])
    all_symbols = torch.tensor(confusion).sum().item()

    # Construct a neat confusion matrix
    for i in range(len(confusion)):
        confusion[i].insert(0, target_i2w[i])
    first_row = ["Predicted/Real"]
    first_row.extend(target_i2w)
    confusion.insert(0, first_row)
    # t = AsciiTable( confusion )

    # print( t.table )
    print("Correctly predicted words    : ", correct_symbols)
    print("Incorrectly predicted words  : ", all_symbols - correct_symbols)
    print("Correctly predicted sentences  : ", correct_sentences)
    print("Incorrectly predicted sentences: ", incorrect_sentences)
    print()


if __name__ == '__main__':

    # ==================== Main program ==================== #
    # Decode the command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-tr', '--train', default='l1_train.txt', help='A training file')
    parser.add_argument('-de', '--dev', default='l1_dev.txt', help='A test file')
    parser.add_argument('-te', '--test', default='l1_test.txt', help='A test file')
    parser.add_argument('-ef', '--embeddings', default='', help='A file with word embeddings')
    parser.add_argument('-et', '--tune-embeddings', action='store_true', help='Fine-tune GloVe embeddings')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-hs', '--hidden_size', type=int, default=25, help='Size of hidden state')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-a', '--attention', action='store_true', help='Use attention weights')
    parser.add_argument('-b', '--bidirectional', action='store_true', help='The encoder is bidirectional')
    parser.add_argument('-g', '--gru', action='store_true', help='Use GRUs instead of ordinary RNNs')
    parser.add_argument('-s', '--save', action='store_true', help='Save model')
    parser.add_argument('-l', '--load', type=str, help="The directory with encoder and decoder models to load")

    args = parser.parse_args()

    is_cuda_available = torch.cuda.is_available()
    print("Is CUDA available? {}".format(is_cuda_available))
    if is_cuda_available:
        print("Current device: {}".format(torch.cuda.get_device_name(0)))
    else:
        print('Running on CPU')
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.load:
        source_w2i = pickle.load(open(os.path.join(args.load, "source_w2i"), 'rb'))
        source_i2w = pickle.load(open(os.path.join(args.load, "source_i2w"), 'rb'))
        target_w2i = pickle.load(open(os.path.join(args.load, "target_w2i"), 'rb'))
        target_i2w = pickle.load(open(os.path.join(args.load, "target_i2w"), 'rb'))

        settings = json.load(open(os.path.join(args.load, "settings.json")))

        use_attention = settings['attention']

        encoder = EncoderRNN(
            len(source_i2w),
            embedding_size=settings['embedding_size'],
            hidden_size=settings['hidden_size'],
            encoder_bidirectional=settings['bidirectional'],
            use_gru=settings['use_gru'],
            tune_embeddings=settings['tune_embeddings'],
            device=device
        )
        decoder = DecoderRNN(
            len(target_i2w),
            embedding_size=settings['embedding_size'],
            hidden_size=settings['hidden_size'] * (settings['bidirectional'] + 1),
            use_attention=use_attention,
            use_gru=settings['use_gru'],
            device=device
        )

        encoder.load_state_dict(torch.load(os.path.join(args.load, "encoder.model"), map_location=device))
        decoder.load_state_dict(torch.load(os.path.join(args.load, "decoder.model"), map_location=device))

        print("Loaded model with the following settings")
        print("-" * 40)
        pprint(settings)
        print()
    else:
        # ==================== Training ==================== #
        # Reproducibility
        # Read a bit more here -- https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(5719)
        np.random.seed(5719)
        torch.manual_seed(5719)
        torch.use_deterministic_algorithms(True)

        if is_cuda_available:
            torch.backends.cudnn.benchmark = False

        use_attention = args.attention

        # Read datasets
        training_dataset = TranslationDataset(args.train)
        dev_dataset = TranslationDataset(args.dev, record_symbols=False)

        print("Number of source words: ", len(source_i2w))
        print("Number of target words: ", len(target_i2w))
        print("Number of training sentences: ", len(training_dataset))
        print()

        # If we have pre-computed word embeddings, then make sure these are used
        if args.embeddings:
            embedding_size, embeddings = load_glove_embeddings(args.embeddings)
        else:
            embedding_size = args.hidden_size
            embeddings = None

        training_loader = DataLoader(training_dataset, batch_size=args.batch_size, collate_fn=PadSequence())
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=PadSequence())

        criterion = nn.CrossEntropyLoss()
        # criterion = nn.NLLLoss()

        encoder = EncoderRNN(
            len(source_i2w),
            embeddings=embeddings,
            embedding_size=embedding_size,
            hidden_size=args.hidden_size,
            encoder_bidirectional=args.bidirectional,
            tune_embeddings=args.tune_embeddings,
            use_gru=args.gru,
            device=device
        )
        decoder = DecoderRNN(
            len(target_i2w),
            embedding_size=embedding_size,
            hidden_size=args.hidden_size * (args.bidirectional + 1),
            use_attention=use_attention,
            use_gru=args.gru,
            device=device
        )

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)

        encoder.train()
        decoder.train()
        print(datetime.now().strftime("%H:%M:%S"), "Starting training.")

        for epoch in range(args.epochs):
            total_loss = 0
            for source, target in training_loader:  # tqdm(training_loader, desc="Epoch {}".format(epoch + 1)):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss = 0
                # hidden is (D * num_layers, B, H)
                outputs, hidden = encoder(source)
                if args.bidirectional:
                    hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=1).unsqueeze(0)

                # The probability of doing teacher forcing will decrease
                # from 1 to 0 over the range of epochs.
                teacher_forcing_ratio = 1  # - epoch/args.epochs

                # The input to the decoder in the first time step will be
                # the boundary symbol, regardless if we are using teacher
                # forcing or not.
                idx = [target_w2i[START_SYMBOL] for sublist in target]
                predicted_symbol = [target_w2i[START_SYMBOL] for sublist in target]

                target_length = len(target[0])
                for i in range(target_length):
                    use_teacher_forcing = (random.random() < teacher_forcing_ratio)
                    if use_teacher_forcing:
                        predictions, hidden = decoder(idx, hidden, outputs)
                    else:
                        # Here we input the previous prediction rather than the
                        # correct symbol.
                        predictions, hidden = decoder(predicted_symbol, hidden, outputs)
                    _, predicted_tensor = predictions.topk(1)
                    predicted_symbol = predicted_tensor.squeeze().tolist()

                    # The targets will be the ith symbol of all the target
                    # strings. They will also be used as inputs for the next
                    # time step if we use teacher forcing.
                    idx = [sublist[i] for sublist in target]
                    loss += criterion(predictions.squeeze(), torch.tensor(idx).to(device))
                loss /= (target_length * args.batch_size)
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                total_loss += loss
            print(datetime.now().strftime("%H:%M:%S"), "Epoch", epoch, "loss:", total_loss.detach().item())
            total_loss = 0

            if epoch % 10 == 0:
                print("Evaluating on the dev data...")
                evaluate(dev_dataset, encoder, decoder)

        # ==================== Save the model  ==================== #

        if (args.save):
            dt = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
            newdir = 'model_' + dt
            os.mkdir(newdir)
            torch.save(encoder.state_dict(), os.path.join(newdir, 'encoder.model'))
            torch.save(decoder.state_dict(), os.path.join(newdir, 'decoder.model'))
            with open(os.path.join(newdir, 'source_w2i'), 'wb') as f:
                pickle.dump(source_w2i, f)
                f.close()
            with open(os.path.join(newdir, 'source_i2w'), 'wb') as f:
                pickle.dump(source_i2w, f)
                f.close()
            with open(os.path.join(newdir, 'target_w2i'), 'wb') as f:
                pickle.dump(target_w2i, f)
                f.close()
            with open(os.path.join(newdir, 'target_i2w'), 'wb') as f:
                pickle.dump(target_i2w, f)
                f.close()

            settings = {
                'training_set': args.train,
                'test_set': args.test,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'hidden_size': args.hidden_size,
                'attention': args.attention,
                'bidirectional': args.bidirectional,
                'embedding_size': embedding_size,
                'use_gru': args.gru,
                'tune_embeddings': args.tune_embeddings
            }
            with open(os.path.join(newdir, 'settings.json'), 'w') as f:
                json.dump(settings, f)

    # ==================== Evaluation ==================== #

    encoder.eval()
    decoder.eval()
    print("Evaluating on the test data...")

    test_dataset = TranslationDataset(args.test, record_symbols=False)
    print("Number of test sentences: ", len(test_dataset))
    print()

    evaluate(test_dataset, encoder, decoder)

    # ==================== User interaction ==================== #

    decoder.display_attention = True
    while (True):
        text = input("> ")
        if text == "":
            continue
        try:
            source_sentence = [source_w2i[w] for w in nltk.word_tokenize(text)]
        except KeyError:
            print("Erroneous input string")
            continue
        outputs, hidden = encoder([source_sentence])
        if encoder.is_bidirectional:
            hidden = hidden.permute((1, 0, 2)).reshape(1, -1).unsqueeze(0)

        predicted_symbol = target_w2i[START_SYMBOL]
        target_sentence = []
        attention_probs = []
        num_attempts = 0
        while num_attempts < MAX_PREDICTIONS:
            if use_attention:
                predictions, hidden, alpha = decoder([predicted_symbol], hidden, outputs)
                attention_probs.append(alpha.permute(0, 2, 1).squeeze().detach().tolist())
            else:
                predictions, hidden = decoder([predicted_symbol], hidden, outputs)

            _, predicted_tensor = predictions.topk(1)
            predicted_symbol = predicted_tensor.detach().item()
            target_sentence.append(predicted_symbol)

            num_attempts += 1

            if predicted_symbol == target_w2i[END_SYMBOL]:
                break

        for i in target_sentence:
            print(target_i2w[i].encode('utf-8').decode(), end=' ')
        print()

        if use_attention:
            # Construct the attention table
            ap = torch.tensor(attention_probs).T
            if len(ap.shape) == 1:
                ap = ap.unsqueeze(0)
            attention_probs = ap.tolist()

            for i in range(len(attention_probs)):
                for j in range(len(attention_probs[i])):
                    attention_probs[i][j] = "{val:.2f}".format(val=attention_probs[i][j])
            for i in range(len(attention_probs)):
                if i < len(text):
                    attention_probs[i].insert(0, source_i2w[source_sentence[i]])
                else:
                    attention_probs[i].insert(0, ' ')
            first_row = ["Source/Result"]
            for w in target_sentence:
                first_row.append(target_i2w[w])
            attention_probs.insert(0, first_row)
            t = AsciiTable(attention_probs)
            print(t.table)
