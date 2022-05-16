from lstm_model import *


def read_args():
    # Decode the command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help="Debug mode")
    # Notice: model_type is always required
    parser.add_argument('-dr', '--dropout', type=float, default='0.1', help='Dropout')
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
    parser.add_argument('-msl', '--max_source_len', type=int, default=1024, help="The maximum number of tokens in the input sequence."
                                                                                 "All the source sequences exceeding this value,"
                                                                                 "will be cut in order to fit such value")
    args = parser.parse_args()

    print("\033[1;32mArguments:\n\033[0m")
    print(f"\033[1;32m{json.dumps(args.__dict__, indent=4)}\033[0m")
    return args
