import argparse
import json


def read_args():
    # Decode the command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help="Debug mode")
    parser.add_argument('-tr_s', '--train_set', type=str,
                        help='Path to the root folder containing all the train samples')
    parser.add_argument('-te_s', '--test_set', type=str, help='Path to the root folder containing all the test samples')
    # Notice: model_type is always required
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-hs', '--hidden_size', type=int, default=25, help='Size of hidden state')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--bidirectional', action='store_true', help='The encoder is bidirectional')
    parser.add_argument('-s', '--save', action='store_true', help='Save model')
    parser.add_argument('-l', '--load', type=str, help="The directory with pre-trained model to load")
    parser.add_argument('-dr', '--dropout', type=float, default='0.1', help='Dropout')

    args = parser.parse_args()

    print("\033[1;32mArguments:\n\033[0m")
    print(f"\033[1;32m{json.dumps(args.__dict__, indent=4)}\033[0m")
    return args
