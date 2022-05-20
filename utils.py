from lstm_model import *
from prettytable import PrettyTable
import dataset
from transformers import SchedulerType, MODEL_MAPPING


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def read_args():
    # Decode the command-line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--debug', default=False, action='store_true', help="Debug mode")
    parser.add_argument('-mnop', '--model_name_or_path', required=False, help="Not needed")
    parser.add_argument('-n', '--neptune', default=False, action='store_true', help="Log loss metrics on Neptune.AI. Need to set the env variable NEPTUNE_API_TOKEN first")
    parser.add_argument('-uaw', '--use_attention_window', default=False, action='store_true', help="Boolean flag: it specifies if to use Windowed-Attention mechanism or Full-attention")
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


def read_bart_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
             "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
             "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
             "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
             "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
             "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
             "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
             "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    args = parser.parse_args()
    return args


def save_model(encoder, decoder, args, epoch):
    dt = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_')
    newdir = f'model_epoch{epoch}_' + dt
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
        'embedding_size': args.embedding_size,
        'use_gru': args.gru,
        'tune_embeddings': args.tune_embeddings
    }
    with open(os.path.join(newdir, 'settings.json'), 'w') as f:
        json.dump(settings, f)


def load_lstm_model(args):
    source_w2i = pickle.load(open(os.path.join(args.load, "source_w2i"), 'rb'))
    source_i2w = pickle.load(open(os.path.join(args.load, "source_i2w"), 'rb'))
    target_w2i = pickle.load(open(os.path.join(args.load, "target_w2i"), 'rb'))
    target_i2w = pickle.load(open(os.path.join(args.load, "target_i2w"), 'rb'))
    dataset.source_w2i, dataset.i2w, dataset.target_w2i, dataset.target_i2w = source_w2i, source_i2w, target_w2i, target_i2w
    settings = json.load(open(os.path.join(args.load, "settings.json")))

    use_attention = settings['attention']

    encoder = EncoderLSTM(
        len(source_i2w),
        embedding_size=settings['embedding_size'],
        hidden_size=settings['hidden_size'],
        encoder_bidirectional=settings['bidirectional'],
        tune_embeddings=settings['tune_embeddings'],
        device=args.device
    )
    decoder = DecoderLSTM(
        len(target_i2w),
        embedding_size=settings['embedding_size'],
        hidden_size=settings['hidden_size'] * (settings['bidirectional'] + 1),
        use_attention=use_attention,
        device=args.device
    )

    encoder.load_state_dict(torch.load(os.path.join(args.load, "encoder.model"), map_location=args.device))
    decoder.load_state_dict(torch.load(os.path.join(args.load, "decoder.model"), map_location=args.device))

    print("Loaded model with the following settings")
    print("-" * 40)
    pprint(settings)
    print()
    return encoder, decoder