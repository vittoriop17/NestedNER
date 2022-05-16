from torch.utils.data import DataLoader
from dataset import *
from utils import *


def load_and_eval_hugging_face_model(args):
    """

    :param args: set of CLI arguments.
    It requires the following mandatory arguments:
    1. hf_model_path: str: path to the folder containing the fine-tuned model info
    2. test_set: str: path to the test set used for evaluation
    ...
    """


def main(args):
    train_ds = WikiHowDataset(args.train_set)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    for epoch_idx in range(args.epochs):
        for batch_idx, (batch_articles, batch_summaries) in enumerate(train_dl):
            breakpoint()


if __name__=='__main__':
    args = read_args()
    main(args)

