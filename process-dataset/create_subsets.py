import numpy as np
import pandas as pd


MAX_TRAIN_SIZE = 4096
MAX_VAL_SIZE = 1024
MAX_TEST_SIZE = 1024


def main():
    train_titles_file_in, train_titles_file_out = "all_train.txt", "subset_train.txt"
    val_titles_file_in, val_titles_file_out = "all_val.txt", "subset_val.txt"
    test_titles_file_in, test_titles_file_out = "all_test.txt", "subset_test.txt"
    np.random.seed(42)
    train_titles_np = np.loadtxt(train_titles_file_in, dtype=str)
    val_titles_np = np.loadtxt(val_titles_file_in, dtype=str)
    test_titles_np = np.loadtxt(test_titles_file_in, dtype=str)
    train_len = min(MAX_TRAIN_SIZE, len(train_titles_np))
    val_len = min(MAX_VAL_SIZE, len(val_titles_np))
    test_len = min(MAX_TEST_SIZE, len(test_titles_np))
    train_indices = np.random.choice(range(len(train_titles_np)), size=train_len, replace=False)
    val_indices = np.random.choice(range(len(val_titles_np)), size=val_len, replace=False)
    test_indices = np.random.choice(range(len(test_titles_np)), size=test_len, replace=False)
    np.savetxt(train_titles_file_out, train_titles_np[train_indices],  fmt='%s')
    np.savetxt(val_titles_file_out, val_titles_np[val_indices],  fmt='%s')
    np.savetxt(test_titles_file_out, test_titles_np[test_indices],  fmt='%s')


if __name__ == '__main__':
    main()
