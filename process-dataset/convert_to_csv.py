import numpy as np
import os


def read_text_and_summary(filepath):
    with open(filepath, "r", encoding='utf-8') as fin:
        summary = ""
        text = ""
        append_to_summary = False
        for line in fin.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("@summary"):
                append_to_summary = True
            elif line.startswith("@article"):
                append_to_summary = False
            elif append_to_summary:
                summary += line.replace('"', "'") + " "
            else:
                text += line.replace('"', "'") + "  "
    return text, summary


def main():
    articles_path = "articles"
    subset_files = ["subset_train.txt",
                    "subset_val.txt",
                    "subset_test.txt"]
    final_files = ["train__text_summary.csv",
                   "val__text_summary.csv",
                   "test__text_summary.csv"]
    all_filenames = [np.loadtxt(subset_file, dtype=str, encoding='utf-8') for subset_file in subset_files]
    for i in range(len(all_filenames)):
        with open(final_files[i], "w", encoding='utf-8') as fout:
            fout.write("text,summary\n")
            for filename in all_filenames[i]:
                filepath = os.path.join(articles_path, filename+".txt")
                text, summary = read_text_and_summary(filepath)
                fout.write(f'"{text}","{summary}"\n')


if __name__ == '__main__':
    main()
