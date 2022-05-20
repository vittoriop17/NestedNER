# WikiHowSummarization
Course project about Text Summarization.
## Setup
### Requirements
Open the CMD under the root directory of the project, then run:
    
    pip install -r requirements.txt
### Additional files
The files used for training/validation/test can be found under the folder **process-dataset**.
Nevertheless, you first need to download the GloVe embedding file. In particular, we used glove.6B.50d.txt that can be downloaded from [GloVe 6B](https://www.kaggle.com/datasets/anindya2906/glove6b).
Once you downloaded it, add the file under the root directory of the project.
## Models
### LSTM-based models
To train and evaluate the LSTM-based models, use the main.py module.
Once again, open the CMD under the root directory of the project, then run:

    python main.py --max_source_len 1024 --train "process-dataset\train__text_summary.csv" --dev "process-dataset\val__text_summary.csv" --test "process-dataset\test__text_summary.csv" --embeddings glove.6B.50d.txt --tune-embeddings --attention --bidirectional --save 

This is the configuration we used for the best LSTM-based model. Moreover, we only provide direct access to the Windowed-Attention models.

### BART-based models
We leveraged an existing code from [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) in order to use the pre-trained model offered by HuggingFace.
A detailed explanation of the code can be found inside the README.md file under the folder **pytorch-api**.
From now on, you will need to open the CMD under the folder **pytorch-api**.
1. To run the fine-tuning (3 epochs only) of the pre-trained bart-base model, run:


        python run_summarization_no_trainer.py --model_name_or_path facebook/bart-base --train_file "..\train__text_summary.csv" --validation_file "..\test__text_summary.csv" --output_dir ".\tst-summarization" --num_train_epochs 3

2. To run the evaluation of the pre-trained bart-base model **without fine-tuning**, run:
    
        
        python run_summarization_no_trainer.py --model_name_or_path facebook/bart-base --train_file "..\train__text_summary.csv" --validation_file "..\test__text_summary.csv" --output_dir ~/tmp/tst-summarization --num_train_epochs 0

3. To run the evaluation of the pre-trained bart-base model **with fine-tuning**, run:

   
        python run_summarization_no_trainer.py --model_name_or_path ".\tst-summarization" --train_file "..\train__text_summary.csv" --validation_file "..\test__text_summary.csv" --output_dir ~/tmp/tst-summarization --num_train_epochs 0

In order to run the third command, it is necessary to run the first one, otherwise there will be no folder called **tst-summarization**.
