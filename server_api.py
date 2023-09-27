from flask import Flask, jsonify, request
from utils import *
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def load_bart_model(args):
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        print("You are instantiating a new config instance from scratch.")
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        print("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""
    return model, tokenizer, config


args_lstm = read_args()
args_lstm.load = "D:\\UNIVERSITA\\KTH\\Semestre 2\\LanguageEngineering\\models\\w_attn_window\\max_source_len_1536__epoch10"
args_bart_pretrained = read_bart_args()
bart_model_pretrained, bart_tokenizer_pretrained, config_pretrained = load_bart_model(args_bart_pretrained)
args_bart_finetuned = args_bart_pretrained
args_bart_finetuned.model_name_or_path = "D:\\UNIVERSITA\\KTH\\Semestre 2\\LanguageEngineering\\models\\BartFineTuned"
bart_model_finetuned, bart_tokenizer_finetuned, config_finetuned = load_bart_model(args_bart_finetuned)
args_lstm.device = 'cuda' if torch.cuda.is_available() else 'cpu'
app = Flask(__name__)
lstm_encoder, lstm_decoder = load_lstm_model(args_lstm)


@app.route('/model_lstm/', methods=['GET'])
def get_lstm_predictions():
    text = request.values['sentence']
    try:
        source_sentence = [dataset.source_w2i[w] if w in dataset.source_w2i.keys() else dataset.source_w2i['<unk>'] for w in nltk.word_tokenize(text)]
    except KeyError:
        print("Erroneous input string")
        return jsonify({"prediction": "Error occured"})
    outputs, hidden, cell = lstm_encoder([source_sentence])
    if lstm_encoder.is_bidirectional:
        hidden = hidden.permute((1, 0, 2)).reshape(1, -1).unsqueeze(0)
        cell = cell.permute((1, 0, 2)).reshape(1, -1).unsqueeze(0)

    predicted_symbol = dataset.target_w2i[START_SYMBOL]
    target_sentence = []
    attention_probs = []
    num_attempts = 0
    while num_attempts < MAX_PREDICTIONS:
        predictions, hidden, cell = lstm_decoder([predicted_symbol], hidden, cell, outputs, idx=num_attempts)
        _, predicted_tensor = predictions.topk(1)
        predicted_symbol = predicted_tensor.detach().item()
        target_sentence.append(predicted_symbol)

        num_attempts += 1

        if predicted_symbol == dataset.target_w2i[END_SYMBOL]:
            break
    pred_summary = ''
    for i in target_sentence:
        pred_summary += dataset.target_i2w[i].encode('utf-8').decode() + ' '
    return jsonify({'prediction': pred_summary})


@app.route('/model_pretrained/', methods=['GET'])
def get_bart_pretrained_predictions():
    gen_kwargs = {
        "max_length": args_bart_pretrained.val_max_target_length if args_bart_pretrained is not None else config_pretrained.max_length,
        "num_beams": args_bart_pretrained.num_beams,
    }
    sentence = request.values['sentence']
    batch = bart_tokenizer_pretrained(sentence, return_tensors="pt", max_length=args_bart_pretrained.max_source_length)
    generated_ids = bart_model_pretrained.generate(batch["input_ids"], **gen_kwargs)
    pred_summary = bart_tokenizer_pretrained.batch_decode(generated_ids, skip_special_tokens=True)
    return jsonify({'prediction': pred_summary})


@app.route('/model_finetuned/', methods=['GET'])
def get_bart_finetuned_predictions():
    gen_kwargs = {
        "max_length": args_bart_finetuned.max_target_length if args_bart_finetuned is not None else config_finetuned.max_length,
        "num_beams": args_bart_finetuned.num_beams,
    }
    sentence = request.values['sentence']
    batch = bart_tokenizer_finetuned(sentence, return_tensors="pt", max_length=args_bart_finetuned.max_source_length)
    generated_ids = bart_model_finetuned.generate(batch["input_ids"], **gen_kwargs)
    pred_summary = bart_tokenizer_finetuned.batch_decode(generated_ids, skip_special_tokens=True)
    return jsonify({'prediction': pred_summary})


if __name__=='__main__':
    app.run()