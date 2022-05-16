from utils import *
import neptune.new as neptune
from lstm_model import *
import tqdm


def load_and_eval_hugging_face_model(args):
    """

    :param args: set of CLI arguments.
    It requires the following mandatory arguments:
    1. hf_model_path: str: path to the folder containing the fine-tuned model info
    2. test_set: str: path to the test set used for evaluation
    ...
    """


def main(args):


    train_ds = SummarizationDataset(args.train_set, args.max_source_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    for epoch_idx in range(args.epochs):
        for batch_idx, (batch_articles, batch_summaries) in enumerate(train_dl):
            breakpoint()



if __name__=='__main__':
    # ==================== Main program ==================== #
    # Decode the command-line arguments
    args = read_args()
    # initialize NEPTUNE client
    run = neptune.init(
        project="vittoriop.17/WikiHowSummarization",
        api_token=os.getenv("NEPTUNE_API_TOKEN"))  # your credentials
    run['parameters'] = args.__dict__

    is_cuda_available = torch.cuda.is_available()
    print("Is CUDA available? {}".format(is_cuda_available))
    if is_cuda_available:
        print("Current device: {}".format(torch.cuda.get_device_name(0)))
    else:
        print('Running on CPU')
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    if args.load:
        source_w2i = pickle.load(open(os.path.join(args.load, "source_w2i"), 'rb'))
        source_i2w = pickle.load(open(os.path.join(args.load, "source_i2w"), 'rb'))
        target_w2i = pickle.load(open(os.path.join(args.load, "target_w2i"), 'rb'))
        target_i2w = pickle.load(open(os.path.join(args.load, "target_i2w"), 'rb'))

        settings = json.load(open(os.path.join(args.load, "settings.json")))

        use_attention = settings['attention']

        encoder = EncoderLSTM(
            len(source_i2w),
            embedding_size=settings['embedding_size'],
            hidden_size=settings['hidden_size'],
            encoder_bidirectional=settings['bidirectional'],
            tune_embeddings=settings['tune_embeddings'],
            device=device
        )
        decoder = DecoderLSTM(
            len(target_i2w),
            embedding_size=settings['embedding_size'],
            hidden_size=settings['hidden_size'] * (settings['bidirectional'] + 1),
            use_attention=use_attention,
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
        training_dataset = SummarizationDataset(args.train)
        dev_dataset = SummarizationDataset(args.dev, record_symbols=False)

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

        training_loader = DataLoader(training_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=PadSequence())
        dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=PadSequence())

        criterion = nn.CrossEntropyLoss()
        # criterion = nn.NLLLoss()

        encoder = EncoderLSTM(
            len(source_i2w),
            embeddings=embeddings,
            embedding_size=embedding_size,
            hidden_size=args.hidden_size,
            encoder_bidirectional=args.bidirectional,
            tune_embeddings=args.tune_embeddings,
            device=device
        )
        decoder = DecoderLSTM(
            len(target_i2w),
            embedding_size=embedding_size,
            hidden_size=args.hidden_size * (args.bidirectional + 1),
            use_attention=use_attention,
            device=device
        )

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)

        encoder.train()
        decoder.train()

        print(datetime.now().strftime("%H:%M:%S"), "Starting training.")

        for epoch in range(args.epochs):
            total_loss = 0
            for source, target in tqdm.tqdm(training_loader, desc="Epoch {}".format(epoch + 1)):
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss = 0
                # hidden is (D * num_layers, B, H)
                outputs, hidden, cell = encoder(source)
                if args.bidirectional:
                    hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=1).unsqueeze(0)
                    cell = torch.cat([cell[0, :, :], cell[1, :, :]], dim=1).unsqueeze(0)

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
                        predictions, hidden, cell = decoder(inp=idx, hidden_state=hidden, cell_state=cell, encoder_outputs=outputs)
                    else:
                        # Here we input the previous prediction rather than the
                        # correct symbol.
                        predictions, hidden, cell = decoder(inp=predicted_symbol, hidden_state=hidden, cell_state=cell, encoder_outputs=outputs)
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
                run['train/batch_loss'].log(loss.detach().item())
            app_loss = total_loss.detach().item()
            print(datetime.now().strftime("%H:%M:%S"), "Epoch", epoch, "loss:", app_loss)
            run['train/epoch_loss'].log(app_loss)
            total_loss = 0
            # print("Evaluating on the dev data...")
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

    test_dataset = SummarizationDataset(args.test, record_symbols=False)
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
        outputs, hidden, cell = encoder([source_sentence])
        if encoder.is_bidirectional:
            hidden = hidden.permute((1, 0, 2)).reshape(1, -1).unsqueeze(0)
            cell = cell.permute((1, 0, 2)).reshape(1, -1).unsqueeze(0)

        predicted_symbol = target_w2i[START_SYMBOL]
        target_sentence = []
        attention_probs = []
        num_attempts = 0
        while num_attempts < MAX_PREDICTIONS:
            if use_attention:
                predictions, hidden, cell, alpha = decoder([predicted_symbol], hidden, cell, outputs)
                attention_probs.append(alpha.permute(0, 2, 1).squeeze().detach().tolist())
            else:
                predictions, hidden, cell = decoder([predicted_symbol], hidden, cell, outputs)

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
    run.stop()

