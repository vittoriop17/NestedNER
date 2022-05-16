from dataset import *

# ==================== Encoder ==================== #

class EncoderLSTM(nn.Module):
    """
    Encodes a batch of source sentences.
    """

    def __init__(self, no_of_input_symbols, embeddings=None, embedding_size=16, hidden_size=25,
                 encoder_bidirectional=False, device='cpu', tune_embeddings=False, **kwargs):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.is_bidirectional = encoder_bidirectional
        self.embedding = nn.Embedding(no_of_input_symbols, embedding_size)
        if embeddings != None:
            self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float),
                                                 requires_grad=tune_embeddings)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=self.is_bidirectional)
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
        output, (h, c) = self.lstm(word_embeddings)
        return output, h, c


# ==================== Decoder ==================== #

class DecoderLSTM(nn.Module):

    def __init__(self, no_of_output_symbols, embedding_size=16, hidden_size=25, use_attention=True,
                 display_attention=False, device='cpu'):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(no_of_output_symbols, embedding_size)
        self.no_of_output_symbols = no_of_output_symbols
        self.W = nn.Parameter(torch.rand(hidden_size, hidden_size) - 0.5)
        self.U = nn.Parameter(torch.rand(hidden_size, hidden_size) - 0.5)
        self.v = nn.Parameter(torch.rand(hidden_size, 1) - 0.5)
        self.use_attention = use_attention
        self.display_attention = display_attention
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, no_of_output_symbols)
        self.device = device
        self.to(device)

    def forward(self, inp, hidden_state, cell_state, encoder_outputs):
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
            context, alpha = self._compute_context(encoder_outputs, hidden_state)
            context = context.view(1, B, -1)
            lstm_output, (h, c) = self.lstm(word_embeddings, (context, cell_state))
        else:
            lstm_output, (h, c) = self.lstm(word_embeddings, (hidden_state, cell_state))
        if self.use_attention and self.display_attention:
            return self.output(lstm_output), h, c, alpha
        else:
            return self.output(lstm_output), h, c

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
def evaluate(ds, encoder, decoder, args, test_or_val='val', neptune_run=None):
    confusion = [[0 for a in target_i2w] for b in target_i2w]
    correct_sentences, incorrect_sentences = 0, 0
    metric = load_metric("rouge")
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    for x, y in ds:
        loss = 0
        predicted_sentence = []
        outputs, hidden, cell = encoder([x])
        if encoder.is_bidirectional:
            hidden = hidden.permute((1, 0, 2)).reshape(1, -1).unsqueeze(0)
            cell = cell.permute((1, 0, 2)).reshape(1, -1).unsqueeze(0)
        predicted_symbol = target_w2i[START_SYMBOL]
        for correct in y:
            predictions, hidden, cell = decoder([predicted_symbol], hidden, cell, outputs)
            _, predicted_tensor = predictions.topk(1)
            predicted_symbol = predicted_tensor.detach().item()
            confusion[int(predicted_symbol)][int(correct)] += 1
            predicted_sentence.append(predicted_symbol)
            loss += criterion(predictions.squeeze(), torch.tensor(correct).to(args.device))
        loss /= (len(y))
        total_loss += loss
        if predicted_sentence == y:
            correct_sentences += 1
        else:
            incorrect_sentences += 1
        metric.add(
            predictions=list(filter(lambda c: c not in SPECIAL_CHAR_IDX, predicted_sentence)),
            references=list(filter(lambda c: c not in SPECIAL_CHAR_IDX, y)),
        )
    app_loss = total_loss.detach().item()
    neptune_run[f'{test_or_val}/epoch_loss'].log(app_loss)
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

    print("Rouge metrics:\n")
    result = metric.compute(use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"\033[1;32m{json.dumps(result, indent=4)}\033[0m")
    print()
