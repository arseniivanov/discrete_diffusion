import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
import math

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_data, tokenize_fun=lowercase_tokenizer, max_voc_size: int | None = None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    out = []
    for example in train_data:
        out.extend(tokenize_fun(example['text']))

    c = Counter(out)
    
    if max_voc_size is not None:
        words = c.most_common(max_voc_size - 4) 
    else:
        words = list(c.items())

    words[:0] = [(k, 1) for k in [pad_token, unk_token, bos_token, eos_token]] #prepend

    str_to_int = {}
    int_to_str = {}
    for c, entry in enumerate(words):
        str_to_int[entry[0]] = c
        int_to_str[c] = entry[0]

    return A1Tokenizer(str_to_int, int_to_str, model_max_length, unk_token, bos_token, eos_token, pad_token)

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, str_to_int, int_to_str, model_max_length, unk_token, bos_token, eos_token, pad_token):
        self.pad_token_id = str_to_int[pad_token]
        self.model_max_length = model_max_length
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token 
        self.pad_token = pad_token

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')

        transformed = []
        running_max = 0
        for text in texts:
            tokenized = lowercase_tokenizer(text)
            int_toks = []
            for tok in tokenized:
                int_tok = self.str_to_int.get(tok, self.str_to_int[self.unk_token])
                int_toks.append(int_tok)
            int_toks[:0] = [self.str_to_int[self.bos_token]]
            int_toks.append(self.str_to_int[self.eos_token])

            if truncation:
                int_toks = int_toks[:self.model_max_length]
            transformed.append(int_toks)
            if len(int_toks) > running_max:
                running_max = len(int_toks)

        if padding:
            for sent in transformed:
                if len(sent) < running_max:
                    sent.extend([self.str_to_int[self.pad_token]]*(running_max - len(sent)))

        if return_tensors:
            transformed = torch.tensor(transformed)

        return BatchEncoding({'input_ids': transformed})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=10000, embedding_size=256, hidden_size=512, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.GRU(config.embedding_size, config.hidden_size, batch_first=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None):
        """The forward pass of the RNN-based language model.
        
           Args:
             - input_ids:  The input tensor (2D), consisting of a batch of integer-encoded texts.
             - labels:     The reference tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             A CausalLMOutput containing
               - logits:   The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
               - loss:     The loss computed on this batch.               
        """
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        logits = self.unembedding(rnn_out)
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = self.loss_func(shift_logits, shift_labels)
        else:
            loss = None

        return CausalLMOutput(logits=logits, loss=loss)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), args.learning_rate)
        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size)
        
        for i in range(args.num_train_epochs):
            tot_loss = 0
            for batch in train_loader:
                texts = batch['text']
                tokenized_batch = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
                input_ids = tokenized_batch['input_ids']
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                out = self.model(input_ids, labels=labels)
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()

            print(f"Training loss after epoch {i}: {tot_loss}")

            val_loss = 0
            num_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    texts = batch['text']
                    tokenized_batch = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
                    input_ids = tokenized_batch['input_ids']
                    labels = input_ids.clone()
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    out = self.model(input_ids, labels=labels)
                    loss = out.loss
                    val_loss += loss.item()
                    num_batches += 1

            mean_val_loss = val_loss / num_batches
            
            # 5. Exponentiate the mean loss to get perplexity
            perplexity = math.exp(mean_val_loss)
            print(f"Perplexity on validation set: {perplexity}")
            
            self.model.train()

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

def nearest_neighbors(emb, voc, inv_voc, word, n_neighbors=5):
    test_emb = emb.weight[voc[word]]
    
    sim_func = nn.CosineSimilarity(dim=1)
    cosine_scores = sim_func(test_emb, emb.weight)
    
    near_nbr = cosine_scores.topk(n_neighbors+1)
    topk_cos = near_nbr.values[1:]
    topk_indices = near_nbr.indices[1:]
    
    return [ (inv_voc[ix.item()], cos.item()) for ix, cos in zip(topk_indices, topk_cos) ]

if __name__ == "__main__":
    nltk.download('punkt_tab')
    TRAIN_FILE = 'train.txt'
    VAL_FILE = 'val.txt'

    from datasets import load_dataset
    dataset = load_dataset('text', data_files={'train': TRAIN_FILE, 'val': VAL_FILE})
    dataset = dataset.filter(lambda x: x['text'].strip() != '')

    print("Building tokenizer...")
    if os.path.exists('tok'):
        tokenizer = A1Tokenizer.from_file('tok')
    else:
        tokenizer = build_tokenizer(dataset['train'], max_voc_size=2**14,model_max_length=10)
        tokenizer.save('tok')

    # 3. Run the sanity check
    test_texts = ['This is a test.', 'Another test.']
    print("\nTokenizing texts:")
    print(test_texts)
    
    output = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True)
    
    print("\nSanity Check Output:")
    print(output)

    if os.path.exists('trainer_output'):
        #eval path
        model = A1RNNModel.from_pretrained('trainer_output')

        if True:
            print(nearest_neighbors(model.embedding, tokenizer.str_to_int, tokenizer.int_to_str, "sweden"))
            exit()

        test_sentence = ["She lives in San"]
        toks = tokenizer(test_sentence, padding=True, truncation=True, return_tensors='pt')
        input_ids = toks['input_ids']
        out = model(input_ids)
        next_word_logits = out.logits[0, -2, :] # last token is EOS, fetch second to last, [batch, tokens, dict_size]
        best_token_id = torch.argmax(next_word_logits).item()
        predicted_word = tokenizer.int_to_str[best_token_id]
        print(f"Input: {test_sentence[0]}")
        print(f"Predicted next word: {predicted_word}")

    else:
        conf = A1RNNModelConfig(vocab_size=len(tokenizer), embedding_size=256, hidden_size= 2**8)
        model = A1RNNModel(conf)

        training_args = TrainingArguments(
            output_dir='trainer_output',
            optim='adamw_torch',
            learning_rate=1e-3,
            num_train_epochs=3,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            logging_strategy='steps',
            logging_steps=100,
            save_strategy='epoch',
        )

        trainer = A1Trainer(model, training_args, dataset['train'], dataset['val'], tokenizer)
        trainer.train()
