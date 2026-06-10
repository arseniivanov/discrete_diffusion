from numpy import full
import torch
import torchvision
from torch import nn
from transformers import TrainingArguments
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
import os

from A1_skeleton import A1Tokenizer, build_tokenizer, A1Trainer, nearest_neighbors

class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, 
                 vocab_size=50304,
                 hidden_size=256,
                 intermediate_size=1024,
                 num_attention_heads=8,
                 num_hidden_layers=4,
                 rope_theta=10000.0,
                 hidden_act='silu',
                 max_position_embeddings=2048,
                 rms_norm_eps=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_states):
        gate = self.gate(hidden_states)
        trans = self.act(self.down_proj(hidden_states))
        return self.up_proj(gate*trans)

class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)

    def forward(self, hidden_states):
        return self.norm(hidden_states)


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.W_o = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.qnorm = A2RMSNorm(config)
        self.knorm = A2RMSNorm(config)
        self.heads = config.num_attention_heads

    def forward(self, hidden_states, rope_rotations):
        q = self.W_q(hidden_states)
        k = self.W_k(hidden_states)
        v = self.W_v(hidden_states)

        b, seq, h_dim = hidden_states.shape

        q = self.qnorm(q)
        k = self.knorm(k)
        q = q.view(b, seq, self.heads, h_dim//self.heads).transpose(1, 2)
        k = k.view(b, seq, self.heads, h_dim//self.heads).transpose(1, 2)
        v = v.view(b, seq, self.heads, h_dim//self.heads).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, rope_rotations)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        attn_out = attn_out.transpose(1, 2).reshape(b, seq, h_dim)
        output = self.W_o(attn_out)
        return output


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        self.attention = A2Attention(config)
        self.mlp = A2MLP(config)
        self.post_attn_norm = A2RMSNorm(config)
        self.post_swiglu_norm = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        out1 = hidden_states + self.post_attn_norm(self.attention(hidden_states, rope_rotations))
        return out1 + self.post_swiglu_norm(self.mlp(out1))

class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = A2RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [A2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.post_init()

    def forward(self, input_ids, labels=None):
        hidden_states = self.embedding(input_ids)
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope_rotations)

        norm = self.norm(hidden_states)
        logits = self.unembedding(norm)

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = self.loss_func(shift_logits, shift_labels)
        else:
            loss = None

        return CausalLMOutput(logits=logits, loss=loss)

#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin

def generate_causal(model, prompt, max_length, temperature=1.0, topk=5):
    full_text = prompt
    for _ in range(max_length):
        out = model(full_text)
        next_token_logits = out.logits[0, -1, :]
        next_token_logits /= temperature

        topk_values, topk_indices = torch.topk(next_token_logits, k=topk)
        example_distr = torch.distributions.Categorical(logits=topk_values)
        sampled = example_distr.sample()
        sampled_token = topk_indices[sampled]
        sampled_token = sampled_token.view(1, 1)
        full_text = torch.cat([full_text, sampled_token], dim=1)
        if sampled_token == 3: #eos token
            break
    return full_text[0].tolist()

if __name__ == "__main__":
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

    if os.path.exists('trainer_output'):
        #eval path
        model = A2Transformer.from_pretrained('trainer_output')

        if False:
            print(nearest_neighbors(model.embedding, tokenizer.str_to_int, tokenizer.int_to_str, "sweden"))
            exit()

        if True:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_name = 'allenai/OLMo-2-0425-1B'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            test_sentence = ["In natural language, a transformer is"]
            toks = tokenizer(test_sentence, padding=True, truncation=True, return_tensors='pt')
            input_ids = toks['input_ids']
            
            out = generate_causal(
                model, 
                input_ids, 
                max_length=30, 
                temperature=1.0, 
                topk=5,
            )
            
            # The HuggingFace way to convert a list of IDs back to a string:
            output = tokenizer.decode(out, skip_special_tokens=True)

            print(f"Input: {test_sentence[0]}")
            print(f"Predicted answer: {output}")
            exit()
        if True:
            test_sentence = ["In natural language, a transformer is"]
            toks = tokenizer(test_sentence, padding=True, truncation=True, return_tensors='pt')
            input_ids = toks['input_ids']
            out = generate_causal(model, input_ids, max_length=30, temperature=1.0, topk=5)
            output = ""
            for tok in out:
                output += tokenizer.int_to_str[tok]

            print(f"Input: {test_sentence[0]}")
            print(f"Predicted answer: {output}")
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
        exit()


    model = A2Transformer(A2ModelConfig())

    training_args = TrainingArguments(
        output_dir='trainer_output',
        optim='adamw_torch',          # Required by assignment assertion
        learning_rate=1e-3,           # Good starting point for RNNs
        num_train_epochs=3,           # Enough to see the loss drop
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        logging_strategy='steps',     # Nice to have: see loss updates during the epoch
        logging_steps=100,
        save_strategy='epoch',         # Optional: save the model after each epoch
    )
    trainer = A1Trainer(model, training_args, dataset['train'], dataset['val'], tokenizer)
    trainer.train()
