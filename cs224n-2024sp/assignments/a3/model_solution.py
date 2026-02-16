"""
A bare-bones GPT-2 style transformer.
"""

import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int
from torch.nn.functional import softmax
from dataclasses import dataclass
from einops import rearrange
from transformers import GPT2LMHeadModel
import huggingface_hub

from utils import state_dict_converter


# TODO: Add in attention mask to the entire assignment
# TODO: Maybe add KV caching


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    context_length: int
    vocab_size: int


class CausalAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Using attention dim from attention is all you need
        assert config.d_model % config.n_heads == 0
        self.d_attention = int(config.d_model / config.n_heads)

        #self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)

        self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)

        self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False
        )

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO, complete
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        B,S,D = Q.shape
        d_head = self.d_attention
        H = D // d_head
        Q = Q.view(B, S, H, d_head).transpose(1, 2)
        K = K.view(B, S, H, d_head).transpose(1, 2)
        V = V.view(B, S, H, d_head).transpose(1, 2) # [B,H,S,d_head]
        
        d_k = self.d_attention
        scores = Q @ K.transpose(-2,-1)/ math.sqrt(d_k) # [B,H,S,S]
        
        mask =self.causal_mask[:,:,:S,:S]
        scores = scores.masked_fill(mask==0,float('-inf'))
        
        attn = torch.softmax(scores,dim=-1) # [B,H,S,S]
        out =attn @ V # [B,H,S,d_head]
        out = out.transpose(1,2).contiguous().view(B,S,D)
        
        out = self.W_o(out)
        return out



class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip

class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = GELU()

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        # TODO, complete
        y = self.fc1(x)
        y = self.gelu(y)
        z = self.fc2(y)
        return z
        

class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config)
        self.attention = CausalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO complete
        x = x + self.attention(self.pre_layer_norm(x))
        x = x + self.mlp(self.post_layer_norm(x))
        return x


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.backbone = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:

        # TODO, complete
        B,S = x.shape
        device = x.device
        
        tok = self.embeddings(x) # [B,S,d_model]
        pos_ids = torch.arange(S,device=device)
        pos = self.position_embeddings(pos_ids)[None,:,:] #[1,S,d_model]
        
        h= tok + pos
        
        for block in self.backbone:
            h= block(h)
        
        h = self.final_layer_norm(h) # [B,S,d_model]
        logits = self.lm_head(h) # [B,S,vocab_size]
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        num_new_tokens: int,
    ) -> Int[Tensor, "batch_size seq_len+num_new_tokens"]:

        # TODO, complete
        for _ in range(num_new_tokens):
            x_cond = x[:, -self.config.context_length :]
            logits = self(x_cond) # [B,S,vocab]
            next_token_logits = logits[:, -1, :] # [B,vocab]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # [B,1]
            x = torch.cat([x, next_token], dim=1) # [B,S ++]
        return x


    def get_loss_on_batch(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"], 
    ) -> Float[Tensor, ""]:

        # TODO, complete
        logits = self(input_ids) # [B,S,vocab_size]
        
        logits_for_pred = logits[:,:-1,:] # [B,S-1,vocab_size]
        targets = input_ids[:,1:] # [B,S-1]
        
        loss = F.cross_entropy(
            logits_for_pred.reshape(-1,logits_for_pred.size(-1)),
            targets.reshape(-1),
        )
        
        return loss


    @classmethod
    def from_pretrained(cls):
        """
        We simply always load up the GPT-2 model
        """

        # Config for GPT-2
        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            context_length=1024,
            vocab_size=50257,
        )

        model = cls(config)

        # Load weights from HuggingFace
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        converted_state_dict: Dict[str, Tensor] = state_dict_converter(model_hf.state_dict())

        model.load_state_dict(converted_state_dict)

        return model


if __name__ == "__main__":

    # Uncomment this if you are not logged in
    #huggingface_hub.login()
    
    model = Transformer.from_pretrained()
