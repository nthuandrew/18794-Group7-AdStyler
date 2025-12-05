import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertPooler
from typing import Optional, Union, List, Tuple
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    # logging,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)


from transformers import BertConfig


class BertEmbeddingsWithMeta(nn.Module):
    def __init__(self, config, metadata_vocab_size: int = None, metadata_emb_dim: int = None):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # metadata embedding layer (optional)
        self.metadata_embeddings = None
        if metadata_vocab_size is not None and metadata_emb_dim is not None:
            assert metadata_emb_dim == config.hidden_size, "metadata_emb_dim should match hidden_size"
            self.metadata_embeddings = nn.Embedding(metadata_vocab_size, metadata_emb_dim)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        metadata_ids=None,
        metadata_embeds=None,
        past_key_values_length: int = 0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            buffered = self.token_type_ids[:, :seq_length]
            token_type_ids = buffered.expand(batch_size, seq_length)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeds = self.token_type_embeddings(token_type_ids)
        position_embeds = self.position_embeddings(position_ids) if self.position_embedding_type == "absolute" else 0

        embeddings = inputs_embeds + token_type_embeds + position_embeds

        # 如果有 metadata，則把 metadata embedding 加進 embeddings
        if metadata_ids is not None:
            meta = self.metadata_embeddings(metadata_ids)  # (batch, hidden_size)
            meta = meta.unsqueeze(1).expand(-1, seq_length, -1)
            embeddings = embeddings + meta
        elif metadata_embeds is not None:
            meta = metadata_embeds.unsqueeze(1).expand(-1, seq_length, -1)
            embeddings = embeddings + meta

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModelWithMeta(BertPreTrainedModel):
    def __init__(self, config, metadata_vocab_size: int = None, metadata_emb_dim: int = None, **kwargs):
        super().__init__(config, **kwargs)
        self.embeddings = BertEmbeddingsWithMeta(config, metadata_vocab_size, metadata_emb_dim)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if getattr(config, "add_pooling_layer", True) else None
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        metadata_ids=None,
        metadata_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            metadata_ids=metadata_ids,
            metadata_embeds=metadata_embeds,
            past_key_values_length=(past_key_values[0][0].shape[2] if past_key_values is not None else 0),
        )

        # 用與 standard BERT 一樣的 encoder pipeline
        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, embedding_output.size()[:-1])

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

# --- 測試 / main flow (dummy data) ---
if __name__ == "__main__":
    # config + model 初始化
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        max_position_embeddings=512,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        add_pooling_layer=False
    )
    metadata_vocab_size = 10  # 假設 metadata 總共有 10 種可能 label
    model = BertModelWithMeta(config, metadata_vocab_size=metadata_vocab_size, metadata_emb_dim=config.hidden_size)

    # dummy input: 假設我們有 batch_size = 2, 每句 5 個 token
    batch_size = 2
    seq_len = 5
    dummy_input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    # dummy metadata ids
    dummy_metadata_ids = torch.tensor([3, 7], dtype=torch.long)  # batch_size = 2

    outputs = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        metadata_ids=dummy_metadata_ids,
    )
    
    # print("last_hidden_state.shape:", outputs.last_hidden_state.shape)
    # if outputs.pooler_output is not None:
    #     print("pooler_output.shape:", outputs.pooler_output.shape)