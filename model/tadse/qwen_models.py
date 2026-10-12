"""Qwen encoder adapter for the recovered TaDSE objective.

Training uses the backbone's native embedding width. Dimension truncation belongs
in evaluation, not in a newly learned projection.
"""

from copy import copy
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoModel, Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel

from .models import cl_forward, cl_init, sentemb_forward


class Qwen3ForPatternCL(Qwen3PreTrainedModel):
    """One Qwen3 encoder, with the original four-view TaDSE loss."""

    base_model_prefix = "model"

    def __init__(self, config, model_args=None, encoder=None):
        super().__init__(config)
        if model_args is None:
            stored = getattr(config, "tadse_model_args", None)
            if stored is None:
                raise ValueError("model_args are required for a new TaDSE model")
            model_args = SimpleNamespace(**stored)
        self.model_args = copy(model_args)
        if self.model_args.pooler_type != "pad_left_no_pooler":
            raise ValueError("Qwen TaDSE requires pad_left_no_pooler")
        # The recovered pattern transform reads this field as its input width.
        self.model_args.hidden_size = config.hidden_size
        config.tadse_model_args = vars(self.model_args).copy()
        config.use_cache = False
        if encoder is None:
            self.model = Qwen3Model(config)
            cl_init(self, config)
        else:
            if not isinstance(encoder, Qwen3Model):
                raise TypeError("encoder must be a Qwen3Model")
            if encoder.config.hidden_size != config.hidden_size:
                raise ValueError("encoder and wrapper widths must match")
            # Initialize only the new TaDSE modules before registering the already
            # pretrained encoder. Never run initialization over its weights.
            cl_init(self, config)
            self.model = encoder
        # cl_init creates these modules for the BERT path. Neither is used by
        # native last-token utterance pooling; avoid unused trainable parameters.
        del self.mlp
        if not self.model_args.apply_ptn_transform:
            del self.ptn_transform

    @classmethod
    def from_encoder_pretrained(
        cls, pretrained_model_name_or_path, *, model_args,
        attention_dropout=0.1, **kwargs,
    ):
        """Load an unmodified pretrained encoder, with an explicit dropout choice.

        The 0.1 dropout default is a new fine-tuning experiment setting, not a
        recovered historical Qwen recipe. Base-model tensors remain unchanged.
        """
        config = kwargs.pop("config", None)
        if config is None:
            config_keys = ("revision", "cache_dir", "local_files_only", "token")
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                **{key: kwargs[key] for key in config_keys if key in kwargs},
            )
        config.attention_dropout = attention_dropout
        config.use_cache = False
        encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path, config=config, **kwargs
        )
        return cls(config, model_args=model_args, encoder=encoder)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _encode(self, input_ids, attention_mask, **kwargs):
        # This bound method is an adapter, not another registered encoder module.
        kwargs.pop("token_type_ids", None)
        kwargs.pop("head_mask", None)
        # Explicit positions make left-padded batch and singleton encodings agree.
        if kwargs.get("position_ids") is None:
            positions = attention_mask.long().cumsum(-1) - 1
            kwargs["position_ids"] = positions.masked_fill(attention_mask == 0, 0)
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, **kwargs,
        )

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
        output_attentions=None, output_hidden_states=None, return_dict=None,
        sent_emb=False, mlm_input_ids=None, mlm_labels=None,
    ):
        if mlm_input_ids is not None or mlm_labels is not None:
            raise ValueError("Qwen TaDSE does not support the BERT MLM objective")
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and an explicit attention_mask are required")
        expected_rank = 2 if sent_emb else 3
        if input_ids.ndim != expected_rank or input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have matching view shapes")
        if not sent_emb and input_ids.shape[1] != 4:
            raise ValueError("TaDSE training requires exactly four views")
        mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        if mask.shape[-1] == 0 or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("attention_mask must be nonempty and binary")
        if not torch.all(mask[:, -1] == 1) or torch.any(mask[:, 1:] < mask[:, :-1]):
            raise ValueError("Qwen TaDSE requires left padding and at least one real token")
        if position_ids is not None and position_ids.shape == input_ids.shape:
            position_ids = position_ids.reshape(-1, position_ids.shape[-1])
        if inputs_embeds is not None:
            raise ValueError("inputs_embeds are not supported by the four-view adapter")
        forward = sentemb_forward if sent_emb else cl_forward
        return forward(
            self, self._encode, input_ids=input_ids,
            attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )
