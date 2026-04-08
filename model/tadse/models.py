import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import Qwen3Model
from transformers.activations import gelu
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.models.bert.modeling_bert import (
    BertLMPredictionHead,
    BertModel,
    BertPreTrainedModel,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaModel,
    RobertaPreTrainedModel,
)


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class PatternMLPLayer(nn.Module):
    """
    Head for getting pattern representations.
    """

    def __init__(
            self, config, ptn_dim, activation="tanh"
    ):
        super().__init__()
        self.dense = nn.Linear(ptn_dim, config.hidden_size)
        # self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.use_non_linear_transformation = use_non_linear_transformation

        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        # if self.use_non_linear_transformation:
        #     x = self.dense2(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'pad_left_no_pooler': Pad sentence to left without pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "pad_left_no_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], (
                "unrecognized pooling type %s" % self.pooler_type
        )

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        elif self.pooler_type == "pad_left_no_pooler":
            return last_hidden[:, -1]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    cls.mlp = MLPLayer(config)
    cls.simcse_utt_sim = Similarity(temp=cls.model_args.simcse_utt_temp)
    cls.simcse_ptn_sim = Similarity(temp=cls.model_args.simcse_ptn_temp)
    cls.pairwise_sim = Similarity(temp=cls.model_args.pairwise_temp)
    # another layer specifically for patterns, as to try to bring them into similar utterance hyperspace.
    cls.ptn_transform = PatternMLPLayer(
        config,
        cls.model_args.hidden_size
    )
    # TODO : Pattern Transformation Experiments
    # TODO : Difference with EASE is that this layer comes on top of Pattern representations
    # cls.entity_transformation = EntityMLPLayer(
    #     config,
    #     cls.model_args.entity_emb_dim,
    #     cls.model_args.use_non_linear_transformation,
    #     cls.model_args.activation,
    # )
    #
    # if cls.model_args.use_another_transformation_for_hn:
    #     cls.hn_entity_transformation = MLPLayer(config)
    # else:
    #     cls.hn_entity_transformation = cls.entity_transformation

    cls.init_weights()


def cl_forward(
        cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # MLMs not used now
        mlm_input_ids=None,
        mlm_labels=None
):

    """
    The main difference between ours and SimCSE's original implementation is that
    we also use our novel pattern contrastive learning loss between sentences and their related patterns.
    """
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    attention_mask = attention_mask.view(
        (-1, attention_mask.size(-1))
    )  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(
            (-1, token_type_ids.size(-1))
        )  # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True
        if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
        else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True
            if cls.model_args.pooler_type in ["avg_top2", "avg_first_last"]
            else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view(
        (batch_size, num_sent, pooler_output.size(-1))
    )  # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    # TODO : Apparently there's a forcible use hyperparam
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    # TODO : Main Difference of our TaDSE model
    utt1, utt2, ptn1, ptn2 = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2], pooler_output[:, 3]

    if cls.model_args.apply_ptn_transform:
        ptn1 = cls.ptn_transform(ptn1)
        ptn2 = cls.ptn_transform(ptn2)
    # TODO: Hard negative?
    # print(f"Sizes : Utt - {utt1.size()}, {utt2.size()}, Patt - {ptn1.size()}, {ptn2.size()}")

    # Dist training is removed here

    # Embeddings
    pairwise_cos_sim_ptn_neg = cls.pairwise_sim(utt1.unsqueeze(1), ptn1.unsqueeze(0))
    pairwise_cos_sim_utt_neg = cls.pairwise_sim(ptn1.unsqueeze(1), utt1.unsqueeze(0))
    utt_cos_sim = cls.simcse_utt_sim(utt1.unsqueeze(1), utt2.unsqueeze(0))
    ptn_cos_sim = cls.simcse_ptn_sim(ptn1.unsqueeze(1), ptn2.unsqueeze(0))

    # TODO : Ptn transformation - report results with / without pattern transformation
    # if cls.model_args.use_entity_transformation:
    #     entity_embedding = cls.entity_transformation(entity_embedding)

    # labels: 0, 1?
    labels = torch.arange(utt_cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    simcse_utt_loss = loss_fct(utt_cos_sim, labels)
    simcse_ptn_loss = loss_fct(ptn_cos_sim, labels)

    # print(f"Losses : Utt - {simcse_utt_loss.size()}, Patt - {simcse_ptn_loss.size()}")

    # pairwise contrastive learning loss
    pairwise_labels = torch.arange(pairwise_cos_sim_ptn_neg.size(0)).long().to(cls.device)
    pairwise_ptn_neg_loss = loss_fct(pairwise_cos_sim_ptn_neg, pairwise_labels)
    pairwise_utt_neg_loss = loss_fct(pairwise_cos_sim_utt_neg, pairwise_labels)

    pairwise_loss = pairwise_ptn_neg_loss if cls.model_args.pairwise_neg_type == "pattern" else \
        pairwise_utt_neg_loss if cls.model_args.pairwise_neg_type == "utterance" else \
        0.5 * pairwise_utt_neg_loss + 0.5 * pairwise_ptn_neg_loss if cls.model_args.pairwise_neg_type == "balanced" else \
        None

    # print(f"Pairwise Loss : Pairwise - {pairwise_loss.size()}")

    # TODO : pattern transformation?

    # TODO : Consider utt2 ptn2 loss
    # TODO : Consider utt1 ptn2 loss

    # Calculate loss for pairwise model.
    loss = (
            cls.model_args.simcse_utt_loss_ratio * simcse_utt_loss
            + cls.model_args.simcse_ptn_loss_ratio * simcse_ptn_loss
            + cls.model_args.pairwise_loss_ratio * pairwise_loss
    )

    # Calculate loss for MLM
    masked_lm_loss = None
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1)
        )
        loss = loss + cls.model_args.mlm_loss_ratio * masked_lm_loss

    if not return_dict:
        output = (utt_cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    sequence_classifier_output = SequenceClassifierOutput(
        loss=loss,
        logits=utt_cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    sequence_classifier_output.simcse_utt_loss = (
            cls.model_args.simcse_utt_loss_ratio * simcse_utt_loss
    )
    sequence_classifier_output.simcse_ptn_loss = (
            cls.model_args.simcse_ptn_loss_ratio * simcse_ptn_loss
    )
    sequence_classifier_output.pairwise_loss = cls.model_args.pairwise_loss_ratio * pairwise_loss
    sequence_classifier_output.mlm_loss = None
    if masked_lm_loss is not None:
        sequence_classifier_output.mlm_loss = (
                cls.model_args.mlm_loss_ratio * masked_lm_loss
        )
    return sequence_classifier_output


## Sentence embedding output
def sentemb_forward(
        cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True
        if cls.pooler_type in ["avg_top2", "avg_first_last"]
        else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if (
            cls.pooler_type == "cls" and not cls.model_args.mlp_only_train
    ):
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )

class Qwen3ForPatternCL(Qwen3Model):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.model = Qwen3Model(config)
        cl_init(self, config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sent_emb=False,
            mlm_input_ids=None,
            mlm_labels=None
    ):
        if sent_emb:
            return sentemb_forward(
                self,
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(
                self,
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

class BertForPatternCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config)
        self.lm_head = BertLMPredictionHead(config)
        cl_init(self, config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sent_emb=False,
            mlm_input_ids=None,
            mlm_labels=None
    ):
        if sent_emb:
            return sentemb_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(
                self,
                self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class RobertaForPatternCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        cl_init(self, config)

    def init_pattern_embedding(self, pattern_embeddings):
        self.pattern_embedding.weight = nn.Parameter(
            torch.FloatTensor(pattern_embeddings)
        )
        self.pattern_embedding.weight.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sent_emb=False,
            mlm_input_ids=None,
            mlm_labels=None
    ):
        if sent_emb:
            return sentemb_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(
                self,
                self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                title_id=title_id,
                hn_title_ids=hn_title_ids,
            )
