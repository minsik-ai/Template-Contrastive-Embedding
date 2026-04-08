import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from model.module import IntentClassifier


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)

        self.bert = BertModel(config=config)  # Load pretrained bert
        for param in self.bert.parameters():
            param.requires_grad = False

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)

        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))

        outputs = (intent_loss, intent_logits, pooled_output)  # add hidden states and attention if they are here

        return outputs  # intent_loss, intent_logits, embeddings
