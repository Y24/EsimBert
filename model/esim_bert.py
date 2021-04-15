from model.utils import replace_masked
from model.layers import SoftmaxAttention
import torch
from torch import nn
from transformers import RobertaModel


class EsimBERT(nn.Module):
    def __init__(self, bert_type: str = "roberta-base", num_classes=3, dropout=0.5):
        super(EsimBERT, self).__init__()
        print("Init EsimBert model...")
        self.bert_hidden_sizes = {"roberta-base": 768, "roberta-large": 1024}
        assert(bert_type in self.bert_hidden_sizes.keys())
        self.bert_type = bert_type
        self.hidden_size = self.bert_hidden_sizes[bert_type]
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.dropout = dropout
        self.bert_model = RobertaModel.from_pretrained(
            bert_type, hidden_size=self.hidden_size)
        for i in self.parameters():
            i.requires_grad = False
        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        print("EsimBert model is bulit!")

    def forward(self,
                premises_ids,
                premises_mask,
                hypotheses_ids,
                hypotheses_mask):
        encoded_premises = self.bert_model(
            premises_ids, attention_mask=premises_mask)['last_hidden_state']
        encoded_hypotheses = self.bert_model(
            hypotheses_ids, attention_mask=hypotheses_mask)['last_hidden_state']
        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        """ v_ai = self._composition(projected_premises, premises_mask < .5)
        v_bj = self._composition(projected_hypotheses, hypotheses_mask < .5) """
        v_ai = projected_premises
        v_bj = projected_hypotheses
        v_ai = nn.functional.softmax(v_ai, dim=-1)
        v_bj = nn.functional.softmax(v_bj, dim=-1)
        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


if __name__ == "__main__":
    model = EsimBERT()
