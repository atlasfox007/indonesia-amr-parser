from gsii_parser.amrlib.models.parse_gsii.bert_utils import BertEncoderTokenizer, BertEncoder
from gsii_parser.amrlib.models.parse_gsii.data_loader import ArraysToTensor

import torch 

tokenizer = BertEncoderTokenizer.from_pretrained("indobert-base-p2", do_lower_case=False)

vocabs = {}
vocabs["bert_tokenizer"] = tokenizer

tokenizer = vocabs["bert_tokenizer"]

bert_token, token_subword_index = tokenizer.tokenize(['halo', 'nama', 'saya', 'ijat'])

print(bert_token)
print(token_subword_index)

bert_encoder = BertEncoder.from_pretrained("indobert-base-p2")
for p in bert_encoder.parameters():
    p.requires_grad = False

bert_token = torch.from_numpy(bert_token).unsqueeze(0)
token_subword_index = torch.from_numpy(token_subword_index).unsqueeze(0).to(torch.long)
# token_subword_index = torch.from_numpy(token_subword_index)
bert_embed,_ = bert_encoder(bert_token, token_subword_index)

print(bert_embed.shape)
bert_embed = bert_embed.transpose(0, 1)
print(bert_embed.shape)
