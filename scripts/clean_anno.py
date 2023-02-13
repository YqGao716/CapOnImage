# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer, GPT2LMHeadModel
import pdb
import json
from tqdm import tqdm
import io , sys
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

def attn_mask(lengths, max_len=None):
  ''' Creates a boolean mask from sequence lengths.
      lengths: LongTensor, (batch, )
  '''
  batch_size = lengths.size(0)
  max_len = max_len or lengths.max()
  return ~(torch.arange(0, max_len)
          .type_as(lengths)
          .repeat(batch_size, 1)
          .ge(lengths.unsqueeze(1)))

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model.cuda()
model.eval()

newdic = {}
ann = json.load(open("ref_captions.json"))

for key in tqdm(ann):
  newdic[key] = {}
  newdic[key]['info'] = ann[key]['info']
  newdic[key]['all_locs'] = ann[key]['all_locs']
  newdic[key]['ocr_txts'] = []
  newdic[key]['ocr_locs'] = []

  tokens_batch, lens = [], []
  for i in range(len(ann[key]['ocr_txts'])):
    idx = tokenizer.encode(ann[key]['ocr_txts'][i])
    tokens_batch.append(idx + [102] * max(0, (12-len(idx))))
    lens.append(len(ann[key]['ocr_txts'][i]))
  tokens_tensor = torch.tensor(tokens_batch).cuda()
  lens = torch.tensor(lens).cuda()
  mask = attn_mask(lens-1, max_len=9)

  with torch.no_grad():
    outputs = model(tokens_tensor)[0]
    predictions = outputs[:, 1:-2]
    probs = torch.gather(predictions, -1, tokens_tensor[:, 2:-1].unsqueeze(-1)).squeeze(-1)
    maxprobs = predictions.max(dim=-1)[0]
    score = (probs - maxprobs) * mask
    score = (score.sum(dim=-1) / mask.sum(dim=-1)).detach().cpu().tolist()

  for i in range(len(ann[key]['ocr_txts'])):
    text = ann[key]['ocr_txts'][i]
    if score[i] > -4.5 or text in ann[key]['info']:
      clean_txt = text.replace(']',"")
      clean_txt = clean_txt.replace('-',"")
      clean_txt = clean_txt.replace('￥',"")
      clean_txt = clean_txt.replace('口',"")
      x1, y1, x2, y2 = ann[key]['ocr_locs'][i].split('_')[:4]
      if int(x2) <= int(x1) or int(y2) <= int(y1) or len(clean_txt) < 2:
        continue
      newdic[key]['ocr_txts'].append(clean_txt)
      newdic[key]['ocr_locs'].append('_'.join([x1, y1, x2, y2]))

  if len(newdic[key]['ocr_locs']) < 2:
    newdic.pop(key)

print(len(ann))
print(len(newdic))
json.dump(newdic, open("ref_captions_clean.json",'w'))

