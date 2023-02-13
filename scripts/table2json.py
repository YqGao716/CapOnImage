import json
from collections import Counter
import pdb
import re
from tqdm import tqdm


table = open("fund_base_n_pict_text_training_processed_woman_removed.txt",'r').readlines()
res = {}

for line in tqdm(table):
  line = line.strip()
  data = line.split('\t')
  img_id = data[0]

  title = ''.join(data[1].split())
  info = data[2].split('\x01')
  tmp = {}
  for i in range(len(info)):
    if len(info[i].split('\x02')) != 2:
      continue
    value, key = info[i].split('\x02')
    if key in tmp:
      tmp[key].append(value)
    else:
      tmp[key] = [value]
  if '功效' in tmp:
    info = title+"".join(tmp['功效'])
  else:
    info = title
  start = None
  if '【' in info:
    start = info.index('【')
    try:
      end = info.index('】')
    except:
      end = len(info)
  elif '[' in info:
    start = info.index('[')
    try:
      end = info.index(']')
    except:
      end = len(info)
  elif '(' in info:
    start = info.index('(')
    try:
      end = info.index(')')
    except:
      end = len(info)
  elif '（' in info:
    start = info.index('（')
    try:
      end = info.index('）')
    except:
      end = len(info)
  if start is not None:
    info = info[:start]+info[end+1:]
  info = re.sub('[\d+a-zA-Z]','',info)
  info = info.replace("{","")
  info = info.replace("/","")
  info = info.replace("！","")
  info = info.replace(":","")
  info = info.replace("-","")

  res[img_id] = {}
  res[img_id]['info'] = info
  res[img_id]['all_locs'] = data[-1].split(';')
  res[img_id]['ocr_locs'] = []
  res[img_id]['ocr_txts'] = []

  ocr_tuples = data[-2].split(';')
  for item in ocr_tuples:
    loc = '_'.join(item.split('_')[:-1])
    txt = item.split('_')[-1]
    clean_txt = re.sub('[\d+a-zA-Z!/！、>!]','',txt)
    if len(clean_txt) == len(txt):
      res[img_id]['ocr_txts'].append(txt)
      res[img_id]['ocr_locs'].append(loc)

  if len(res[img_id]['ocr_txts']) < 2:
    res.pop(img_id)


print(len(res))
json.dump(res,open("ref_captions.json",'w'))



