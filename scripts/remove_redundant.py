import numpy as np
import json
import pdb
import collections
import time

def check(tocompare, total, cur):
  thres = round(total * 0.7)
  counter = collections.Counter(tocompare)
  for key, value in counter.most_common(2):
    if key != cur and value >= thres:
      return False
  return True

a = open("fund_base_n_pict_text_training_processed_woman.txt",'r').readlines()
unique = open("fund_base_n_pict_text_training_processed_woman_removed.txt",'w')

cap2id = {}
for i in range(len(a)):
  if i % 100000 == 0:
    print(i)
  ocr_txt = a[i].strip().split('\t')[-2]
  ocr_txt = ocr_txt.split(';')
  for item in ocr_txt:
    cap = item.split('_')[-1]
    if cap not in cap2id:
      cap2id[cap] = [i]
    else:
      cap2id[cap].append(i)

save = set()
for i in range(len(a)):
  if i % 1000 == 0:
    print(i+1, len(save), len(save)/(i+1))
  ocr_txt = a[i].strip().split('\t')[-2]
  ocr_txt = ocr_txt.split(';')
  tocompare = []
  for item in ocr_txt:
    cap = item.split('_')[-1]
    tocompare.extend(cap2id[cap])
  if not check(tocompare, len(ocr_txt), i):
    for item in ocr_txt:
      cap = item.split('_')[-1]
      cap2id[cap].remove(i)
  else:
    save.add(i)
    unique.write(a[i])

unique.close()


