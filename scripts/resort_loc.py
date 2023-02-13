import json
import numpy as np
import pdb
from tqdm import tqdm

def getCenter(ocr_locs):
  centers = []
  for loc in ocr_locs:
    x1, y1, x2, y2 = loc.split('_')
    centers.append([(float(x1)+float(x2))/2, (float(y1)+float(y2))/2])
  return centers

def findNearst(anchor, centers):
  centers = np.array(centers)
  anchor = np.array(anchor)
  dist = pow(centers-anchor, 2).sum(axis=-1)
  return dist.argmin()

def resort(centers, old_locs, old_txts):
  new_locs, new_txts = [], []
  prev = 0
  while len(centers) > 0:
    new_locs.append(old_locs[prev])
    new_txts.append(old_txts[prev])
    old_locs = old_locs[:prev]+old_locs[prev+1:]
    old_txts = old_txts[:prev]+old_txts[prev+1:]
    cur = centers[prev]
    centers = centers[:prev]+centers[prev+1:]
    if len(centers) == 0:
      break
    prev = findNearst(cur, centers)
  return new_locs, new_txts

ref = json.load(open("ref_captions_clean.json"))

for key in tqdm(ref):
  ocr_locs = ref[key]['ocr_locs']
  centers = getCenter(ocr_locs)
  ref[key]['ocr_locs'], ref[key]['ocr_txts'] = resort(centers, ocr_locs, ref[key]['ocr_txts'])

json.dump(ref, open("ref_captions.json",'w'))


