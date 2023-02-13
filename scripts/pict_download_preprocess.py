import cv2
import os
import glob
from multiprocessing import Pool
import torchvision.transforms as transforms
import wget
import random
import json
import numpy as np
from PIL import Image
import numpy as np
import urllib.request
import sys

#pict_url_file = '/disk5/xinglin.hxl/pict/matching_clip_test_sample.txt'
#pict_url_file = '/disk5/xinglin.hxl/pict/fund_base_n_clip_matching_item_training_data.txt'
#pict_url_file = '/disk4/yuqing.syq/pict/fund_base_n_pict_text_wt_eval_processed.txt'
#pict_url_file = '/disk4/yuqing.syq/pict/fund_base_n_pict_text_training_processed_woman_removed.txt'
pict_url_list = []
file = json.load(open("ref_captions.json"))
for key in file:
    pict_url = file[key]['url'].split('/')[-1]
    pict_url_list.append((key, pict_url))

# with open(pict_url_file, 'r') as file:
#     for line in file:
#         line = line.strip()
#         if not line:
#             continue
#         tmp_list = line.split('\t')
#         content_id = tmp_list[0]
#         pict_url = tmp_list[3]
#         pict_url_list.append((content_id, pict_url))

random.shuffle(pict_url_list)

def resize_ratio(img, item_id, min_size=256):
    h, w, c = img.shape
    if h < w:
        nh = min_size
        nw = int(w / h * min_size)
    else:
        nw = min_size
        nh = int(h / w * min_size)
    img = cv2.resize(img, (nw, nh))
    return img, h, w

def process_download_pict(part_pict_list):
    cnt = 0
    for idx, (item_id, pict_url) in enumerate(part_pict_list):
        try:
            attempts = 0
            success = False
            while attempts < 3 and not success:
                img_content = urllib.request.urlopen('http://asearch.alicdn.com/bao/uploaded/' + pict_url, timeout=3).read()
                if img_content is not None:
                    success = True
                else:
                    attempts += 1

            if img_content is not None:
                img_data = np.asarray(bytearray(img_content), dtype='uint8')
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img, h, w = resize_ratio(img, item_id)
                img = Image.fromarray(img)
                img.save('./tuwen_imgs/' + item_id + '.png')

        except Exception as e:
            print(e)
            print(line)

        print("{}: {} done.".format(idx + 1, os.path.basename('./tuwen_imgs/' + item_id + '.png')[:-4]))

num_process = 80
print('Parent process %s.' % os.getpid())
p = Pool(num_process)
part_list_len = len(pict_url_list) // num_process
for i in range(num_process):
    if i == num_process - 1:
        part_list = pict_url_list[part_list_len*i:]
    else:
        part_list = pict_url_list[part_list_len*i:part_list_len*(i+1)]
    p.apply_async(process_download_pict, args=(part_list,))
print('Waiting for all subprocesses done...')
p.close()
p.join()
print('All subprocesses done.')
