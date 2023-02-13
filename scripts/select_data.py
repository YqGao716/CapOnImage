#coding=utf-8
#-*- coding:utf-8 -*-
from __future__ import print_function
from odps.udf import annotate
from odps.distcache import get_cache_archive
import re

@annotate("string->bigint")
class getOCRnum(object):
    def evaluate(self, arg0):
        if arg0 == "":
            return 0
        data = arg0.split(';')
        return len(data)

@annotate("string->string")
class removeSpace(object):
    def evaluate(self, arg0):
        data = arg0.split('\n')
        return "".join(data)

@annotate("string->string")
class combineOCRLoc(object):
    def evaluate(self, arg0):
        if arg0 == "":
            return ""
        res = []
        data = arg0.split(';')
        ocrs = {}
        for item in data:
            ocr_txt = item.split(",")[-1]
            if ocr_txt not in ocrs:
                ocrs[ocr_txt] = 1
            else:
                ocrs[ocr_txt] += 1

        for item in data:
            ocr_txt = item.split(",")[-1]
            ocr_loc = ','.join(item.split(",")[:-1]).split(":")[0]
            coords = ocr_loc.split(',')
            x1, y1 = coords[:2]
            x2, y2 = coords[4:6]
            ocr_loc = '_'.join([x1, y1, x2, y2])
            if ocr_txt != '淘鲜' and ocrs[ocr_txt] == 1:
                ocr_txt = self.clean(ocr_txt)
                if ocr_txt != "" and self.is_valid(ocr_txt):
                    res.append('_'.join([ocr_loc, ocr_txt]))
        if len(res) == 1:
            return ""
        return ';'.join(res)
    
    def is_valid(self, txt):
        if self.getLength(txt) < 2 or self.getLength(txt) > 10:
            return False 
        my_re = re.compile(r'[A-Za-z]',re.S)
        if len(re.findall(my_re, txt)):
            return False
        txt = txt.decode('utf-8')
        for ch in txt:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False
    
    def getLength(self, x):
        if x[0].isalpha():
            return len(x.split(' '))
        else:
            return len(x.decode("utf-8"))
    
    def clean(self, x):
        x = x.decode("utf-8")
        x = re.sub("[|[.(),^<+——()?【】“”！，。？、~@#￥%……&*（）°“”：]+", "",x)
        x = x.encode("utf-8")
        x = x.replace("\xe2\x80\x9d", "")
        x = x.replace("\xe3\x80\x91", "")
        x = x.replace("\xef\xbc\x9a", "")
        x = x.replace("\xef\xbc\x8c", "")
        x = x.replace("\xe3\x80\x82", "")
        x = x.replace("\xe2\x80\x9c", "")
        x = x.replace("\xe3\x80\x90", "")
        x = x.replace("\xc2\xb7", "")
        return "".join(x.split())

@annotate("string->string")
class getAllOcrLoc(object):
    def evaluate(self, arg0):
        if arg0 == "":
            return ""
        res = []
        data = arg0.split(';')
        for item in data:
            ocr_loc = ','.join(item.split(",")[:-1]).split(":")[0]
            coords = ocr_loc.split(',')
            x1, y1 = coords[:2]
            x2, y2 = coords[4:6]
            res.append('_'.join([x1, y1, x2, y2]))
        return ';'.join(res)


        