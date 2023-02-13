# data preprocessing steps

## pre-process the table and keep the ocr_txt in 2-10 characters
select item_id, title, property_concat, tfs, combineOCRLoc_yixi(tci), getAllOcrLoc_yixi(tci) from fund_base_n_pict_text_training_before_preprocessing_with_cate_1
where cate_level1_id = 16 and getOCRnum_yixi(combineOCRLoc_yixi(tci)) > 1;

## remove very similar or same images in the dataset
python remove_redundant.py

## convert the removed table to json file
python table2json.py

## clean the ocr_txt with GPT-2, remove those with high ppl
python clean_anno.py

## resort the ocr_locs to make neighbour ocrs together
python resort_loc.py

## download the images and resize them in ratio with min_size of 256
python pict_download_preprocess.py