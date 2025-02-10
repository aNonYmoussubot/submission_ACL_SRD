from cleantext import clean
import re
import os
workspace = 'xxx'
clean_func = lambda x: clean(
    x,
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=False,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_url="<url>",
    replace_with_email="<email>",
    replace_with_phone_number="<phone>",
    lang="en"                       # set to 'de' for German special handling
)

def split_present_absent(text, keyphrases):
    # stemmed_text = stem_text(text)
    # stemmed_keyphrases = [stem_text(kp) for kp in keyphrases]
    # return [kp for kp in stemmed_keyphrases if kp in stemmed_text], [kp for kp in stemmed_keyphrases if kp not in stemmed_text]
    return [kp for kp in keyphrases if kp in text], [kp for kp in keyphrases if kp not in text]

import json
from tqdm import tqdm

RAW_CORPUS_FILE =  os.path.join(os.getcwd(), "kp20k_corpus", "train.json")
RAW_CORPUS_FILE = f'{workspace}/raw_kp_corpus/kp20k/train.json'
all_clean_texts = []
idx = 0
with open(RAW_CORPUS_FILE, "r", encoding='utf-8') as f:
    for line in tqdm(f):
        tmp_dict = {}
        line = json.loads(line)
        title, abstract = clean_func(line["title"]), clean_func(line["abstract"])
        title = re.sub(' +', ' ', title)
        abstract = re.sub(' +', ' ', abstract)
        if not title.endswith('.'):
            title = title + "."
        tmp_dict['title'] = title
        tmp_dict['idx'] = idx
        tmp_dict['abstract'] = abstract      
        text = title + " " + abstract
        keywords = line['keywords']
        keyphrases = [kp.lower() for kp in keywords]
        present_keyphrases, absent_keyphrases = split_present_absent(text, keyphrases)
        tmp_dict['present_kp'] = present_keyphrases
        tmp_dict['absent_kp'] = absent_keyphrases
        tmp_dict['text'] = text  
        all_clean_texts.append(tmp_dict)
        idx += 1
output_file = './kp20k_clean.json'
with open(output_file, "w") as f:
    json.dump(all_clean_texts, f, indent = 2)

