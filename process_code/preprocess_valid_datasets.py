from cleantext import clean
import re
import os
from random import shuffle
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
valid_corpus_path = './valid_corpus.json'
valid_list = []
for name in ["duc", "openkp", "kptimes", "stackexchange", "kpbiomed"]:
    data_name = name
    if name == 'duc':
        continue
    elif name =='kpbiomed':
        RAW_CORPUS_FILE = os.path.join(os.getcwd(), "test_datasets", name, "val.jsonl")
    else:
        RAW_CORPUS_FILE = os.path.join(os.getcwd(), "test_datasets", name, "valid.json")
    print(os.path.exists(RAW_CORPUS_FILE))
    # RAW_CORPUS_FILE = "../test_datasets/{}/valid.json".format(data_name)      
    all_clean_texts = []
    with open(RAW_CORPUS_FILE, "r", encoding='utf-8') as f:
        for line in tqdm(f):
            tmp_dict = {}
            line = json.loads(line)
            if data_name == 'openkp':
                text = clean_func(line['text'])
                tmp_dict['text'] = text
            elif data_name =='stackexchange':
                text = clean_func(line['title']) + '. '+ clean_func(line['question'])
                tmp_dict['text'] = text
            else:
                title, abstract = clean_func(line["title"]), clean_func(line["abstract"])
                title = re.sub(' +', ' ', title)
                abstract = re.sub(' +', ' ', abstract)
                if not title.endswith('.'):
                    title = title + "."
                tmp_dict['title'] = title
                tmp_dict['abstract'] = abstract      
                text = title + " " + abstract
            if data_name == 'stackexchange':
                keywords = line["tags"].split(';') 
            elif data_name == 'kptimes':
                keywords = line['keyword'].split(';')
            elif data_name == 'openkp':
                keywords = line['KeyPhrases']
            elif data_name == 'kpbiomed':
                keywords = line['keyphrases']
            else:
                keywords = line['keywords'].split(';')
            keyphrases = [kp.lower() for kp in keywords]
            present_keyphrases, absent_keyphrases = split_present_absent(text, keyphrases)
            tmp_dict['present_kp'] = present_keyphrases
            tmp_dict['absent_kp'] = absent_keyphrases  
            tmp_dict['domain'] = data_name
            all_clean_texts.append(tmp_dict)
    valid_list.extend(all_clean_texts[:2000])
    shuffle(valid_list)
    output_file = os.path.join(os.getcwd(), "test_datasets", data_name, "valid_clean.json")
    with open(output_file, "w") as f:
        json.dump(all_clean_texts[:2000], f, indent = 2)
    with open(valid_corpus_path, "w") as fr:
        json.dump(valid_list, fr, indent = 2)    
    

         

