from cleantext import clean
import os
from volcenginesdkarkruntime import Ark
import re
os.environ['ARK_API_KEY'] = 'xxx'
Ark_base_url = 'https://ark.cn-beijing.volces.com/api/v3'
client = Ark(
    api_key=os.environ.get("ARK_API_KEY"),
    base_url=Ark_base_url,
    timeout=180, 
    max_retries=3,
)

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

RAW_CORPUS_FILE = os.path.join(os.getcwd(), "test_datasets", "openkp", "test_clean.json")
output_file = os.path.join(os.getcwd(), "test_datasets", "openkp", "test-fix.json") 
res_list = []

with open(RAW_CORPUS_FILE, "r") as f:
    for idx, line in tqdm(enumerate(f)):
        if idx >= 2200:
            continue

        tmp_dict = {}
        line = json.loads(line)
        raw_text = line['text']
        tmp_src = raw_text.split()[:2048]
        raw_text = ' '.join(tmp_src)

        template = f'''Please add punctuation to the following text to make it more semantically fluent. Do not emphasize the text; if you add `'\n'`, please add `.`\nText content: {raw_text}\n Only output the modified text. Modified text:'''
        try:
            completion = client.chat.completions.create(
                model="model_ids",
                messages=[
                    {"role": "system", "content": "You are an expert in natural language processing!"},
                    {"role": "user", "content": template},
                ],
                temperature=0.5,  
                )
 
            res = completion.choices[0].message.content
        except:
            print('sample {} is broken'.format(idx))
            continue
        text = clean_func(res)
        tmp_dict['text'] = text
        keywords = line['KeyPhrases']
        keyphrases = [kp.lower() for kp in keywords]
        present_keyphrases, absent_keyphrases = split_present_absent(text, keyphrases)
        tmp_dict['present_kp'] = present_keyphrases
        tmp_dict['absent_kp'] = absent_keyphrases  
        res_list.append(tmp_dict)

with open(output_file, "w") as f:
    json.dump(res_list, f, indent = 2)

