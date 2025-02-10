import os
import json
import sys
import openai
from pathlib import Path
from tqdm import tqdm
from volcenginesdkarkruntime import Ark
os.environ['ARK_API_KEY'] = 'xxx'
Ark_base_url = 'https://ark.cn-beijing.volces.com/api/v3'
client = Ark(
    api_key=os.environ.get("ARK_API_KEY"),
    base_url=Ark_base_url,
    timeout=180, 
    max_retries=3,
)

# stack, duc, kptimes, openkp, kpbiomed
# chatgpt, gp4o, deepseek

llm_name = 'deepseek'
model_map = {
    "chatgpt": "gpt-3.5-turbo",
    "gpt4o": "gpt-4o-2024-11-20",
    "deepseek": "deepseek/deepseek-chat" 
}
data_name = 'kptimes'
 
input_file_dir =  os.path.join(os.getcwd(), "test_datasets", data_name) 
input_data_file = '{}/test_clean_close.json'.format(input_file_dir)

output_file_dir = os.path.join(os.getcwd(), "output", data_name, llm_name) 
if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)    
output_data_file = "{}/pred.json".format(output_file_dir)


with open(input_data_file, 'r', encoding='utf-8') as file:
    data = json.load(file)[:]

res_list = []
for idx, each_line in tqdm(enumerate(data), total=len(data)):
    res_dict = each_line
    if data_name == 'openkp' or data_name == 'stackexchange':
        text = each_line['text']
    else:
        title = each_line['title']
        abstracts = each_line['abstract']
        text = title + abstracts

    template = f'''
=======\nPlease generate accurate present and absent keyphrases in lowercase for the given sample.\n
=======\n" ####Sample####: \n"text": "{text}"\n" **please only output at least five present and absent keyphrases with standard json format. **'''
       
    try:

        completion = client.chat.completions.create(
            model="xxx",
            messages=[
                {"role": "system", "content": "You are an expert in keyphrase generation!"},
                {"role": "user", "content": template},
            ],
            temperature=0.5, 
        )
        res = completion.choices[0].message.content
        a = res.replace('```json', '').replace('```', '').replace('\n', '')
        res11 = json.loads(a)
    except:
        log_info = "line {} is broken".format(idx)
        print(log_info)
        
    print(res11)
    res_dict['predict'] = res11
    res_list.append(res_dict)
      

with open(output_data_file, 'w', encoding='utf-8') as f:
    json.dump(res_list, f, ensure_ascii=False, indent=4)




