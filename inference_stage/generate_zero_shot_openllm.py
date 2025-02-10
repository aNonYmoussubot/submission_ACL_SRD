import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import sys
import math
import torch
import argparse
import transformers
from transformers import GenerationConfig, AwqConfig
import re
from tqdm import tqdm
from awq import AutoAWQForCausalLM
from typing import Dict, Optional, Sequence


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.19, help='')
    parser.add_argument('--top_p', type=float, default=0.7, help='')
    parser.add_argument('--do_sample', type=bool, default=False, help='')    
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    parser.add_argument('--input_data_file', type=str, default='input_data/', help='')
    parser.add_argument('--output_data_file', type=str, default='output_data/', help='')
    parser.add_argument('--trained_model', type=str, default='')
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--llm_name', type=str, default='')        
    parser.add_argument('--num_beams', type=int, default=5, help='Number of beams for beam search')
    parser.add_argument('--num_return_sequences', type=int, default=5, help='Number of sequences to return')

    args = parser.parse_args()
    return args

def formatting_prompts_func(text, tokenizer):
    output_texts = []
    messages = []   
    template= f'''
=======\nPlease generate accurate present and absent keyphrases in lowercase for the given sample.\n
=======\n" ####Sample####: \n"text": "{text}"\n" **please only output at least five present and absent keyphrases with standard json format.**'''
            
    system = {"role": "system", 
            "content": "You are an expert in keyphrase generation!"}
    user = {
        "content": template,
        "role": "user"
    }  
    # Append the 'user' message to the 'messages' list.
    messages.append(system)
    messages.append(user)   
    res = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    output_texts.append(res) 
    return output_texts



def build_generator(
    args, item, model, tokenizer, temperature=0.15, top_p=0.19, num_beams=5, num_return_sequences=5,  do_sample=False, max_gen_len=4096, use_cache=True
):
    def response(item):
        if args.data_name == 'openkp' or args.data_name == 'stack':
            text = item['text']
        else:
            title = item['title']
            abstracts = item['abstract']
            text = title + ' ' + abstracts
        
        prompt = formatting_prompts_func(text, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            do_sample = False,  ## used for whether use temperature and top_p
            temperature=0.5, # temperature, 
            use_cache=use_cache
        )
        print(len(output))
        out = tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        out = out.split(prompt[0])[1].replace('<|eot_id|>','').strip()

        print(out)
        return out

    return response


def fix_json_string(input_str):
    output = re.sub(r"(\w+)'s", r'\1"s', input_str)
    output = re.sub(r"'([^']*)'", r'"\1"', output)
    output = re.sub(r',\s*\]', ']', output)
    if not re.search(r'\]\s*$', output):
        output = re.sub(r',\s*$', ']', output)    
    if not re.search(r'}\s*$', output):
        output += '}'
    match = re.search(r'\{.*?\}', output, re.DOTALL)
    if match:
        output = match.group(0)
    return output




def main(args):
    quantization_config = AwqConfig(
        bits=4,
        fuse_max_seq_len=2048, # Note: Update this as per your use-case
        do_fuse=False,
    )


    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )
    if args.llm_name == 'llama':
        model = AutoAWQForCausalLM.from_pretrained(
            args.base_model,
            quantization_config = quantization_config,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",

        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",attn_implementation="flash_attention_2", 
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with open(args.input_data_file, 'r') as fr:
        data = json.load(fr)
        test_data = data

    test_data_pred = []
    for i in tqdm(range(len(test_data))):
        flag = True

        item = test_data[i]
        new_item = item
        respond = build_generator(args, item, model, tokenizer, temperature=args.temperature, top_p=args.top_p, do_sample=args.do_sample, num_beams = args.num_beams, num_return_sequences = args.num_return_sequences,
                              max_gen_len=args.max_gen_len, use_cache=not args.flash_attn)   # the temperature and top_p are highly different with previous alpaca exp, pay attention to this if there is sth wrong later
        output1 = respond(item)
        pattern = r'```json(.*?)```'
        match = re.search(pattern, output1, re.DOTALL)
        if match:
            output1 = match.group(1).strip()
        else:
            output1 = output1
        if not output1.endswith('}'):
            if not output1.endswith(']'):
                if output1.endswith(', "'):
                    output1 = output1[:-2]
                elif output1.endswith(','):
                    output1 = output1[:-1]
                else:
                    output1 += '"'
                output1 += ']'
        try:
            res11 = []
            if not re.search(r'}\s*$', output1):
                output1 += '}'
            res11 = json.loads(output1)
            print(res11)
        except:
            output = re.sub(r"(?<!\\)'(.*?)(?<!\\)'", r'"\1"', output1)
            output = re.sub(r',\s*\]', ']', output)
            try:
                res11 = json.loads(output)
            except: 
                log_info = "line {} is broken".format(i)
                print(log_info, output1)
                res11 = {
                    'present keyphrases': ['hello world'],
                    'absent keyphrases': ['hello world'],                    
                    }
                flag = False
        if flag:
            new_item["predict"] = res11
            test_data_pred.append(new_item)        

    with open(args.output_data_file, "w") as f:
        json.dump(test_data_pred, f, indent = 2)


if __name__ == "__main__":
    args = parse_config()
    model_map = {
        "qwen": "Qwen/Qwen2.5-14B-Instruct",
        "llama": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
    }
    # stack, duc, kptimes, openkp, kpbiomed
    args.llm_name = 'qwen'
    args.base_model = model_map[args.llm_name]
    args.datasets = 'kpbiomed'
    args.do_sample = False
    args.context_size = 4096
    args.max_gen_len = 256
    args.flash_attn = False
    input_file_dir = os.path.join(os.getcwd(), "test_datasets", args.datasets )
    args.input_data_file = '{}/test_clean.json'.format(input_file_dir)
    output_file_dir = os.path.join(os.getcwd(), "output", args.datasets, args.llm_name)
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)    
    args.output_data_file = "{}/pred.json".format(output_file_dir)
    main(args)

