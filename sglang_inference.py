import json
import os
import re
import unicodedata
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from openai import OpenAI
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from reward import compute_score,extract_solution
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=str)
    parser.add_argument('--modelname', type=str)
    args = parser.parse_args()
    return args

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

'''
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
'''

async_client = AsyncOpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key
)

class EvalDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_jsonl(data_path)
        
    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if "question" in item and "answer" in item:
                        data.append(item)
                except json.JSONDecodeError as e:
                    print(e)
       # data = data[:8]
        return data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


async def send_async_request(question, model_path,ds_name):
    
    if ds_name == "BBH":
        messages = [
            {"role": "user", "content": "\nLet's think step by step. Output the final answer after \"####\". For multiple-choice question, output only the single letter option (e.g., #### A). "+question} 
        ]
    elif ds_name == "MMLU":
        messages = [
            {"role": "user", "content": "\nLet's think step by step. Output the final answer after \"####\". For multiple-choice question, output only the single letter option (e.g., #### A), DO NOT output any content after the letter option (e.g., #### A. 100). "+question} 
        ]
    else:
        messages = [
            {"role": "user", "content": "\nLet's think step by step. Output the final answer after \"####\""+question}#, do not include any prefix such as 'final answer:'. 
        ]
    
    if "AIME" in ds_name or ds_name=="MATH":
        max_len = 20000
    else:
        max_len = 5000
    
    if "AIME" in ds_name:   
        TEMPERATURE=1
        n_samples=8
    else:
        TEMPERATURE=0
        n_samples=1
    
    response = await async_client.chat.completions.create(
        model=model_path,
        messages=messages,
        max_tokens=max_len,
        temperature=TEMPERATURE,
        timeout = 200000,
        n=n_samples
    )
    return response

async def batch_async_requests(questions, model_path, ds_name,max_concurrency=64):
    semaphore = asyncio.Semaphore(max_concurrency)  
    tasks = []
    
    async def bounded_request(q,pbar):
        async with semaphore:
            result = await asyncio.wait_for(
                    send_async_request(q, model_path,ds_name),
                    timeout=200000  
                )
            pbar.update(1)
            return result 
        
        
    pbar = tqdm_asyncio(total=len(questions))
    
    # 创建所有异步任务
    for q in questions:
        tasks.append(bounded_request(q,pbar))
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    pbar.close()
    return results


def evaluate_model( model_path,dataset,ds_name):    
    dataloader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False
    )
    
    results = []
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(dataloader):
        questions = batch['question']
        ground_truths = batch['answer']
        
        request_results = asyncio.run(batch_async_requests(questions, model_path,ds_name))
        
        for chat_response,question,ground_truth in zip(request_results,questions,ground_truths):
            if "AIME" not in ds_name:
                generated_text = chat_response.choices[0].message.content.strip()
            
                final_answer = extract_solution(generated_text)
                is_correct = bool(compute_score(generated_text,ground_truth))
                if is_correct:
                    total_correct += 1
            else:
                generated_texts = [chat_response.choices[i].message.content.strip() for i in range(len(chat_response.choices))]
                for generated_text in generated_texts:
                    final_answer = extract_solution(generated_text)
                    is_correct = bool(compute_score(generated_text,ground_truth))
                    if is_correct:
                        total_correct +=1
                        break
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": final_answer,
                "is_correct": is_correct,
                "full_text": generated_text
            })
        
            total_samples += 1
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"total questions: {total_samples}, correct: {total_correct}, accuracy: {accuracy:.4f}")
    
    return results, accuracy


def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    args = parse_args()
    epoch=args.epoch
    model_name = args.modelname
    
    eval_model = f"{model_name}_{epoch}"
    save_path = os.path.join("/data/sty/sty-lmf/saves/qwen3-4b/eval_result/", eval_model)
    save_mode = True
    
    #model_path = f"/data/sty/onff/model/{model_name}/epoch_{epoch}"
    model_path = f"/data/sty/sty-lmf/saves/qwen3-4b/${model_name}/${epoch}"
    #save_path = os.path.join("/data/sty/onff/eval_result/", "lmfsft1")
    os.makedirs(save_path, exist_ok=True)
    
    all_datasets = ['gsm8k', 'MATH','AIME24', 'AIME25', 'MMLU', 'BBH']
    #all_datasets = ['gsm8k', 'MATH','AIME24', 'AIME25']
    all_acc = {}
    
    for ds in all_datasets:
        print(f"Dataset: {ds}")
        dataset = EvalDataset(os.path.join("/data/sty/sty-lmf/dataset/test", ds + ".jsonl"))
        
        results, accuracy = evaluate_model(model_path, dataset,ds)
        all_acc[ds] = accuracy
        
        if save_mode:
            save_results(results, os.path.join(save_path, ds + ".jsonl"))
    
    if save_mode:
        with open(os.path.join(save_path, "acc.json"), 'w') as f:
            json.dump(all_acc, f, indent=4)
        


if __name__ == "__main__":
    main()