import json

def merge_outputs(input_file, output_file, separator):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        new_item = {
            "instruction": item["instruction"],
            "input": item["input"],
            "output": separator + separator.join(item["outputs"]) + separator
        }
        processed_data.append(new_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    merge_outputs(
        input_file="/data/sty/sty-lmf/data/MATH_4B.jsonl",
        output_file="/data/sty/sty-lmf/data/MATH_all.json",
        separator="######"
    )