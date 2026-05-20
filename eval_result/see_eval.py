import json

def filter_diff_data(file_a: str, file_b: str, out_path: str):
    # 读取B文件：key为问题文本，存is_correct、generated_answer、full_text
    b_map = {}
    with open(file_b, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = d["question"]
            b_map[q] = {
                "is_correct_b": d["is_correct"],
                "gen_ans_b": d["generated_answer"],
                "full_text_b": d["full_text"]
            }

    res = []
    # 遍历A文件，筛选A正确、B错误
    with open(file_a, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a_d = json.loads(line)
            q = a_d["question"]
            if a_d["is_correct"] is not True:
                continue
            if q not in b_map:
                continue
            if b_map[q]["is_correct_b"] is not False:
                continue

            # 合并字段
            merge_d = {
                "question": q,
                "generated_answer_a": a_d["generated_answer"],
                "full_text_a": a_d["full_text"],
                "generated_answer_b": b_map[q]["gen_ans_b"],
                "full_text_b": b_map[q]["full_text_b"]
            }
            res.append(merge_d)

    # 写入jsonl
    with open(out_path, "w", encoding="utf-8") as f:
        for item in res:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# 使用示例
if __name__ == "__main__":
    filter_diff_data("/data/sty/sty-lmf/eval_result/sft_random/MMLU.jsonl", "/data/sty/sty-lmf/eval_result/distill_temp4/MMLU.jsonl", "/data/sty/sty-lmf/eval_result/diff_ab.jsonl")