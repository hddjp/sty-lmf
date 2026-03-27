import re


def extract_solution(solution_str):

    solutions = solution_str.split('####')
    if len(solutions)==1:
        return None
    else:    
        return solutions[-1].replace("$", "")



def compute_score(solution_str, ground_truth,data_source=None,extra_info=None, format_score=0.0, score=1.0):

   # with open("/data/sty/onff/testfile/reward_log.txt","a") as f:
   #     f.write({"solution_str":solution_str,"ground_truth":ground_truth})
    answer = extract_solution(solution_str=solution_str)

    processed_ground_truth = process_answer(ground_truth)
    
    processed_answer = process_answer(answer)
    
    if processed_answer is None:
        return 0.0
    else:
        if processed_answer==processed_ground_truth:
            return score
        else:
            return format_score


def process_answer(answer):
    if answer is None:
        return None
    answer_str = str(answer).strip()
    
    return strip_string(answer_str)
    
    

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]

    left = "\\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    return s


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def fix_fracs(string):
    try:
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        return new_str
    except Exception:
        with open("/data/sty/onff/testfile/log.txt","a") as f:
            f.write(string)
        return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def remove_right_units(string):
    '''
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    '''
    pattern = r'\\text\{(.*?)\}'
    return re.sub(pattern, r'\1', string)


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string

    try:
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
    except Exception as e:
        with open("/data/sty/onff/testfile/log.txt","a") as f:
            f.write(string)
        return string
    return new_string


def strip_string(string):
    if not string:
        return ""
#    string = string.strip('$')    
    string = remove_boxed(string)
    
    string = string.replace("\n", "")

    string = string.replace("\\!", "")

    string = string.replace("\\\\", "\\")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")

    string = remove_right_units(string)

    string = string.replace("\\\\%", "")
    string = string.replace("\\%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) > 0 and string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = fix_sqrt(string)

    string = string.replace(" ", "")

    string = fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = fix_a_slash_b(string)

    if re.fullmatch(r'[\d,]+', string):
        string = string.replace(',', '')
    if not re.fullmatch(r'\d+', string):
        string = string.lower()

    string = re.sub(r'\.00$', '', string)
        
    return string