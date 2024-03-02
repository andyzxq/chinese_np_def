from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP("stanford-corenlp-4.5.1")
nlp_cn = StanfordCoreNLP("stanford-corenlp-4.5.1", lang="zh")

def findAllNP(parse_str):
    parse_list = parse_str.split("\r\n")
    parentheses_match_count = 0
    np_list = []
    word_count = 1
    result_dict = {}
    np_count = 0
    for parse_line in parse_list:
        while(True):
            #把每一行字符前面的空格去掉
            if parse_line[0] == " ":
                parse_line = parse_line[1:]
            else:
                break
        parse_line_list = parse_line.split(" ")
        for parse_line_item in parse_line_list:
            if (parse_line_item[0] == "(") and (parse_line_item[1:3] != "NP") and (len(np_list) == 0):
                continue
            elif parse_line_item[0] == "(" and parse_line_item[1:3] != "NP" and len(np_list) != 0:
                parentheses_match_count = parentheses_match_count + 1
            elif parse_line_item[0] == "(" and parse_line_item[1:3] == "NP":
                np_list.append(str(parentheses_match_count) + "_" + str(np_count))
                result_dict[str(parentheses_match_count) + "_" + str(np_count)] = str(word_count) + "-"
                parentheses_match_count = parentheses_match_count + 1
                np_count = np_count + 1
            elif parse_line_item[-1] == ")":
                np_match = -1
                if len(np_list) != 0:
                    np_match = int(np_list[-1].split("_")[0])
                while(True):
                    if parse_line_item[-1] == ")" and len(np_list) != 0:
                        parse_line_item = parse_line_item[0:len(parse_line_item)-1]
                        parentheses_match_count = parentheses_match_count - 1
                        if parentheses_match_count == np_match:
                            result_dict[np_list[-1]] = result_dict[np_list[-1]] + str(word_count)
                            np_list.pop()
                    else:
                        break
                word_count = word_count + 1
    return result_dict


if __name__ == '__main__':
    """file2 = open('en_1.txt', mode='r', encoding="UTF-8")
    file4 = open('en_1_np.txt', mode='w', encoding="UTF-8")
    count = 0
    try:
        while True:
            dict_cn_line = {}
            dict_en_line = {}
            line = file2.readline()
            count = count + 1
            if line:
                if  count > 200000 and count <= 400000:
                    result = findAllNP(nlp.parse(line))
                    file4.write(line + " ".join(result.values()) + "\n")
                    print(count)
            else:
                break
    finally:
        file2.close()
        file4.close()
        nlp.close()"""

    file1 = open('cn_1.txt', mode='r', encoding="UTF-8")
    file3 = open('cn_1_np.txt', mode='w', encoding="UTF-8")
    count = 0
    try:
        while True:
            dict_cn_line = {}
            dict_en_line = {}
            line = file1.readline()
            count = count + 1
            if line:
                if count <= 200000:
                    result = findAllNP(nlp_cn.parse(line))
                    file3.write(line + " ".join(result.values()) + "\n")
                    print(count)
            else:
                break
    finally:
        file1.close()
        file3.close()
        nlp_cn.close()