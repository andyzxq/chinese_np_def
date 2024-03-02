from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP("stanford-corenlp-4.5.1")
nlp_cn = StanfordCoreNLP("stanford-corenlp-full-2018-01-31", lang="zh")

def match_np(cn_np, en_np, alignment):
    """对英文NP进行遍历
    根据alignment结果，对已识别的中英文NP进行匹配，词对齐结果最多的视为匹配对象(票数一样多则全算)
    当一个英文NP匹配到多个中文, 并且其中一个包含另一个时，匹配最短NP"""

    """对中文NP进行遍历
        根据alignment结果，对已识别的中英文NP进行匹配，词对齐结果最多的视为匹配对象(票数一样多则全算)
        当一个中文NP匹配到多个英文, 并且其中一个包含另一个时，匹配最短NP"""

    """当互选时保存在结果中"""

    en_np_list = en_np.split(" ")
    cn_np_list = cn_np.split(" ")

    # 先对英文NP进行遍历
    # 将中英文np处理成list，将alignment处理成dict(key为英文对应位置，value为中文对应位置）

    align_dict_en2cn = {}
    temp_align = alignment.split(" ")
    for item in temp_align:
        value, key = item.split("-")
        if key not in align_dict_en2cn.keys():
            align_dict_en2cn[key] = value
        else:
            align_dict_en2cn[key] = align_dict_en2cn[key] + " " + value

    #print(align_dict_en2cn)

    result_dict_en = {}
    #进行匹配，key为英文NP位置，value为英文NP所对应的所有中文NP
    for en_np_item in en_np_list:
        vote = {}
        s_en_np, e_en_np = en_np_item.split("-")
        for i in range(0, len(cn_np_list)):
            cn_np_item = cn_np_list[i]
            s_cn_np, e_cn_np = cn_np_item.split("-")
            count = 0
            for j in range(int(s_en_np), int(e_en_np) + 1):
                if str(j) in align_dict_en2cn.keys():
                    temp_cn_value_list = align_dict_en2cn[str(j)].split(" ")
                    for k in range(int(s_cn_np), int(e_cn_np) + 1):
                        if str(k) in temp_cn_value_list:
                            count = count + 1
            vote[i] = count
        #将匹配的np放入词典
        temp = sorted(vote.items(), key=lambda x: x[1], reverse=True)
        #print(temp)
        for i in range(0, len(temp)):
            if temp[i][1] != 0:
                cn_np_index = temp[i][0]
                if en_np_item not in result_dict_en:
                    result_dict_en[en_np_item] = cn_np_list[cn_np_index]
                else:
                    result_dict_en[en_np_item] = result_dict_en[en_np_item] + " " + cn_np_list[cn_np_index]
            if i < len(temp) - 1 and temp[i + 1][1] < temp[i][1]:
                break

    #只保留所选的互相包含里的最短中文NP
    for key in result_dict_en.keys():
        value_list = result_dict_en[key].split(" ")
        temp_d = {}
        temp_value = []
        for item in value_list:
            s, e = item.split("-")
            temp_d[item] = int(e) - int(s) + 1
        temp_dd = sorted(temp_d.items(), key=lambda x: x[1])
        #print(temp_dd)
        for item in temp_dd:
            np = item[0]
            np_s, np_e = np.split("-")
            flag = 0
            for iitem in temp_value:
                ss, ee = iitem.split("-")
                if int(ss) >= int(np_s) and int(ss) <= int(np_e) and int(ee) >= int(np_s) and int(ee) <= int(np_e):
                    flag = 1
                    break
            if flag == 0:
                temp_value.append(np)
        result_dict_en[key] = " ".join(temp_value)

    # 再对中文NP进行遍历
    # 将中英文np处理成list，将alignment处理成dict(key为中文对应位置，value为英文对应位置）

    align_dict_cn2en = {}
    temp_align = alignment.split(" ")
    for item in temp_align:
        key, value = item.split("-")
        if key not in align_dict_cn2en.keys():
            align_dict_cn2en[key] = value
        else:
            align_dict_cn2en[key] = align_dict_cn2en[key] + " " + value

    # print(align_dict_en2cn)

    result_dict_cn = {}
    # 进行匹配，key为中文NP位置，value为中文NP所对应的所有英文NP
    for cn_np_item in cn_np_list:
        vote = {}
        s_cn_np, e_cn_np = cn_np_item.split("-")
        for i in range(0, len(en_np_list)):
            en_np_item = en_np_list[i]
            s_en_np, e_en_np = en_np_item.split("-")
            count = 0
            for j in range(int(s_cn_np), int(e_cn_np) + 1):
                if str(j) in align_dict_cn2en.keys():
                    temp_en_value_list = align_dict_cn2en[str(j)].split(" ")
                    for k in range(int(s_en_np), int(e_en_np) + 1):
                        if str(k) in temp_en_value_list:
                            count = count + 1
            vote[i] = count
        # 将匹配的np放入词典
        temp = sorted(vote.items(), key=lambda x: x[1], reverse=True)
        # print(temp)
        for i in range(0, len(temp)):
            if temp[i][1] != 0:
                en_np_index = temp[i][0]
                if cn_np_item not in result_dict_cn:
                    result_dict_cn[cn_np_item] = en_np_list[en_np_index]
                else:
                    result_dict_cn[cn_np_item] = result_dict_cn[cn_np_item] + " " + en_np_list[en_np_index]
            if i < len(temp) - 1 and temp[i + 1][1] < temp[i][1]:
                break

    #print(result_dict_cn)

    #只保留互不包含的最短英文np
    for key in result_dict_cn.keys():
        value_list = result_dict_cn[key].split(" ")
        temp_d = {}
        temp_value = []
        for item in value_list:
            s, e = item.split("-")
            temp_d[item] = int(e) - int(s) + 1
        temp_dd = sorted(temp_d.items(), key=lambda x: x[1])
        #print(temp_dd)
        for item in temp_dd:
            np = item[0]
            np_s, np_e = np.split("-")
            flag = 0
            for iitem in temp_value:
                ss, ee = iitem.split("-")
                if int(ss) >= int(np_s) and int(ss) <= int(np_e) and int(ee) >= int(np_s) and int(ee) <= int(np_e):
                    flag = 1
                    break
            if flag == 0:
                temp_value.append(np)
        result_dict_cn[key] = " ".join(temp_value)

    #双选的保存在结果清单中
    result_dict = {}
    for key, values in result_dict_en.items():
        value_list = values.split(" ")
        flag = 0
        for value in value_list:
            if key not in result_dict_cn[value]:
                flag = 1
                break
        if flag == 0:
            result_dict[key] = values

    final_dict = {}
    #将结果中的key值中被包含的key删除(关心整体的NP，比如关心that kid 's life， 而不是that kid 's)
    temp_d = {}
    temp_value = []
    for item in result_dict.keys():
        s, e = item.split("-")
        temp_d[item] = int(e) - int(s) + 1
    temp_dd = sorted(temp_d.items(), key=lambda x: x[1], reverse=True)
    for item in temp_dd:
        np = item[0]
        np_s, np_e = np.split("-")
        flag = 0
        for iitem in temp_value:
            ss, ee = iitem.split("-")
            if int(np_s) >= int(ss) and int(np_s) <= int(ee) and int(np_e) >= int(ss) and int(np_e) <= int(ee):
                flag = 1
                break
        if flag == 0:
            temp_value.append(np)
    for key in temp_value:
        final_dict[key] = result_dict[key]

    return final_dict

def if_definAndPluarity(en_str, en_np, cn_str, cn_np):

    """
    先判断是否为人称代词，若是人称代词则返回空值（对人称代词不感兴趣）
    然后寻找判断去见，若有逗号分隔符则分开成不同segment判断结果合并（若不一致则采取默认），然后对一个segment用介词和疑问词再次分割
    找到判定区间后，根据是否有“‘s”再次确认区间
    最后对于单复数信息，看最后一个名词的词性标注（默认复数）
    对于特指信息 ，根据关键词判断（默认泛指）
    (对于人名的辨认没有解决)
    """
    en_np_s, en_np_e = en_np.split("-")
    temp = en_str.split(" ")
    judge_np = " ".join(temp[int(en_np_s) - 1: int(en_np_e)])

    Personal_Pronouns = ["i", "you", "he", "she", "it", "me", "him", "her", "we", "they", "us", "them", "this", "that"]
    Prepositions = ["about", "above", "after", "along", "among", "around", "as", "at", "before", "behind", "below", "beneath",
                    "beside", "besides", "between", "beyond", "by", "down", "during", "except", "excepting", "for", "from",
                    "in", "inside", "into", "near", "of", "off", "on", "onto", "outside", "over", "round", "since", "toward",
                    "under", "up", "upon", "with", "within", "without"]
    W_words = ["what", "where", "when", "who", "why", "how", "whom", "whose", "which"]
    DF_list = ["the", "her", "his", "this", "these", "that", "your", "my", "their", "our"]
    DN_list = ["a", "any", "every"]

    #人称代词返回为空（对人称代词不感兴趣）
    if judge_np in Personal_Pronouns:
        return ""

    #寻找判断区间，若包含逗号，则分为互相独立的不同segment。而后若第一个词为疑问词，则取疑问词之后。若为介词和疑问词在中间，则取前部分。
    result_count = {"P":0, "S":0, "DF":0, "DN":0}   #对于多个并列语句块，投票多数为结果，若一样多采用默认
    segment_list = judge_np.split(",")
    for segment in segment_list:
        pos = nlp.pos_tag(segment)
        word_list = segment.split(" ")
        seg_s = 0
        while(word_list.count("") > 0):
            word_list.pop(word_list.index(""))
        if len(word_list) == 0:
            continue
        seg_end = len(word_list)
        if word_list[0] in W_words:
            seg_s = 1
        #找介词和疑问词之前的部分
        for i in range(seg_s, seg_end):
            if word_list[i] in Prepositions or word_list[i] in W_words or pos[i][1] == "VBG":
                seg_end = i
                break
        #判断单复数信息，根据判断区间内的最后一个名词单复数确认，默认为复数。 由于“’s”也是取后面，所以直接从后开始找就可以
        pors = "P"
        for i in range(0, seg_end - seg_s):
            if pos[seg_end - 1 - i][1] in ["NN", "NNP", "VBN"]:
                pors = "S"
                break
            elif pos[seg_end - 1 - i][0] == "one":
                pors = "S"
                break
            elif pos[seg_end - 1 - i][1] in ["NNS", "NNPS"]:
                pors = "P"
                break
            else:
                continue
        result_count[pors] = result_count[pors] + 1
        #判断特指信息，默认为泛指，若有相应的关键字则为相应的属性
        dfordn = "DN"
        for i in range(seg_s, seg_end):
            if pos[i][0] in DF_list or pos[i][1] == "CD":
                dfordn = "DF"
            elif pos[i][0] in DN_list:
                dfordn = "DN"
            else:
                continue
        result_count[dfordn] = result_count[dfordn] + 1

    #将所有的segment的结果合并，去多数决策，若票数一样多则取默认值
    result = ""
    if result_count["P"] >= result_count["S"]:
        result = result + "P"
    else:
        result = result + "S"
    if result_count["DN"] >= result_count["DF"]:
        result = result + " " + "DN"
    else:
        result = result + " " + "DF"

    return result

if __name__ == '__main__':
    Personal_Pronouns_cn = ["他", "她", "它", "他们", "她们", "它们", "你", "你们", "这", "这个"]
    file1 = open('cn_1_np_8k.txt', mode='r', encoding="UTF-8")
    file2 = open('en_1_np_8k.txt', mode='r', encoding="UTF-8")
    file3 = open('cn_en_align.txt', mode='r', encoding="UTF-8")
    file4 = open('data.txt', mode='w', encoding="UTF-8")
    count = 0
    cn_str = ""
    en_str = ""
    cn_np = ""
    en_np = ""
    alignment = ""
    try:
        while True:
            count = count + 1
            if count % 2 == 1:
                cn_str = file1.readline()
                en_str = file2.readline()
            elif count % 2 == 0:
                cn_np = file1.readline()
                en_np = file2.readline()
                alignment = file3.readline()
            if cn_str and en_str and cn_np != "\n" and en_np != "\n" and alignment != "\n" and count % 2 == 0:
                cn_np = cn_np.replace("\n", "").replace("\r", "")
                en_np = en_np.replace("\n", "").replace("\r", "")
                cn_str = cn_str.replace("\n", "").replace("\r", "")
                en_str = en_str.replace("\n", "").replace("\r", "")
                alignment = alignment.replace("\n", "").replace("\r", "")
                #print(cn_np)
                #print(en_np)
                #print(alignment)
                if count / 2 > 0 and count / 2 <= 200:
                    #print(count / 2)
                    match_result = match_np(cn_np, en_np, alignment)
                    #print(match_result)
                    result_line1 = ""
                    result_line2 = ""
                    for en_np, cn_np in match_result.items():
                        r_if_definAndPluarity = if_definAndPluarity(en_str, en_np, cn_str, cn_np)
                        en_np_s, en_np_e = en_np.split("-")
                        temp = en_str.split(" ")
                        en_np_str = " ".join(temp[int(en_np_s) - 1: int(en_np_e)])
                        cn_np_list = cn_np.split(" ")
                        temp_cn = cn_str.split(" ")
                        if r_if_definAndPluarity != "":
                            for item in cn_np_list:
                                cn_np_s, cn_np_e = item.split("-")
                                cnnp = " ".join(temp_cn[int(cn_np_s) - 1: int(cn_np_e)])
                                if cnnp in Personal_Pronouns_cn:
                                    continue
                                if result_line1 == "":
                                    result_line1 = cn_np + ":" + r_if_definAndPluarity
                                else:
                                    result_line1 = result_line1 + "," + cn_np + ":" + r_if_definAndPluarity
                                if result_line2 == "":
                                    result_line2 = en_np_str +":" + cnnp + ":" + r_if_definAndPluarity
                                else:
                                    result_line2 = result_line2 + "," + en_np_str + ":" + cnnp + ":" + r_if_definAndPluarity
                    #写入数据
                    if result_line1 != "" and result_line2 != "":
                        print(int(count / 2))
                        print(result_line2)
                        file4.write(en_str + "\n" + cn_str + "\n" + result_line1 + "\n" + result_line2 + "\n" + "\n")
                elif count / 2 <= 0:
                    continue
                else:
                    break
    finally:
        file1.close()
        file2.close()
        file3.close()
        file4.close()
        nlp.close()
