import pandas as pd
import os

def readdata(filepath):
    result = []
    path = os.path.join(filepath)
    df = pd.read_csv(path, sep=',', header=None, encoding="utf-8")
    for index, line in df.iterrows():
        # 数据由6部分组成，文本、单复数、特指泛指、NP起始位置、NP终止位置、句子词的词性
        word_list = []
        tag_list = []
        np_start_index = -1
        np_end_index = -1

        wordtag_list = line[0].split(" ")
        flag = 0
        iindex = -1
        for i in range(len(wordtag_list)):
            if "/" in wordtag_list[i]:
                temp = wordtag_list[i].split("/")
                word_list.append(temp[0])
                tag_list.append(temp[1])
                iindex = iindex + 1
            elif wordtag_list[i] == "*" and flag == 0:
                np_start_index = iindex + 1
                flag = 1
                continue
            elif wordtag_list[i] == "*" and flag == 1:
                np_end_index = iindex
                continue

        if line[1] == "DF":
            df_label = "DF"
        else:
            df_label = "UD"

        result.append([word_list, tag_list, df_label, np_start_index, np_end_index])

    return result

if __name__ == '__main__':

    Personal_Pronouns_de = ['他的', '他们的', '她的', '她们的', '它的', '它们的', '你的', '你们的', '我的', '我们的']
    Personal_Pronouns = ['他', '他们', '她', '她们', '它', '它们', '你', '你们', '我', '我们']
    noname_list = ['这', '那']
    noname_list2 = ['这个', '那个', '这些', '那些']

    # 读取数据
    total_count = 0
    explict_count = 0
    # Dir = "./resultData/data_40w_"
    Dir = "./resultData/test"
    result = []
    for i in range(1):
        filename = Dir + ".csv"
        rawdata = pd.read_csv(filename, sep=',', header=None, encoding="utf-8")
        rawdata = rawdata.values.tolist()
        result.append(rawdata[0] + ["explict or implict for DF UD"])
        datas = readdata(filename)
        for j in range(len(datas)):
            if j == 0 :
                continue
            total_count += 1
            np_text = datas[j][0][datas[j][3] : datas[j][4] + 1]
            np_pos = datas[j][1][datas[j][3] : datas[j][4] + 1]
            if 'NNP' in np_pos or 'NNPS' in np_pos or 'NR' in np_pos:
                explict_count += 1
                result.append(rawdata[j] + ["explict"])
            elif 'DT' in np_pos:
                explict_count += 1
                result.append(rawdata[j] + ["explict"])
            else:
                if [char for char in Personal_Pronouns_de] in np_text:
                    explict_count += 1
                    result.append(rawdata[j] + ["explict"])
                elif '的' in np_text:
                    index = np_text.index('的') - 1
                    if index >= 0 and np_text[index] in Personal_Pronouns:
                        explict_count += 1
                        result.append(rawdata[j] + ["explict"])
                    else:
                        result.append(rawdata[j] + ["implict"])
                elif [char for char in noname_list2] in np_text:
                    explict_count += 1
                    result.append(rawdata[j] + ["implict"])
                elif 'M' in np_pos:
                    index1 = np_pos.index('M') - 1
                    index2 = np_pos.index('M') - 2
                    if (index1 >= 0 and np_text[index1] in noname_list) or (index2 >= 0 and np_text[index2] in noname_list):
                        explict_count += 1
                        result.append(rawdata[j] + ["explict"])
                    else:
                        result.append(rawdata[j] + ["implict"])
                elif 'CD' in np_pos:
                    index = np_pos.index('CD') - 1
                    if index >= 0 and np_text[index] in noname_list:
                        explict_count += 1
                        result.append(rawdata[j] + ["explict"])
                    else:
                        result.append(rawdata[j] + ["implict"])
                else:
                    result.append(rawdata[j] + ["implict"])
                    continue
        result = pd.DataFrame(result)
        result.to_csv(filename.replace("UsedData", "resultData"), index=False, header=False, encoding="utf-8")
        result = []
        print(i)

    print(total_count)
    print(explict_count)
    print((explict_count/total_count))