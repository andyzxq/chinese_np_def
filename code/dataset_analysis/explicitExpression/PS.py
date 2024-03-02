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

        if line[1] == "S":
            ps_label = "S"
        else:
            ps_label = "P"

        result.append([word_list, tag_list, ps_label, np_start_index, np_end_index])

    return result

if __name__ == '__main__':
    # 读取数据
    total_count = 0
    explict_count = 0
    # Dir = "./UsedData/data_40w_"
    Dir = "./UsedData/test"
    result = []
    for i in range(1):
        filename = Dir + ".csv"
        rawdata = pd.read_csv(filename, sep=',', header=None, encoding="utf-8")
        rawdata = rawdata.values.tolist()
        result.append(rawdata[0] + ["explict or implict for S P"])
        datas = readdata(filename)
        for j in range(len(datas)):
            if j == 0 :
                continue
            total_count += 1
            np_text = datas[j][0][datas[j][3] : datas[j][4] + 1]
            np_pos = datas[j][1][datas[j][3] : datas[j][4] + 1]
            if 'CD' in np_pos or 'M' in np_pos or 'OD' in np_pos:
                explict_count += 1
                result.append(rawdata[j] + ["explict"])
            else:
                result.append(rawdata[j] + ["implict"])
        result = pd.DataFrame(result)
        result.to_csv(filename.replace("UsedData", "resultData"), index=False, header=False, encoding="utf-8")
        result = []
        print(i)

    print(total_count)
    print(explict_count)
    print((explict_count/total_count))
