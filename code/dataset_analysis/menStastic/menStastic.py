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

        if line[2] == "DF":
            df_label = "DF"
        else:
            df_label = "UD"

        if line[1] == "P":
            PS_label = "P"
        else:
            PS_label = "S"

        result.append([word_list, tag_list, df_label, PS_label, np_start_index, np_end_index, wordtag_list])

    return result

if __name__ == '__main__':

    stop_wprds = ['哥们', '姐们', '爷们', '娘们', '我们', '你们', '他们', '她们', '它们']

    # 读取数据
    result = {'P_DF':0, 'P_UD':0, 'S_DF':0, 'S_UD':0}
    resultword = {'P_DF': [], 'P_UD': [], 'S_DF': [], 'S_UD': []}
    Dir = "./UsedData/data_40w_"
    for i in range(5):
        filename = Dir + str(i+1) + ".csv"
        rawdata = pd.read_csv(filename, sep=',', header=None, encoding="utf-8")
        rawdata = rawdata.values.tolist()
        datas = readdata(filename)
        for j in range(len(datas)):
            if j == 0 :
                continue
            np_text = datas[j][0][datas[j][4]: datas[j][5] + 1]
            if len(np_text[-1]) != 0:
                if np_text[-1][-1] == '们' and np_text[-1] not in stop_wprds:
                    result[datas[j][3] + '_' + datas[j][2]] += 1
                    resultword[datas[j][3] + '_' + datas[j][2]].append(" ".join(datas[j][6]))
        print(i)

    print(result)

    for key in resultword.keys():
        data = pd.DataFrame(resultword[key])
        data.to_csv("./Result/" + key + ".csv", header=False, index=False, encoding='utf-8')