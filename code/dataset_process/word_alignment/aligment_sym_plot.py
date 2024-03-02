import  numpy as np
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def alignment_sym(cn2en_dic, en2cn_dic):
    #print(cn2en_dic)
    #print(en2cn_dic)
    """
    1.  首先，找到对齐的词语对，将所有双向对齐加入 Alignment。
    2.  遍历 Alignment，对其邻居进行检测，如果邻居中存在单向对齐的点，并且 该点的两个词中存在一个没有与任何词双向对齐，则将该点加入 Alignment。
        循环这个操作直到没有新节点加入。
    3.  最后遍历矩阵中所有节点，如果存在单向对齐的点，并且该点的两个词中存在一个没有与任何词双向对齐，则将该点也加入 Alignment。
    """
    #生成初始矩阵
    result_matrix = np.zeros([len(cn2en_dic.keys()), len(en2cn_dic.keys())])
    #将单向对其的词放入矩阵
    for key in cn2en_dic:
        if cn2en_dic[key] != "":
            enKey_list = cn2en_dic[key].split(" ")
            for enKey in enKey_list:
                result_matrix[int(key)][int(enKey)] = 1
    for key in en2cn_dic:
        if en2cn_dic[key] != "":
            cnKey_list = en2cn_dic[key].split(" ")
            for cnKey in cnKey_list:
                result_matrix[int(cnKey)][int(key)] = 1
    di_alignment = []
    #寻找双对齐的词放入矩阵
    for key in cn2en_dic.keys():
        if cn2en_dic[key] != "":
            enKey_list = cn2en_dic[key].split(" ")
            for enKey in enKey_list:
                if str(key) in en2cn_dic[int(enKey)]:
                    result_matrix[int(key)][int(enKey)] = 2
                    di_alignment.append(str(int(key))+"-"+str(int(enKey)))
    #遍历进行第二步
    while True:
        flag = 0
        for idc in di_alignment:
            idc_x, idc_y = idc.split("-")
            idc_x = int(idc_x)
            idc_y = int(idc_y)
            #判断其上邻居是否可以加入
            if idc_x - 1 >= 0:
                #判断其是否为单向对其
                if result_matrix[idc_x -1][idc_y] == 1:
                    #判断该点的两个词中存在一个没有与任何词双向对齐
                    if 2 not in result_matrix[idc_x -1,:].tolist() or 2 not in result_matrix[:, idc_y].tolist():
                        result_matrix[idc_x -1][idc_y] = 2
                        di_alignment.append(str(idc_x -1) + "-" + str(idc_y))
                        flag = 1
            # 判断其下邻居是否可以加入
            if idc_x + 1 < len(cn2en_dic.keys()):
                #判断其是否为单向对其
                if result_matrix[idc_x + 1][idc_y] == 1:
                    #判断该点的两个词中存在一个没有与任何词双向对齐
                    if 2 not in result_matrix[idc_x + 1,:].tolist() or 2 not in result_matrix[:, idc_y].tolist():
                        result_matrix[idc_x + 1][idc_y] = 2
                        di_alignment.append(str(idc_x + 1) + "-" + str(idc_y))
                        flag = 1
            #判断其左邻居是否可以加入
            if idc_y - 1 >= 0:
                #判断其是否为单向对其
                if result_matrix[idc_x][idc_y - 1] == 1:
                    #判断该点的两个词中存在一个没有与任何词双向对齐
                    if 2 not in result_matrix[idc_x,:].tolist() or 2 not in result_matrix[:, idc_y - 1].tolist():
                        result_matrix[idc_x][idc_y - 1] = 2
                        di_alignment.append(str(idc_x) + "-" + str(idc_y - 1))
                        flag = 1
            # 判断其右邻居是否可以加入
            if idc_y + 1 < len(en2cn_dic.keys()):
                #判断其是否为单向对其
                if result_matrix[idc_x][idc_y + 1] == 1:
                    #判断该点的两个词中存在一个没有与任何词双向对齐
                    if 2 not in result_matrix[idc_x,:].tolist() or 2 not in result_matrix[:, idc_y + 1].tolist():
                        result_matrix[idc_x][idc_y + 1] = 2
                        di_alignment.append(str(idc_x) + "-" + str(idc_y + 1))
                        flag = 1
        if flag == 0:
            break
    #遍历进行第三步
    rows, cols = result_matrix.shape
    for i in range(0, rows):
        for j in range(0, cols):
            if result_matrix[i][j] == 1:
                if 2 not in result_matrix[i, :].tolist() or 2 not in result_matrix[:, j].tolist():
                    result_matrix[i][j] = 2
                    di_alignment.append(str(i) + "-" + str(j))
    return di_alignment

if __name__ == '__main__':
    #读入数据
    cn2en = []
    en2cn = []
    cn_txt = []
    en_txt = []
    file1 = open('wordAssignmentRelativeMaterials\cn2en.A3.final', mode='r', encoding="UTF-8")
    file2 = open('wordAssignmentRelativeMaterials\en2cn.A3.final', mode='r', encoding="UTF-8")
    file3 = open('cn_1.txt', mode='r', encoding="UTF-8")
    file4 = open('en_1.txt', mode='r', encoding="UTF-8")
    try:
        count = 0
        while True:
            dict_cn_line = {}
            dict_en_line = {}
            line = file1.readline()
            line2 = file2.readline()
            if line and line2:
                count = count + 1
                if count % 3 == 0:
                    temp_dict_cn = {}
                    word_count_cn = -1
                    read_flag_cn = 0
                    temp_str_cn = ""
                    temp_list_cn = line.split(" ")
                    #print(temp_list_cn)
                    for item in temp_list_cn:
                        if item == "({":
                            #print("here1")
                            word_count_cn = word_count_cn + 1
                            read_flag_cn = 1
                            continue
                        elif read_flag_cn == 1 and item != "})":
                            temp_str_cn = temp_str_cn + item + " "
                            #print(temp_str_cn)
                            continue
                        elif item == "})":
                            #print("here2")
                            read_flag_cn = 0
                            if temp_str_cn != "":
                                temp_str_cn = temp_str_cn[0:len(temp_str_cn)-1]
                                temp_dict_cn[word_count_cn] = temp_str_cn
                                temp_str_cn = ""
                            elif temp_str_cn == "":
                                temp_dict_cn[word_count_cn] = temp_str_cn
                            continue

                    dict_cn_line = temp_dict_cn
                    #print(dict_cn_line)

                    temp_dict_en = {}
                    word_count_en = -1
                    read_flag_en = 0
                    temp_str_en = ""
                    temp_list_en = line2.split(" ")
                    # print(temp_list_en)
                    for item in temp_list_en:
                        if item == "({":
                            # print("here1")
                            word_count_en = word_count_en + 1
                            read_flag_en = 1
                            continue
                        elif read_flag_en == 1 and item != "})":
                            temp_str_en = temp_str_en + item + " "
                            # print(temp_str_en)
                            continue
                        elif item == "})":
                            # print("here2")
                            read_flag_en = 0
                            if temp_str_en != "":
                                temp_str_en = temp_str_en[0:len(temp_str_en) - 1]
                                temp_dict_en[word_count_en] = temp_str_en
                                temp_str_en = ""
                            elif temp_str_en == "":
                                temp_dict_en[word_count_en] = temp_str_en
                            continue

                    dict_en_line = temp_dict_en
                    #cn2en.append(temp_dict_cn)
                    #en2cn.append(temp_dict_en)

                    line3 = file3.readline()
                    line4 = file4.readline()
                    if line3 and line4:
                        line3 = line3.replace(" \n", "").split(" ")
                        line4 = line4.replace(" \n", "").split(" ")

                    # 对称化
                    align_sym_result = alignment_sym(dict_cn_line, dict_en_line)
                    print(align_sym_result)
                    # 画图
                    align_matrix = np.zeros(shape=(len(dict_cn_line.keys()) - 1, len(dict_en_line.keys()) - 1))
                    for idc in align_sym_result:
                        idc_x, idc_y = idc.split("-")
                        if int(idc_x) != 0 and int(idc_y) != 0:
                            align_matrix[int(idc_x) - 1][int(idc_y) - 1] = 1

                    fig, ax = plt.subplots()

                    plt.imshow(align_matrix, cmap='binary')
                    plt.xticks(np.arange(len(line4)), line4, rotation=90)
                    plt.yticks(np.arange(len(line3)), line3)

                    ax.set_xticks(np.arange(len(line4)) + 0.5, minor=True)  # Grid lines
                    ax.set_yticks(np.arange(len(line3)) + 0.5, minor=True)
                    ax.xaxis.grid(True, which='minor')
                    ax.yaxis.grid(True, which='minor')

                    ax.xaxis.set_ticks_position('top')

                    plt.show()
            else:
                break
    finally:
        file1.close()
        file2.close()
        file3.close()
        file4.close()
    #print(cn2en)
    #print(en2cn)
