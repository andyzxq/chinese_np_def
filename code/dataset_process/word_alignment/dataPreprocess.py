from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP("stanford-corenlp-4.5.1")

def data_preprocess():
    file_object1 = open("cn_1_or.txt", 'r', encoding="UTF-8")
    file_object2 = open("en_1_or.txt", 'r', encoding="UTF-8")
    file1 = open('cn_1.txt', mode='w', encoding="UTF-8")
    file2 = open('en_1.txt', mode='w', encoding="UTF-8")
    line_en_w = ""
    line_cn_w = ""
    mix_flag = 0
    count = 0
    try:
        while True:
            count = count + 1
            line_cn = file_object1.readline()
            line_en = file_object2.readline()
            count_quot = 0
            if line_cn and line_en:
                line_en = line_en.replace("\r", "").replace("\n", "")
                line_cn = line_cn.replace("\r", "").replace("\n", "")
                #将单引号还原
                line_en = line_en.replace("&apos;", "'")
                #将双引号还原
                while(True):
                    quot_index = line_en.find("&quot;")
                    if quot_index != -1 and count_quot == 0:
                        line_en = line_en[:quot_index] + "``" + line_en[quot_index + 6 :]
                        count_quot = 1
                    elif quot_index != -1 and count_quot == 1:
                        line_en = line_en[:quot_index] + "''" + line_en[quot_index + 6 :]
                        count_quot = 0
                    elif quot_index == -1:
                        break
                while (True):
                    quot_index = line_cn.find("&quot;")
                    if quot_index != -1 and count_quot == 0:
                        line_cn = line_cn[:quot_index] + "``" + line_cn[quot_index + 6:]
                        count_quot = 1
                    elif quot_index != -1 and count_quot == 1:
                        line_cn = line_cn[:quot_index] + "''" + line_cn[quot_index + 6:]
                        count_quot = 0
                    elif quot_index == -1:
                        break
                #用stanford做英文分词
                line_en = " ".join(nlp.word_tokenize(line_en))
                #合并句子
                if line_en[-1] == ",":
                    if line_en_w == "":
                        line_en_w = line_en
                        line_cn_w = line_cn
                    else:
                        line_en_w = line_en_w + " " + line_en
                        line_cn_w = line_cn_w + " " + line_cn
                    mix_flag = 1
                elif line_en[-1] != "," and mix_flag == 0:
                    file1.write(line_cn + "\n")
                    file2.write(line_en + "\n")
                elif line_en[-1] != "," and mix_flag == 1:
                    line_en_w = line_en_w + " " + line_en
                    line_cn_w = line_cn_w + " " + line_cn
                    file1.write(line_cn_w + "\n")
                    file2.write(line_en_w + "\n")
                    mix_flag = 0
                    line_en_w = ""
                    line_cn_w = ""
                print(count)
            else:
                nlp.close()
                break
    finally:
        file_object1.close()
        file_object2.close()
        file1.close()
        file2.close()


if __name__ == '__main__':
    data_preprocess()


