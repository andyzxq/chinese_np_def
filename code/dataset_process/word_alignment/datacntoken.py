from stanfordcorenlp import StanfordCoreNLP

nlp_cn = StanfordCoreNLP("stanford-corenlp-full-2018-01-31", lang="zh")

def data_preprocess():
    file_object1 = open("cn_1.txt", 'r', encoding="UTF-8")
    file1 = open('cn_1_lyq.txt', mode='w', encoding="UTF-8")
    count = 0
    try:
        while True:
            count = count + 1
            line_cn = file_object1.readline()
            count_quot = 0
            if line_cn:
                line_cn = line_cn.replace("\r", "").replace("\n", "").replace(" ", "")

                #用stanford做中文分词
                line_cn = " ".join(nlp_cn.word_tokenize(line_cn))
                file1.write(line_cn + "\n")
                print(count)
            else:
                nlp_cn.close()
                break
    finally:
        file_object1.close()
        file1.close()

if __name__ == '__main__':
    data_preprocess()