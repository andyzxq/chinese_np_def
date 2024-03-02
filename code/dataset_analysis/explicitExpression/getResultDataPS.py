import pandas as pd
import os
from sklearn.metrics import classification_report

if __name__ == '__main__':
    #读取test文件获取哪些是隐性哪些是显性两个list
    list_im = []
    list_ex = []
    testdata = pd.read_csv("./PredictResult/test.csv", sep=',', header=None, encoding="utf-8").values.tolist()
    for i in range(len(testdata)):
        if testdata[i][4] == "implict":
            list_im.append(i)
        elif testdata[i][4] == "explict":
            list_ex.append(i)

    filedir = "./PredictResult/Definiteness"
    filenames = os.listdir(filedir)
    for filename in filenames:
        ex_yt = []
        ex_yp = []
        im_yt = []
        im_yp = []
        data = pd.read_csv(filedir + '/' + filename, sep=',', header=None, encoding="utf-8").values.tolist()
        for i in range(1, len(data)):
            if i - 1 in list_im:
                im_yp.append(data[i][0])
                im_yt.append(data[i][1])
            elif i - 1 in list_ex:
                ex_yp.append((data[i][0]))
                ex_yt.append(data[i][1])
        print(filename)
        print("**************************implict******************************")
        print(classification_report(im_yt, im_yp, digits=4))
        print("**************************explict******************************")
        print(classification_report(ex_yt, ex_yp, digits=4))