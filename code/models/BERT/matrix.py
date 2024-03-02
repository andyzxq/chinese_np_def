from sklearn.metrics import confusion_matrix

def drawMarrix(y_true, y_pred):

    C = confusion_matrix(y_true, y_pred, labels=['SU', 'SD', 'PU', 'PD'])  # 可将'1'等替换成自己的类别，如'cat'。

    plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    # plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
    # plt.yticks(range(0,5), labels=['a','b','c','d','e'])
    plt.savefig("robert_large_44.png")

if __name__ == '__main__':

    df = pd.read_csv('df.csv', sep=',', encoding="utf-8")
    ps = pd.read_csv('ps.csv', sep=',', encoding="utf-8")

    df = df.values.tolist()
    ps = ps.values.tolist()

    y_t = []
    y_p = []

    for i in range(len(df)):
        if ps[i][0] == 0 and df[i][0] == 0 :
            y_t.append(0)
        elif ps[i][0] == 0 and df[i][0] == 1 :
            y_t.append(1)
        elif ps[i][0] == 1 and df[i][0] == 0 :
            y_t.append(2)
        elif ps[i][0] == 1 and df[i][0] == 1 :
            y_t.append(3)

        if ps[i][1] == 0 and df[i][1] == 0 :
            y_p.append(0)
        elif ps[i][1] == 0 and df[i][1] == 1 :
            y_p.append(1)
        elif ps[i][1] == 1 and df[i][1] == 0 :
            y_p.append(2)
        elif ps[i][1] == 1 and df[i][1] == 1 :
            y_p.append(3)

    drawMarrix(y_t, y_p)