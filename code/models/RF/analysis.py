# 导入必要的库
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 绘图库
from dataprocess import Textprocess
import pickle
from xpinyin import Pinyin

# 解决画图中文字体显示的问题
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']  # 汉字字体集
plt.rcParams['font.size'] = 10  # 字体大小
plt.rcParams['axes.unicode_minus'] = False
# 忽略警告
import warnings

warnings.filterwarnings('ignore')

#获得数据
T = Textprocess()
tr_x, tr_y, te_x, te_y = T.getData()
rf_model = RandomForestClassifier(n_estimators=1000, max_depth=500, min_samples_split=20, min_samples_leaf=1, max_features="sqrt")

words = T.vectorizer.get_feature_names()
print(len(words))
newwords = []
p = Pinyin()
# 给汉子加拼音
for word in words:
    wl = word.split(" ")
    temp = []
    for w in wl:
        if w != "np":
            wpy = p.get_pinyin(w, tone_marks='marks').replace("-", " ")
            temp.append(w +" (" + wpy + ")")
        else:
            temp.append("np")
    newwords.append(" ".join(temp))
print("finish pinyin")

input = open('./model/ps_rf_n13.pkl', 'rb')
rf_model = pickle.load(input)
input.close()
# 获取特征重要性得分
feature_importances = rf_model.feature_importances_
# 创建特征名列表
feature_names = list(newwords)
# 创建一个DataFrame，包含特征名和其重要性得分
feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
# 对特征重要性得分进行排序
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
feature_importances_df = feature_importances_df[0:30]
print(feature_importances_df[0:30])

# 颜色映射
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importances_df)))

# 可视化特征重要性
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importances_df['feature'], feature_importances_df['importance'], color=colors)
ax.invert_yaxis()  # 翻转y轴，使得最大的特征在最上面
ax.set_xlabel('The importance of features', fontsize=12)  # 图形的x标签
ax.set_title('Random forest feature importance visualization', fontsize=16)
for i, v in enumerate(feature_importances_df['importance']):
    ax.text(v + 0.01, i, str(round(v, 3)), va='center', fontname='Times New Roman', fontsize=10)

# # 设置图形样式
# plt.style.use('default')
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框
# ax.spines['left'].set_linewidth(0.5)#左边框粗细
# ax.spines['bottom'].set_linewidth(0.5)#下边框粗细
# ax.tick_params(width=0.5)
# ax.set_facecolor('white')#背景色为白色
# ax.grid(False)#关闭内部网格线

# 保存图形
# plt.savefig('./特征重要性.jpg', dpi=400, bbox_inches='tight')
plt.show()
