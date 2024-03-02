import pandas as pd
import numpy as np
from mlxtend.evaluate import mcnemar

result_1 = pd.read_csv("./predictResult/bert_sp_stability_rs83.csv").values.tolist()
result_2 = pd.read_csv("./predictResult/bert_df_stability_rs65.csv").values.tolist()

r1r2 = 0
w1r2 = 0
r1w2 = 0
w1w2 = 0

for i in range(len(result_1)):
    truel = result_1[i][0]
    p1 = result_1[i][1]
    p2 = result_2[i][1]

    if (p1 == truel) and (p2 == truel):
        r1r2 += 1
    elif (p1 != truel) and (p2 == truel):
        w1r2 += 1
    elif (p1 == truel) and (p2 != truel):
        r1w2 += 1
    else:
        w1w2 += 1

tb_b = np.array([[r1r2, w1r2],[r1w2, w1w2]])

print(tb_b)

chi2, p = mcnemar(ary=tb_b, corrected=True)
print('chi-squared:', chi2)
print('p-value:', p)