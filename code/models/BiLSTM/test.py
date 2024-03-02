from dataset import mydata, Dataset
from config import my_config
import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = [i for i in range(1, 6)]
    fig = plt.figure()
    plt.plot(x, [0.7655, 0.7667, 0.7615, 0.7558, 0.7483], marker='o', markersize=4)
    plt.xticks(x)
    plt.savefig("temp.png")
    plt.show()
