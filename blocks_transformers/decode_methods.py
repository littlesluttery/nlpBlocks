import math
import torch
import numpy as np
from typing import List, Iterable, Dict, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams["font.sans-serif"]=["FangSong"]  # 指定默认字体 SimHei为黑体 FangSong仿宋
mpl.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

input_ids = torch.tensor([[0,1,2]])  # 输入
scores = torch.tensor([[2.1,-1,-10,1,0]]) # 概率值
scores.shape     # 词表大小

def show_scores(scores1,scores2):
    bar_with = 0.35
    x = torch.arange(len(scores1[0]))
    soft_max_scores1 = torch.nn.functional.softmax(scores1,dim=-1)
    soft_max_scores2 = torch.nn.functional.softmax(scores2,dim=-1)
    plt.bar(x,soft_max_scores1[0],bar_with,align="center",color="#66c2a5",label="处理前")
    plt.bar(x+bar_with,soft_max_scores2[0],bar_with,align="center",color="#8da0cb",label="处理后")
    plt.legend()
    plt.show()

show_scores(scores,scores)

