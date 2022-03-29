#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :  normalization_all.py    
@Project      :  data.mat
@Contact      :  QuinneCJ@gmail.com
@Modify Time  :  2022/2/8 18:09 
@Author       :  ZCJ 
@Version    
@Desciption                 
'''
import math
import pandas as pd
import numpy as np
from utils import normalization_by_row,save_to_file,save_sub_to_file
# media_government all_user media_celebrity  government_celebrity
dataset = np.loadtxt('result/compare/all_user/SSE.csv', delimiter=",",skiprows=1)

# np数据从0开始计算，第0维维序号排除，第12维为标签排除，所以为1到12,因为取不到第12列数据
sse = dataset[:, 1:7]
print('******** sse **********')
print(sse)
print(sse.shape[0])

# 对所有的SSE进行归一化
all_sse_n = normalization_by_row(sse)
print('******** sse **********')
print(sse)
all_sse_n = np.array(all_sse_n)
print('******** all_sse_n **********')
print(all_sse_n)
print(all_sse_n.shape[0])


# 写入csv
path1 = 'result/compare/all_user/all_sse_n.csv'
save_to_file(all_sse_n,59,path1)

# 计算SSE的差值将所有的差值存入二维数组
sse_sub = []
# for i in range(1,sse.shape[1]):
#     sse_sub.append([])
#     for j in range(0,sse.shape[0]):
#         result =  sse[j][i] - sse[j][i-1]
#         sse_sub[i-1].append(result)


for i in range(0,sse.shape[0]):
    sse_sub.append([])
    for j in range(1,sse.shape[1]):
        result =  sse[i][j-1] - sse[i][j]
        sse_sub[i].append(result)

sse_sub = np.array(sse_sub)
print('******** sse_sub **********')
print(sse_sub)
print(sse_sub.shape[0])
# 写入csv
path2 = 'result/compare/all_user/sse_sub.csv'
save_sub_to_file(sse_sub,59,path2)


# 对SSE的差值进行归一化
sse_sub_n = normalization_by_row(sse_sub)
sse_sub_n = np.array(sse_sub_n)
# 写入csv
path3 = 'result/compare/all_user/sse_sub_n.csv'
save_sub_to_file(sse_sub_n,59,path3)
print('******* sse_sub_n ***********')
print(sse_sub_n)
print(sse_sub.shape[0])

# 通过计算确定最优值
result = []
for i in range(0,sse_sub.shape[0]):
    result.append([])
    for j in range(0,sse_sub.shape[1]):
        # res = sum_num_n * (sse_sub_n[i][j-1]/all_sse_n[6-i][j-1])   # 值越大越好
        # res = (sse_sub[i][j] + (sse[i][j] +sse[i][j+1])/2)/ abs(math.log(math.log(j+2,1.5)* sse[i][j+1],5)) # 值越大越好
        # res = (sse_sub[i][j] + (sse[i][j] +sse[i][j+1])/2)/  sse[i][j+1]
        k = (j + 2) * 0.3
        res = math.log(sse_sub_n[i][j] * 100,2) /abs(math.log(k,2) * math.log(all_sse_n[i][j+1] * 100,2)) # 值越大越好
        print(abs(math.log(sse_sub_n[i][j] * 10,5)))
        print(abs(math.log(math.log(j+2,2.5) * all_sse_n[i][j+1],2)))
        # res2 = math.pow(res,2)
        result[i].append(res)
        print("k = %s, 随机数 = %s, res = %s"%(j+2,i+59,res))

result = np.array(result)
print('******* result ***********')
print(result)
print(result.shape[0])

# 写入csv
path4 = 'result/compare/all_user/result_final.csv'
save_sub_to_file(result,59,path4)