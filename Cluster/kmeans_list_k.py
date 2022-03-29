from __future__ import division, print_function, absolute_import
import math
import warnings
from turtle import distance

import numpy as np
import pandas as pd
from collections import deque
from scipy._lib._util import _asarray_validated
# from scipy._lib.six import xrange
from pylab import *
from numpy import *
from scipy.cluster.vq import vq, whiten, _vq
from utils import save_to_file

# 加载数据
# def loadDataSet(fileName):
#     data = np.loadtxt(fileName,delimiter=',')
#     data = data[:, 1:12]
#     # cancer_label = dataset[:,10]
#     #data = np.loadtxt(fileName, delimiter='\t', dtype=float, skiprows=1)
#     return data
#     #data =  np.loadtxt(fileName)
from sqlalchemy import all_


def myKmeans(obs, guess, thresh=1e-5):
    code_book = np.asarray(guess)
    diff = np.inf
    prev_avg_dists = deque([diff], maxlen=2)
    while diff > thresh:
        # compute membership and distances between obs and code_book
        obs_code, distort = vq(obs, code_book, check_finite=False)
        prev_avg_dists.append(distort.mean(axis=-1))
        # recalc code_book as centroids of associated obs
        code_book, has_members = _vq.update_cluster_means(obs, obs_code,
                                                          code_book.shape[0])
        code_book = code_book[has_members]
        diff = prev_avg_dists[0] - prev_avg_dists[1]

    return code_book, prev_avg_dists[1]


def kmeans(obs, k, sumi, iter=20, thresh=1e-5):
    '''
    obs
    M×N阵列的每一行都是观察向量。 列是每次观察期间看到的特征。 必须先使用whiten将特征增白。
    k_or_guess
    生成的质心数。 将代码分配给每个质心，这也是质心在生成的code_book矩阵中的行索引。
    通过从观察矩阵中随机选择观察值来选择初始k重心。 可替代地，将k乘以N数组指定初始的k个质心。
    iter
    运行k均值的次数，返回具有最低失真的代码本。 如果为k_or_guess参数的数组指定了初始质心，则将忽略此参数。 此参数不代表k均值算法的迭代次数。
    thresh
    如果自上次k均值迭代以来失真的变化小于或等于阈值，则终止k均值算法。
    check_finite
    是否检查输入矩阵仅包含有限数。 禁用可能会提高性能，但是如果输入中确实包含无穷大或NaN，则可能会导致问题（崩溃，终止）。 默认值：True
    '''

    # initialize best distance value to a large value
    best_dist = np.inf
    for i in range(iter):
        # the initial code book is randomly selected from observations
        guess = kpoints(obs, k,sumi)
        book, dist = myKmeans(obs, guess, thresh=thresh)
        if dist < best_dist:
            best_book = book
            best_dist = dist
    '''
    codebook
    由k个质心组成的k x N数组。 第i个质心代码簿[i]用代码i表示。 生成的质心和代码表示所看到的最低失真，而未必是全局最小失真。
    distortion
    通过的观测值与生成的质心之间的平均（非平方）欧氏距离。 请注意，在k均值算法的上下文中，失真的标准定义有所不同，即平方距离的总和。
    '''
    return best_book, best_dist


def kpoints(dataSet, k, sumi):
    # 获取样本数与特征值
    m, n = dataSet.shape  # 把数据集的行数和列数赋值给m,n
    # 初始化质心,创建(k,n)个以零填充的矩阵, 每个质心有n个坐标值，总共要k个质心
    cluster_centers = np.mat(np.zeros((k, n)))
    # index = np.random.randint(0, m)
    index = 0  #人民日报  id 2803301701
    # 复制函数，修改cluster_centers，不会影响dataSet
    cluster_centers[0,] = np.copy(dataSet[index,])
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(dataSet[j,], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= sumi
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all = sum_all - di
            if sum_all > 0:
                continue
            cluster_centers[i,] = dataSet[j,]
            break
    return cluster_centers


# 计算欧氏距离
def distance(x1, x2):
    # 求两个向量之间的距离
    return sqrt(sum(power(x1 - x2, 2)))


# 对一个样本找到与该样本距离最近的聚类中心
def nearest(point, cluster_centers):
    min_dist = inf
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i,])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist

if __name__ == '__main__':
    # 待聚类的数据点,user_time_count.csv有195381行数据,每行数据有13维:
    dataset = np.loadtxt('data/compare/all_user/all_user_info.csv', delimiter=",",skiprows=1)
    # np数据从0开始计算，第0维维序号排除，第12维为标签排除，所以为1到12,因为取不到第12列数据
    points = dataset[:, 1:7]

    print("points:\n", points)
    all_sse = []
    # k-means聚类
    # 将原始数据做归一化处理
    # 传入不同的参数（k值和随机值） 得到SSE的值，将SSE存入二维数组
    data = whiten(points)

    for i in range(0,32):
        all_sse.append([])
        for j in range(0,5):
            sum_num = round((i+59) * 0.01, 2)
            centroids = kmeans(data, j+1, sum_num)[0]
            sse = kmeans(data, j+1, sum_num)[1]
            all_sse[i].append(sse)
            print("k = %s, 随机数 = %s, SSE = %s"%(j + 1,sum_num,sse))
    # print("Centroids:\n", centroids)

    # 写入csv
    path = 'result/compare/all_user/SSE.csv'
    all_sse = np.array(all_sse)
    save_to_file(all_sse, 59, path)




