import pandas as pd
import numpy as np

def normalization_by_row(dataSet):
    array = []
    norRow = []
    for i in range(0, dataSet.shape[0]):
        row = dataSet[i, :]
        array.append([])
        ma = row.max()
        mi = row.min()
        for j in range(0, dataSet.shape[1]):
            if (ma - mi) != 0:
                norRow.append(((row[j] - mi) / (ma - mi)) + 0.1)
            else:
                norRow.append(1.1)
            array[i].append(norRow[j])
        norRow.clear()
    return array


def normalization_by_column(dataSet):
    array = []
    norColumn = []
    for i in range(0, dataSet.shape[1]):
        column = dataSet[:, i]
        array.append([])
        ma = column.max()
        mi = column.min()
        for j in range(0, dataSet.shape[0]):
            if (ma - mi) != 0:
                norColumn.append(((column[j] - mi) / (ma - mi)) + 0.1)
            else:
                norColumn.append(1.1)
            array[i].append(norColumn[j])
    return array


def save_to_file(dataSet,sumi_start,path):

    # dataSet为要保存的数据
    # sumi_start为随机值的起始数据，本实验中为59
    # path为要保存的路径

    sum_index = []
    for j in range(0, dataSet.shape[0]):
        sum_index.append(round((j+sumi_start) * 0.01, 2))
    dataframe = pd.DataFrame(dataSet, columns=['K值为' + str(i + 1) for i in range(dataSet.shape[1])])
    # dataframe.columns = ['K值为'+str(i) for i in range(10)]
    dataframe.insert(0, '随机值', sum_index)

    dataframe.to_csv(path, index=False, encoding='gbk')


def save_sub_to_file(dataSet, sumi_start, path):
    # dataSet为要保存的数据
    # sumi_start为随机值的起始数据，本实验中为59
    # path为要保存的路径

    sum_index = []
    for j in range(0, dataSet.shape[0]):
        sum_index.append(round((j + sumi_start) * 0.01, 2))
    dataframe = pd.DataFrame(dataSet, columns=['K值为' + str(i + 2) for i in range(dataSet.shape[1])])
    # dataframe.columns = ['K值为'+str(i) for i in range(10)]
    dataframe.insert(0, '随机值', sum_index)

    dataframe.to_csv(path, index=False, encoding='gbk')