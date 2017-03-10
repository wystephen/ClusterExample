# -*- coding:utf-8 -*-
# carete by steve at  2017 / 03 / 10　20:50


import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib import cm

import DataGenerator


def KMeans(Data,k=5):
    '''
    KMeans Methond
    :param Data:
    :param k:
    :return:
    '''

    return np.zeros(Data.shape[0])

def ISODATA(Data,k=5):
    '''

    :param Data:
    :param k:
    :return:
    '''




if __name__ == '__main__':
    print("begin Test")
    dg = DataGenerator.DataGenerator()

    test_data ,test_label = dg.HandleData(2000)
    # plot resource data
    test_fig = plt.figure()
    ax = test_fig.add_subplot(131)
    ax.grid(True)

    for i in range(int(max(test_label)-min(test_label))+1):
        index_list = list()
        for k in range(test_label.shape[0]):
            if test_label[k]==i:
                index_list.append(k)
        print(len(index_list))
        ax.plot(test_data[index_list,0],test_data[index_list,1],'*',color=cm.jet(i*70+10))

    # Kmeans Result
    ax1=test_fig.add_subplot(132)
    ax1.grid(True)

    kmeans_lable = KMeans(test_data,5)


    for i in range(int(max(kmeans_lable)-min(kmeans_lable))+1):
        index_list = list()
        for k in range(kmeans_lable.shape[0]):
            if kmeans_lable[k] == i:
                index_list.append(k)

        ax1.plot(test_data[index_list,0],test_data[index_list,1],'+',color=cm.jet(i*70+10))







    plt.show()
