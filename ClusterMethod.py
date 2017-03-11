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

    if Data.shape[0] == 0:
        return np.zeros(Data.shape[0])

    labels = np.zeros(Data.shape[0])

    '''
    Initial k center points
    '''
    centre_points = np.zeros([k,Data.shape[1]])

    centre_points[:,0] = np.random.uniform(min(Data[:,0]),max(Data[:,0]),k)
    centre_points[:,1] = np.random.uniform(min(Data[:,1]),max(Data[:,1]),k)


    '''
    cluster
    '''

    centre_changed = True

    while centre_changed:
        centre_changed = False

        '''
        1. computer centre point
        '''
        for i in range(Data.shape[0]):
            all_dis = np.sum((Data[i,:]-centre_points)**2.0,1)
            # print("all dis:",all_dis)
            # print("index :",np.argmin(all_dis))
            labels[i] = np.argmin(all_dis)


        '''
        2. update centre point
        '''
        tmp_centre_point = np.zeros_like(centre_points)
        centre_counter = np.zeros(k)
        for i in range(Data.shape[0]):
            # print(Data[i,:])
            # print("label:",labels[i])
            tmp_centre_point[int(labels[i]),:] +=Data[i,:]
            centre_counter[int(labels[i])] += 1

        # tmp_centre_point = tmp_centre_point / centre_counter
        for i in range(tmp_centre_point.shape[0]):
            if(centre_counter[i]>0):

                tmp_centre_point[i,:] = tmp_centre_point[i,:] / float(centre_counter[i])
            else:
                continue
            if np.linalg.norm(tmp_centre_point[i,:]-centre_points[i,:]) > 0.1:
                centre_points[i,:] = tmp_centre_point[i,:]
                centre_changed=True

        # if np.linalg.norm(tmp_centre_point-centre_points) > 0.2:
        #     centre_points = tmp_centre_point
        #     centre_changed = True



    print("lable:",labels)
    return labels








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
