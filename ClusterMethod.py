# -*- coding:utf-8 -*-
# carete by steve at  2017 / 03 / 10　20:50


import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib import cm

import DataGenerator


def plotAndSave(CentrePoints, all_data):
    lowb = 0.0
    highb = 5.0
    step = 0.1

    pro_array = np.zeros([int((highb - lowb) / step), int((highb - lowb) / step)])

    for ii in range(50):
        for jj in range(50):
            iii = ii * 0.1
            jjj = jj * 0.1

            # print("i:{0} j:{1} index i:{2} index j:{3}".format(i,j,int((i-lowb)/step),int((j-lowb)/step)))
            current_point = np.zeros(2, dtype=float)
            current_point[1] = iii
            current_point[0] = jjj
            # print(current_point-CentrePoints,"\n\n\n\n\n\n\n\n\n\n")
            # print(int(iii*10),int(jjj*10))
            pro_array[int(iii * 10), int((jjj * 10))] = np.min(
                np.sum((current_point - CentrePoints) ** 2.0, 1) ** 0.5)  #
    pro_array /= np.sum(pro_array)
    print("center points :", CentrePoints)
    # pro_array = (pro_array-0)/np.max(pro_array)*255
    pro_array = np.log((pro_array) + 1.0)

    plt.figure(11 + CentrePoints.shape[0])
    plt.grid(True)

    plt.contourf(pro_array)
    print(pro_array)
    plt.plot(all_data[:, 0] * 10.0, all_data[:, 1] * 10.0, 'b+')
    plt.plot(CentrePoints[:, 0] * 10.0, CentrePoints[:, 1] * 10.0, 'rD')
    plt.savefig("a{0}.jpg".format(CentrePoints.shape[0]))
    print(pro_array.shape)


def KMeans(Data, k=5):
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
    centre_points = np.zeros([k, Data.shape[1]])

    centre_points[:, 0] = np.random.uniform(min(Data[:, 0]), max(Data[:, 0]), k)
    centre_points[:, 1] = np.random.uniform(min(Data[:, 1]), max(Data[:, 1]), k)

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
            all_dis = np.sum((Data[i, :] - centre_points) ** 2.0, 1)
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
            tmp_centre_point[int(labels[i]), :] += Data[i, :]
            centre_counter[int(labels[i])] += 1

        # tmp_centre_point = tmp_centre_point / centre_counter
        for i in range(tmp_centre_point.shape[0]):
            if (centre_counter[i] > 0):

                tmp_centre_point[i, :] = tmp_centre_point[i, :] / float(centre_counter[i])
            else:
                continue
            if np.linalg.norm(tmp_centre_point[i, :] - centre_points[i, :]) > 0.02:
                centre_points[i, :] = tmp_centre_point[i, :]
                centre_changed = True

                # if np.linalg.norm(tmp_centre_point-centre_points) > 0.2:
                #     centre_points = tmp_centre_point
                #     centre_changed = True

    # print("lable:",labels)
    return labels


def KMeansPP(Data, k=5):
    '''

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
    centre_points = np.zeros([k, Data.shape[1]])

    '''
    Select N centre points
    '''
    centre_points[0, :] = Data[int(np.random.uniform(0, Data.shape[0] - 1, 1)), :]
    selected_num = 1

    while selected_num < k:
        plotAndSave(centre_points[:selected_num, :], Data)  # plot probability contour.
        distance_all = np.zeros([Data.shape[0], selected_num])

        for i in range(selected_num):
            distance_all[:, i] = np.sum((Data - centre_points[int(i)]) ** 2.0, 1) ** 0.5

        D = np.min(distance_all, 1)
        D = D / np.sum(D)

        # Select new center points according to probability
        index = 0
        pro = np.random.uniform(0, 1.0, 1)

        pro -= D[0]
        while pro > 0.0:
            index += 1
            pro -= D[index]
        if index < Data.shape[0]:
            centre_points[selected_num, :] = Data[index, :]
            selected_num += 1
        else:
            break

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
            all_dis = np.sum((Data[i, :] - centre_points) ** 2.0, 1)
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
            tmp_centre_point[int(labels[i]), :] += Data[i, :]
            centre_counter[int(labels[i])] += 1

        # tmp_centre_point = tmp_centre_point / centre_counter
        for i in range(tmp_centre_point.shape[0]):
            if (centre_counter[i] > 0):

                tmp_centre_point[i, :] = tmp_centre_point[i, :] / float(centre_counter[i])
            else:
                continue
            if np.linalg.norm(tmp_centre_point[i, :] - centre_points[i, :]) > 0.02:
                centre_points[i, :] = tmp_centre_point[i, :]
                centre_changed = True

                # if np.linalg.norm(tmp_centre_point-centre_points) > 0.2:
                #     centre_points = tmp_centre_point
                #     centre_changed = True

    # print("lable:",labels)
    return labels


def ISODATA(Data, k=5, theta_N=20, theta_S=1.0, theta_c=1.0, L=1, I=10):
    '''
    Iterative Selforganizing Data Analysis Techniques Algorithm
    :param Data:
    :param k:
    :return:
    '''

    # if(Data.shape[0]==0)
    #     return np.zeros()
    '''
    Input Data
    '''

    # initial centre point list
    # centre_point_list = list()

    # for i in range(k):
    #     centre_point_list.append(np.asarray([np.random.uniform(np.min(Data[:,0]),np.max(Data[:,1]),1),
    #                                          np.random.uniform(np.min(Data[:,0],np.max(Data[:,1])))]))
    centre_points = np.zeros([k, 2])

    centre_points[:, 0] = np.random.uniform(np.min(Data[:, 0]), np.max(Data[:, 0]), k)
    centre_points[:, 1] = np.random.uniform(np.min(Data[:, 1]), np.max(Data[:, 1]), k)

    labels = np.zeros(Data.shape[0])

    step_i = 0

    IsChanged = True
    while IsChanged:
        IsChanged = False

        '''
        Compute closet class
        '''

        for i in range(Data.shape[0]):
            all_dis = np.sum((Data[i, :] - centre_points) ** 2.0, 1)
            # print("all dis:",all_dis)
            # print("index :",np.argmin(all_dis))
            labels[i] = np.argmin(all_dis)

        '''
        compute center points and number of member
        '''

        tmp_centre_point = np.zeros_like(centre_points)
        centre_counter = np.zeros(k)
        for i in range(Data.shape[0]):
            # print(Data[i,:])
            # print("label:",labels[i])
            tmp_centre_point[int(labels[i]), :] += Data[i, :]
            centre_counter[int(labels[i])] += 1

        tmp_index = 0
        centre_points = np.zeros([int(sum(centre_counter > theta_N)), 2])
        average_dis = np.zeros(centre_points.shape[0])
        average_counter = np.zeros_like(average_dis)
        for i in range(tmp_centre_point.shape[0]):
            if centre_counter[i] > theta_N:
                index_list = list()
                for k in range(labels.shape[0]):
                    if labels[k] == i:
                        index_list.append(k)

                centre_points[tmp_index, :] = np.mean(Data[index_list, :], 0)
                average_dis[tmp_index] = np.sum((Data[index_list, :] - centre_points[tmp_index]) ** 2, 1) ** 0.5 / len(
                    index_list)
                average_counter[tmp_index] = len(index_list)
                tmp_index += 1

        All_ave_dis = np.mean(average_dis * average_counter)

        step_flag = 0

        if step_i >= I:
            return labels
        elif centre_points.shape[0] <= k / 2:
            # step 8
            print("step 8")
            step_flag = 8

        elif centre_points.shape[0] >= k * 2:
            # ....
            print("step 11")
            step_flag = 11
        else:
            print("step 8")
            step_flag = 8

    return np.zeros(Data.shape[0])


if __name__ == '__main__':
    N = 0
    while N < 100:
        N += 1
    print("begin Test")
    dg = DataGenerator.DataGenerator()

    test_data, test_label = dg.HandleData(1000)
    # plot resource data
    test_fig = plt.figure()
    ax = test_fig.add_subplot(131)
    ax.grid(True)

    for i in range(int(max(test_label) - min(test_label)) + 1):
        index_list = list()
        for k in range(test_label.shape[0]):
            if test_label[k] == i:
                index_list.append(k)
        print(len(index_list))
        ax.plot(test_data[index_list, 0], test_data[index_list, 1], '*', color=cm.jet(i * 70 + 10))

    # Kmeans Result
    ax1 = test_fig.add_subplot(132)
    ax1.grid(True)

    kmeans_lable = KMeans(test_data, 5)

    for i in range(int(max(kmeans_lable) - min(kmeans_lable)) + 1):
        index_list = list()
        for k in range(kmeans_lable.shape[0]):
            if kmeans_lable[k] == i:
                index_list.append(k)

        ax1.plot(test_data[index_list, 0], test_data[index_list, 1], '+', color=cm.jet(i * 70 + 10))

    # ISODATA result
    ax2 = test_fig.add_subplot(133)
    ax2.grid(True)

    iso_label = KMeansPP(test_data, 5)

    for i in range(int(max(iso_label) - min(iso_label) + 1)):
        index_list = list()
        for k in range(iso_label.shape[0]):
            if iso_label[k] == i:
                index_list.append(k)

        ax2.plot(test_data[index_list, 0], test_data[index_list, 1], '+', color=cm.jet(i * 70 + 10))

    test_fig.savefig("all.svg")
    plt.show()
