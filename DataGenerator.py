# -*- coding:utf-8 -*-
# carete by steve at  2017 / 03 / 10　20:05

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

class DataGenerator:
    '''

    '''
    def __init__(self,k=3,sigma=1.0,D=2):
        '''


        '''

        # self.k=3
        # self.sigma=1.0
        # self.D=2
        # print("initial ok")

    def NormalData(self,Data_number=100,k=3,offset=1.0,sigma=1.0,D=2):
        # Find centre points
        centre_point = np.zeros([k,D])
        for i in range(centre_point.shape[0]):
            print("i")


        # General DataSet

    def HandleData(self,numbers=1000,sigma = 0.25):
        DataSet=np.zeros([numbers,2])
        DataLabel=np.zeros([numbers,1])


        centre_points=np.zeros([5,2])
        centre_points[0,:]=[2.0,1.0]
        centre_points[1,:]=[1.0,2.0]
        centre_points[2,:]=[2.0,3.0]
        centre_points[3,:]=[3.3,3.0]
        centre_points[4,:]=[4.0,2.0]

        step_length=int(numbers/centre_points.shape[0])
        # print("step lenght:",step_length)



        for i in range(centre_points.shape[0]):
            # print("i:",i)
            DataSet[(step_length*i):(step_length*(i+1)),0]=np.random.normal(centre_points[i,0],sigma,DataSet[step_length*i:(step_length*(i+1)),0].shape)
            DataSet[(step_length*i):step_length*(i+1),1]=np.random.normal(centre_points[i,1],sigma,DataSet[step_length*i:(step_length*(i+1)),0].shape)
            DataLabel[step_length*i:step_length*(i+1)]=i

        return DataSet,DataLabel








if __name__ == '__main__':
    dg = DataGenerator()
    data,lable = dg.HandleData()

    plt.figure(1)
    plt.grid(True)
    # plt.hold(True)

    plt.plot(data[:,0],data[:,1],'r*')
    plt.show()
