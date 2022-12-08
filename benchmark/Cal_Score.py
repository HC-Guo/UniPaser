#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hongchengguo
"""
import numpy as np
from scipy import sparse as sp
import time
import math

class cal_score():
    def __init__(self, labels_true, labels_pred):
        if len(labels_pred) != len(labels_true):
            raise ValueError('len(label)和len(label_true)不等')
        self.labels_pred = np.array(labels_pred)
        self.labels_true = np.array(labels_true)
        n_samples, = self.labels_true.shape
        c = self.contingency_matrix()
        self.TP = (np.dot(c.data, c.data) - n_samples)/2 #是同一类的，也分到了同一类的个数
        self.FP = (np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples)/2 - self.TP #不是同一类的，分到了同一类的个数
        self.FN = (np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples)/2 - self.TP #是同一类的，分到了不同类的个数
        self.TN = (pow(n_samples,2)-n_samples)/2 - self.TP - self.FP - self.FN #不是同一类的，也分到不同类的个数

    def contingency_matrix(self, dtype=np.int64):
        classes, class_idx = np.unique(self.labels_true, return_inverse=True)
        clusters, cluster_idx = np.unique(self.labels_pred, return_inverse=True)
        self.classes = classes
        self.clusters = clusters
        n_classes = classes.shape[0]
        n_clusters = clusters.shape[0]
        #生成混淆矩阵
        contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                      (class_idx, cluster_idx)),
                                    shape=(n_classes, n_clusters))
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
        self._contingency = contingency

        return contingency

    def precision(self):
        return self.TP/(self.TP + self.FP)

    def recall(self):
        return self.TP/(self.TP + self.FN)

    def f_measure(self):
        return 2*(self.precision()*self.recall())/(self.precision()+self.recall())

    def fowlkes_mallows_score(self):
        #FM指数
        return self.TP/pow((self.TP+self.FP)*(self.TP+self.FN),0.5)
    
    def jaccard_coefficient(self):
        #Jaccard系数
        return self.TP/(self.TP+self.FP+self.FN)
    
    def rand_index(self):
        #Rand指数
        return (self.TP+self.TN)/(self.TP+self.FP+self.FN+self.TN)

    def parsing_accuracy(self):
        #解析PA指数
        contingency = self._contingency
        """
        假如有稀疏矩阵
        [1,0,0,3]
        [0,3,0,4]
        [6,0,0,0]
        """
        data = contingency.data  # 稀疏矩阵非零点[1,3,3,4,6]
        n_samples = len(self.labels_true)
        indices = contingency.indices  # 稀疏矩阵非零点列坐标[0,3,1,3,0]
        indptr = contingency.indptr  # indices换行位置[0,2,4,5]
        n_data = data.shape[0]  # 稀疏矩阵非零点个数
        Ci = np.asarray(contingency.sum(axis=0)).ravel()  # 每一列的和(即聚类簇的个数)
        Pj = np.asarray(contingency.sum(axis=1)).ravel()  # 每一行的和（即标注簇的个数）
        # 开始计算
        index = 0
        PA = 0
        for i_data in range(n_data):
            if i_data >= indptr[index+1]:
                index += 1
            if data[i_data] == Pj[index] and Pj[index] == Ci[indices[i_data]]:
                PA += data[i_data]
        PA /= n_samples
        return PA

    def class_f_byPj(self):

        contingency = self._contingency
        """
        假如有稀疏矩阵
        [1,0,0,3]
        [0,3,0,4]
        [6,0,0,0]
        """
        data = contingency.data        #稀疏矩阵非零点[1,3,3,4,6]
        indices = contingency.indices   #稀疏矩阵非零点列坐标[0,3,1,3,0]
        indptr = contingency.indptr     #indices换行位置[0,2,4,5]
        
        n_classes = contingency.shape[0]    #稀疏矩阵行数，也会等于indices.shape[0]-1
        n_data = data.shape[0]              #稀疏矩阵非零点个数
        P_x, R_x, F_x = np.zeros(n_data), np.zeros(n_data) ,np.zeros(n_data)    #定义每个非零点对应的准确率Px、召回率R_x,F值F_x
        Ci = np.asarray(contingency.sum(axis=0)).ravel()            #每一列的和(即聚类簇的个数)
        Pj = np.asarray(contingency.sum(axis=1)).ravel()          #每一行的和（即标注簇的个数）
        
        F_Pj = np.zeros(n_classes)                        #每行F_x最大值
        ClassF_byPj=0                                     #Class_F值
        
        #开始计算
        index=0
        for i_data in range(n_data):
            if i_data >= indptr[index+1]:
                F_Pj[index] = max(F_x[indptr[index]:indptr[index+1]])
                ClassF_byPj += F_Pj[index] * Pj[index]
                index += 1
            P_x[i_data] = data[i_data] / Ci[indices[i_data]]
            R_x[i_data] = data[i_data] / Pj[index]
            F_x[i_data] = 2 * P_x[i_data] * R_x[i_data] / (P_x[i_data] + R_x[i_data])
        
        n_samples = len(self.labels_true)
        ClassF_byPj = sum(F_Pj * Pj) / n_samples
        
        return ClassF_byPj
        
        
    def class_f_byCi(self):

        contingency = self._contingency
        """
        假如有稀疏矩阵
        [1,0,0,3]
        [0,3,0,4]
        [6,0,0,0]
        """
        data = contingency.data        #稀疏矩阵非零点[1,3,3,4,6]
        indices = contingency.indices   #稀疏矩阵非零点列坐标[0,3,1,3,0]
        indptr = contingency.indptr     #indices换行位置[0,2,4,5]
        
        n_clusters = contingency.shape[1]    #稀疏矩阵列数
        
        n_data = data.shape[0]              #稀疏矩阵非零点个数
        P_x, R_x, F_x = np.zeros(n_data), np.zeros(n_data) ,np.zeros(n_data)    #定义每个非零点对应的准确率Px、召回率R_x,F值F_x
        Ci = np.asarray(contingency.sum(axis=0)).ravel()            #每一列的和(即聚类簇的个数)
        Pj = np.asarray(contingency.sum(axis=1)).ravel()          #每一行的和（即标注簇的个数）
        
        F_Ci = np.zeros(n_clusters)                        #每列F_x最大值
        ClassF_byCi=0                                     #Class_F值
        
        #开始计算
        index=0
        for i_data in range(n_data):
            if i_data >= indptr[index+1]:
                index += 1
            P_x[i_data] = data[i_data] / Ci[indices[i_data]]
            R_x[i_data] = data[i_data] / Pj[index]
            F_x[i_data] = 2 * P_x[i_data] * R_x[i_data] / (P_x[i_data] + R_x[i_data])
            F_Ci[indices[i_data]] = max(F_Ci[indices[i_data]], F_x[i_data])

        n_samples = len(self.labels_true)
        ClassF_byCi = sum(F_Ci * Ci) / n_samples
       
        return ClassF_byCi      
            
    def class_f_byx(self):

        contingency = self._contingency
        """
        假如有稀疏矩阵
        [1,0,0,3]
        [0,3,0,4]
        [6,0,0,0]
        """
        data = contingency.data        #稀疏矩阵非零点[1,3,3,4,6]
        indices = contingency.indices   #稀疏矩阵非零点列坐标[0,3,1,3,0]
        indptr = contingency.indptr     #indices换行位置[0,2,4,5]    
        n_data = data.shape[0]              #稀疏矩阵非零点个数
        P_x, R_x, F_x = np.zeros(n_data), np.zeros(n_data) ,np.zeros(n_data)    #定义每个非零点对应的准确率Px、召回率R_x,F值F_x
        Ci = np.asarray(contingency.sum(axis=0)).ravel()            #每一列的和(即聚类簇的个数)
        Pj = np.asarray(contingency.sum(axis=1)).ravel()          #每一行的和（即标注簇的个数）
        P_average, R_average, F_average = 0, 0, 0
        ClassF_byCi=0                                     #Class_F值
        
        #开始计算
        index=0
        for i_data in range(n_data):
            if i_data >= indptr[index+1]:
                index += 1
            P_x[i_data] = data[i_data] / Ci[indices[i_data]]
            R_x[i_data] = data[i_data] / Pj[index]
            F_x[i_data] = 2 * P_x[i_data] * R_x[i_data] / (P_x[i_data] + R_x[i_data])
        n_samples = len(self.labels_true)
        
        P_average = sum(P_x * data) / n_samples
        R_average = sum(R_x * data) / n_samples
        F_average = sum(F_x * data) / n_samples    
        
        return P_average, R_average, F_average      
        
    def entropy_Ci(self):
        contingency = self._contingency
        """
        假如有稀疏矩阵
        [1,0,0,3]
        [0,3,0,4]
        [6,0,0,0]
        """
        data = contingency.data        #稀疏矩阵非零点[1,3,3,4,6]
        indices = contingency.indices   #稀疏矩阵非零点列坐标[0,3,1,3,0]
        
        n_clusters = contingency.shape[1]    #稀疏矩阵列数
        
        n_data = data.shape[0]              #稀疏矩阵非零点个数
        P_x, entropy = np.zeros(n_data), np.zeros(n_clusters)  #定义每个非零点对应的准确率Px、定义每一列的熵（即每个聚类簇的熵）
        Ci = np.asarray(contingency.sum(axis=0)).ravel()            #每一列的和(即聚类簇的个数)
        numPj_inCi = np.zeros(n_clusters)                           #每一列的非零点个数(即一个聚类簇中，标注簇的种类数)
        ent = np.zeros(n_clusters)                      #每一列熵计算的中间产物 ∑P_x*(-log(P_x))
        #开始计算
        n_samples = len(self.labels_true)
        for i_data in range(n_data):
            P_x[i_data] = data[i_data] / Ci[indices[i_data]]
            ent[indices[i_data]] += P_x[i_data] * (-np.log(P_x[i_data]))
            numPj_inCi[indices[i_data]] += 1
        for i in range(n_clusters):
            if numPj_inCi[i] == 1 and ent[i] == 0:
                entropy[i] = 0
            elif numPj_inCi[i] != 1 and ent[i] != 0:
                entropy[i] = 1 / np.log(numPj_inCi[i]) * ent[i]
                # entropy[i] *= Ci[i]/n_samples
            else:
                raise ValueError('计算有问题,ent[i]为'+str(ent[i]),'而numPj_inCi[i]为'+str(numPj_inCi[i]))
        result = {}
        for i in range(len(entropy)):
            result[self.clusters[i]] = entropy[i]
        return result

    def entropy_Pj(self):
        contingency = self._contingency
        """
        假如有稀疏矩阵
        [1,0,0,3]
        [0,3,0,4]
        [6,0,0,0]
        """
        data = contingency.data  # 稀疏矩阵非零点[1,3,3,4,6]
        indices = contingency.indices  # 稀疏矩阵非零点列坐标[0,3,1,3,0]
        indptr = contingency.indptr  # indices换行位置[0,2,4,5]
        n_clusters = contingency.shape[0]  # 稀疏矩阵列数

        n_data = data.shape[0]  # 稀疏矩阵非零点个数
        P_x, entropy = np.zeros(n_data), np.zeros(n_clusters)  # 定义每个非零点对应的准确率Px、定义每一列的熵（即每个聚类簇的熵）
        Pj = np.asarray(contingency.sum(axis=1)).ravel()  # 每一行的和(即人工标注簇的个数)
        numPj_inCi = np.zeros(n_clusters)  # 每一列的非零点个数(即一个人工标注簇簇中，聚类簇的种类数)
        ent = np.zeros(n_clusters)  # 每一列熵计算的中间产物 ∑P_x*(-log(P_x))
        # 开始计算
        n_samples = len(self.labels_true)
        index = 0
        for i_data in range(n_data):
            if i_data >= indptr[index+1]:
                index += 1
            P_x[i_data] = data[i_data] / Pj[index]
            ent[index] += P_x[i_data] * (-np.log(P_x[i_data]))
            numPj_inCi[index] += 1
        for i in range(n_clusters):
            if numPj_inCi[i] == 1 and ent[i] == 0:
                entropy[i] = 0
            elif numPj_inCi[i] != 1 and ent[i] != 0:
                entropy[i] = 1 / np.log(numPj_inCi[i]) * ent[i]
                # entropy[i] *= Ci[i]/n_samples
            else:
                raise ValueError('计算有问题,ent[i]为' + str(ent[i]), '而numPj_inCi[i]为' + str(numPj_inCi[i]))
        result = {}
        for i in range(len(entropy)):
            result[self.classes[i]] = entropy[i]
        return result
    
    
if __name__ == '__main__':
    labels_true = [1,2,4,4]
    labels_pred = [2,2,4,4]
    col = cal_score(labels_true, labels_pred)
    precision = col.precision()
    recall = col.recall()
    f_measure = col.f_measure()
    PA = col.parsing_accuracy()
    
    
    
    
    