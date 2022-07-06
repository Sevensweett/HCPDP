# *_coding=utf-8_*
import numpy as np
from sklearn import linear_model
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.colors
import matplotlib.pyplot as plt
from functools import reduce

from scipy import stats
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances_argmin
import math
from scipy.integrate import tplquad, dblquad, quad
import scipy.stats

# from imblearn.over_sampling import RandomOverSampler

from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

from scipy.interpolate import make_interp_spline

from sklearn import neighbors

from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块
from sklearn.naive_bayes import GaussianNB  # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
# import KNN
# import logistic
# import mysvm


# from imblearn.over_sampling import RandomOverSampler, smote
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler

from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN

from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import InstanceHardnessThreshold

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
import random

from sklearn import preprocessing
import bayes
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE
from collections import Counter

import scipy as sp

import ot

import sys
import os


from matplotlib.colors import ListedColormap

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 设置字体
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def dataGetPROMISE():
    data = []
    data1 = np.loadtxt('.\\PROMISE\\camel-1.6.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(3, 24))
    data2 = np.loadtxt('.\\PROMISE\\lucene-2.4.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(3, 24))
    data3 = np.loadtxt('.\\PROMISE\\poi-3.0.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(3, 24))
    data4 = np.loadtxt('.\\PROMISE\\synapse-1.2.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(3, 24))
    data5 = np.loadtxt('.\\PROMISE\\velocity-1.6.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(3, 24))
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    return data


def dataGetNASA():
    """
        读取NASA下的的7个项目的数据集
        :return: 长度为7的list，内容为NASA下的7个项目的数据集
    """
    data = []
    data1 = np.loadtxt('.\\NASA\\KC1.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 22))
    data2 = np.loadtxt('.\\NASA\\MC1.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 39))
    data3 = np.loadtxt('.\\NASA\\PC1.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 38))
    data4 = np.loadtxt('.\\NASA\\PC3.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 38))
    data5 = np.loadtxt('.\\NASA\\PC4.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 38))
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    return data


def dataGetAEEEM():
    data = []
    data1 = np.loadtxt('.\\AEEEM\\eq.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 62))
    data2 = np.loadtxt('.\\AEEEM\\jdt.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 62))
    data3 = np.loadtxt('.\\AEEEM\\lc.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 62))
    data4 = np.loadtxt('.\\AEEEM\\ml.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 62))
    data5 = np.loadtxt('.\\AEEEM\\pde.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 62))
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    return data


def dataGetJIRA():
    data = []
    data1 = np.loadtxt('.\\JIRA\\activemq-5.0.0.csv', dtype=np.float64, delimiter=',', skiprows=1,
                       usecols=np.arange(0, 66))
    data2 = np.loadtxt('.\\JIRA\\derby-10.5.1.1.csv', dtype=np.float64, delimiter=',', skiprows=1,
                       usecols=np.arange(0, 66))
    data3 = np.loadtxt('.\\JIRA\\groovy-1_6_BETA_1.csv', dtype=np.float64, delimiter=',', skiprows=1,
                       usecols=np.arange(0, 66))
    data4 = np.loadtxt('.\\JIRA\\hbase-0.94.0.csv', dtype=np.float64, delimiter=',', skiprows=1,
                       usecols=np.arange(0, 66))
    data5 = np.loadtxt('.\\JIRA\\hive-0.9.0.csv', dtype=np.float64, delimiter=',', skiprows=1, usecols=np.arange(0, 66))
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    return data


def dataPreTreated(data):
    List = []
    target = []
    for i in range(len(data)):
        data_new, data_t = np.split(data[i], [-1, ], axis=1)
        List.append(preprocessing.scale(data_new))
        for j in range(len(data_t)):
            if (data_t[j] > 1):
                data_t[j] = 1
        target.append(data_t)
    return List, target


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



sys.stdout = Logger(r'result/egw+JIRA+NASA.txt')

from sklearn.metrics import classification_report
# sys.path.append(r'C:\Users\ASUS\AppData\Local\Programs\Python\Python38')
# from liblinear import *
from liblinear.liblinearutil import *
from sklearn.metrics import confusion_matrix

import time

start = time.time()

data_nasa = dataGetNASA()
data_nasa, target_nasa = dataPreTreated(data_nasa)
print(len(data_nasa))
print(data_nasa[0].shape)
print(data_nasa[0][0])

data_promise = dataGetPROMISE()
data_promise, target_promise = dataPreTreated(data_promise)
print(len(data_promise))
print(data_promise[0].shape)
print(data_promise[0][0])

size_nasa = len(data_nasa)
size_promise = len(data_promise)

allf = np.zeros((size_promise, size_nasa))
allauc = np.zeros((size_promise, size_nasa))
allgmean = np.zeros((size_promise, size_nasa))
allpd = np.zeros((size_promise, size_nasa))
allpf = np.zeros((size_promise, size_nasa))

sallf = np.zeros((size_promise, size_nasa))
sallauc = np.zeros((size_promise, size_nasa))
sallgmean = np.zeros((size_promise, size_nasa))
sallpd = np.zeros((size_promise, size_nasa))
sallpf = np.zeros((size_promise, size_nasa))

avg_f_avg = np.zeros((10, 10))
avg_auc_avg = np.zeros((10, 10))
avg_g_avg = np.zeros((10, 10))
avg_pd_avg = np.zeros((10, 10))
avg_pf_avg = np.zeros((10, 10))

e = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
l = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]

# per_data = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
per_data = [0.1]

num_index = 20

import heapq
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN


e_l = len(e)
l_l = len(l)
for e_i in range(e_l):
    for l_i in range(l_l):
        allf = np.zeros((size_promise, size_nasa))
        allauc = np.zeros((size_promise, size_nasa))
        allgmean = np.zeros((size_promise, size_nasa))

        allf = np.zeros((size_nasa, size_promise))
        allauc = np.zeros((size_nasa, size_promise))
        allgmean = np.zeros((size_nasa, size_promise))
        allpd = np.zeros((size_nasa, size_promise))
        allpf = np.zeros((size_nasa, size_promise))

        sallf = np.zeros((size_promise, size_nasa))
        sallauc = np.zeros((size_promise, size_nasa))
        sallgmean = np.zeros((size_promise, size_nasa))

        sallf = np.zeros((size_nasa, size_promise))
        sallauc = np.zeros((size_nasa, size_promise))
        sallgmean = np.zeros((size_nasa, size_promise))
        sallpd = np.zeros((size_nasa, size_promise))
        sallpf = np.zeros((size_nasa, size_promise))
        for per_data_i in range(1):
            for i in range(size_nasa):
                # 以i作为训练集
                # 以j作为测试集
                datas_train = data_nasa[i]
                targets_train = target_nasa[i]

                # SMOTE通过合成少数样本解决数据不平衡的问题
                smo = SMOTE(random_state=42)
                data_train, target_train = smo.fit_resample(datas_train, targets_train)

                # cc = RandomUnderSampler(ratio='auto',random_state=42)
                # data_train, target_train = cc.fit_sample(datas_train, targets_train)

                target_train = target_train.reshape(-1, 1)

                C1 = sp.spatial.distance.cdist(data_train, data_train)
                C1 /= C1.max()

                ns = data_train.shape[0]
                # unif函数返回长度为“ns”（单纯形）的统一直方图
                p = ot.unif(ns)
                for j in range(size_promise):
                    # data_test = data_nasa[j]
                    # target_test = target_nasa[j]
                    data_test = data_promise[j]
                    target_test = target_promise[j]

                    nt = data_test.shape[0]
                    rand_arr = np.arange(data_test.shape[0])

                    choose_num = math.floor(nt * per_data[per_data_i])

                    num_f = np.zeros(num_index)
                    num_auc = np.zeros(num_index)
                    num_gmean = np.zeros(num_index)
                    num_pd = np.zeros(num_index)
                    num_pf = np.zeros(num_index)
                    for num_i in range(num_index):
                        np.random.shuffle(rand_arr)  # 将rand_arr的中的数字顺序打乱
                        choose_data_test = data_test[rand_arr[0:choose_num]]  # 训练集中的前choose_num个数据项添加到训练集中
                        choose_target_test = target_test[rand_arr[0:choose_num]]

                        remain_data_test = data_test[rand_arr[choose_num:nt]]  # 剩下的数据（choose_num到nt间的数据项）作为测试集
                        remain_target_test = target_test[rand_arr[choose_num:nt]]

                        # 计算两个输入集合间的距离，根据“metric”选择距离度量值，默认为欧氏距离
                        C2 = sp.spatial.distance.cdist(choose_data_test, choose_data_test)
                        # C2 = sp.spatial.distance.cdist(data_test, data_test)
                        C2 /= C2.max()

                        choose_nt = choose_data_test.shape[0]
                        # choose_nt = data_test.shape[0]
                        q = ot.unif(choose_nt)

                        # gw, log = ot.gromov.entropic_gromov_wasserstein(
                        #   C1, C2, p, q, 'square_loss', epsilon=e[e_i], log=True, verbose=True)

                        # ot.gromov.entropic_gromov_wassertein返回两个空间C1和C2之间的最佳耦合
                        gw, log = ot.gromov.entropic_gromov_wasserstein_change(
                            data_train, target_train, choose_data_test, choose_target_test, C1, C2, p, q, 'square_loss',
                            epsilon=e[e_i], leimuda=l[l_i], log=True, verbose=True)

                        # 矩阵的乘法
                        change_data = ns * np.dot(gw, choose_data_test)
                        # change_data = ns * np.dot(gw,data_test)

                        # 输出训练结果
                        # data_jschoose = change_data
                        # target_jschoose = target_train

                        data_jschoose = np.vstack((change_data, choose_data_test))
                        target_jschoose = np.vstack((target_train, choose_target_test))

                        lr = train(target_jschoose.ravel(), data_jschoose, '-s 0')
                        p_label, p_acc, p_val = predict(target_test.ravel(), data_test, lr)
                        # p_label, p_acc, p_val = predict(remain_target_test.ravel(), remain_data_test, lr)

                        # mnb = KNeighborsClassifier(n_neighbors=5);
                        # mnb = GaussianNB();  # 使用默认配置初始化朴素贝叶斯
                        # mnb.fit(data_jschoose,target_jschoose.ravel())
                        # p_label = mnb.predict(np.array(remain_data_test))

                        C_matrix = confusion_matrix(target_test, p_label, labels=[1, 0])
                        tp = C_matrix[0][0]
                        tn = C_matrix[1][1]
                        fp = C_matrix[1][0]
                        fn = C_matrix[0][1]
                        re = tp / (tp + fn)
                        pf_1 = tn / (tn + fp)
                        g_mean = math.sqrt(pf_1 * re)
                        pd = tp / (tp + fn)
                        pf = fp / (fp + tn)

                        f_binary = f1_score(target_test, p_label, average="binary")
                        auc = roc_auc_score(target_test, p_label)

                        num_f[num_i] = f_binary
                        num_auc[num_i] = auc
                        num_gmean[num_i] = g_mean
                        num_pd[num_i] = pd
                        num_pf[num_i] = pf

                    print("i", i, "j", j, "num_i", num_i)
                    print("f")
                    print(num_f)
                    print("auc")
                    print(num_auc)
                    print("gmean")
                    print(num_gmean)
                    print("pd")
                    print(num_pd)
                    print("pf")
                    print(num_pf)

                    allf[i][j] = np.mean(num_f)
                    allauc[i][j] = np.mean(num_auc)
                    allgmean[i][j] = np.mean(num_gmean)
                    allpd[i][j] = np.mean(num_pd)
                    allpf[i][j] = np.median(num_pf)

                    sallf[i][j] = np.std(num_f, ddof=1)
                    sallauc[i][j] = np.std(num_auc, ddof=1)
                    sallgmean[i][j] = np.std(num_gmean, ddof=1)
                    sallpd[i][j] = np.std(num_pd, ddof=1)
                    sallpf[i][j] = np.std(num_pf, ddof=1)

    print("e_i", e_i)
    print("avg")
    print("f")
    for var_f in allf.tolist():
        print(var_f)
    print("auc")
    for var_auc in allauc.tolist():
        print(var_auc)
    print("g")
    for var_g in allgmean.tolist():
        print(var_g)
    print("pd")
    for var_pd in allpd.tolist():
        print(var_pd)

    print("pf")
    for var_pf in allpf.tolist():
        print(var_pf)
    print("std")
    print("f")
    for var_f in sallf.tolist():
        print(var_f)
    print("auc")
    for var_auc in sallauc.tolist():
        print(var_auc)
    print("g")
    for var_g in sallgmean.tolist():
        print(var_g)
    end = time.time()-start

    print("time")
    end = time.time()-start
    print(end)

    print("avg")
    print("f")
    avg_f_avg[e_i] = np.mean(np.mean(allf, axis=0))
    print(avg_f_avg[e_i])
    print("auc")
    avg_auc_avg[e_i] = np.mean(np.mean(allauc, axis=0))
    print(avg_auc_avg[e_i])
    print("g")
    avg_g_avg[e_i] = np.mean(np.mean(allgmean, axis=0))
    print(avg_g_avg[e_i])
    print("pd")
    avg_pd_avg[e_i] = np.mean(np.mean(allpd, axis=0))
    print(avg_pd_avg[e_i])
    print("pf")
    avg_pf_avg[e_i] = np.mean(np.mean(allpf, axis=0))
    print(avg_pf_avg[e_i])

print("result")
print(avg_f_avg)
print(avg_auc_avg)
print(avg_g_avg)
print(avg_pd_avg)
print(avg_pf_avg)
