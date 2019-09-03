# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from math import sqrt
from numpy import array
from scipy.optimize import fmin_l_bfgs_b


# 自适应一次指数平滑
# def auto_exp_smoothing(data, alpha, k):
#     '''
#     一次指数平滑
#     :param alpha:  平滑系数
#     :param s:      数据序列， list
#     :return:       返回一次指数平滑模型参数， list
#     '''
#     def RMSE(params, *args):
#         Y = args[0]
#         alpha = params
#         y = [Y[0]]    # 初始化平滑值
#         for i in range(len(Y)):
#             y.append(alpha*Y[i] + (1-alpha)*y[i])
#         rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:], y[:-1])]) / len(Y))
#         return rmse

#     def linear(x, alpha=None):
#         Y = x[:].values.tolist()
#         if(alpha == None):
#             initial_values = np.array(0.2)
#             boundaries = [(0.01, 0.4)]
#             type = 'linear'
#             parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, type), bounds=boundaries, approx_grad=True)
#             alpha = parameters[0][0]
#         a = [Y[0]]
#         for i in range(1, len(Y)):
#             a.append(alpha*Y[i] + (1-alpha)*a[i-1])

#         rmse = sqrt(sum([(m-n)**2 for m, n in zip(Y[:], a[:])]) / len(Y))
#         return a, alpha, rmse

#     data = pd.Series(data)
#     smooth_data, alpha, rmse = linear(data, alpha=alpha)

#     pre_data = smooth_data[-1]
#     diff = np.array(smooth_data[:-1] - data.values[:-1])
#     high = pre_data + k * diff.std()
#     low = pre_data - k * diff.std()
#     return high, low, pre_data




# def holt_winters(time_series, period, alpha, beta, gamma, k):
#     time_series = list(time_series)
#     if len(time_series) < 3*period:
#         print("the time series is too short!!")
#         return
#
#     def RMSE(params, *args):
#         Y = args[0]
#         m = args[1]
#         alpha, beta, gamma = params
#         rmse = 0
#         a = [sum(Y[0:m]) / float(m)]
#         b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
#         s = [Y[i] - a[0] for i in range(m)]
#         y = [a[0] + b[0] + s[0]]
#
#         for i in range(len(Y)):
#             a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
#             b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
#             s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
#             y.append(a[i + 1] + b[i + 1] + s[i + 1])
#
#         rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))
#         return rmse
#
#     def additive(x, m, alpha=None, beta=None, gamma=None):
#         Y = x[:]
#         if (alpha==None or beta==None or gamma==None):
#             initial_values = array([0.3, 0.1, 0.1])
#             boundaries = [(0, 1), (0, 1), (0, 1)]
#             parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, m), bounds=boundaries, approx_grad=True)
#             alpha, beta, gamma = parameters[0]
#         a = [sum(Y[0:m]) / float(m)]
#         b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
#         s = [Y[i] - a[0] for i in range(m)]
#         y = [a[0] + b[0] + s[0]]
#         rmse = 0
#
#         for i in range(1, len(Y)):
#             a.append(alpha * (Y[i-1] - s[i-1]) + (1-alpha) * (a[i-1] + b[i-1]))
#             b.append(beta * (a[i] - a[i-1]) + (1 - beta) * b[i-1])
#             s.append(gamma * (Y[i-1] - a[i-1] - b[i-1]) + (1-gamma) * s[i-1])
#             y.append(a[i] + b[i] + s[i])
#
#         rmse = sqrt(sum([(c - d) ** 2 for c, d in zip(Y[:], y[:])]) / len(Y))
#         return y, alpha, beta, gamma, rmse
#
#     smooth_data, alpha, beta, gamma, rmse = additive(time_series, m=period, alpha=alpha, beta=beta, gamma=gamma)
#
#     pre_data = smooth_data[-1]
#     diff = []
#     for d1, d2 in zip(smooth_data[:-1], time_series[:-1]):
#         diff.append(d1-d2)
#     # diff = np.array(smooth_data[:-1] - time_series[:-1])
#     diff = np.array(diff)
#     high = pre_data + k * diff.std()
#     low = pre_data - k * diff.std()
#     return high, low, pre_data
#
# def holt_winters2(time_series, period, alpha, gamma, k):
#     time_series = list(time_series)
#     if len(time_series) < 3*period:
#         print("the time series is too short!!")
#         return
#
#     def RMSE(params, *args):
#         Y = args[0]
#         m = args[1]
#         alpha, gamma = params
#         rmse = 0
#         a = [sum(Y[0:m]) / float(m)]
#         s = [Y[i] - a[0] for i in range(m)]
#         y = [a[0] + s[0]]
#
#         for i in range(len(Y)):
#             a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * a[i])
#             s.append(gamma * (Y[i] - a[i]) + (1 - gamma) * s[i])
#             y.append(a[i + 1] + s[i + 1])
#
#         rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))
#         return rmse
#
#     def additive(x, m, alpha=None, gamma=None):
#         Y = x[:]
#         if (alpha==None or gamma==None):
#             initial_values = array([0.3, 0.1])
#             boundaries = [(0, 1), (0, 1)]
#             parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, m), bounds=boundaries, approx_grad=True)
#             alpha, gamma = parameters[0]
#         a = [sum(Y[0:m]) / float(m)]
#         s = [Y[i] - a[0] for i in range(m)]
#         y = [a[0] + s[0]]
#         rmse = 0
#
# #         for i in range(1, len(Y)):
# #             a.append(alpha * (Y[i-1] - s[i-1]) + (1-alpha) * (a[i-1] + b[i-1]))
# #             b.append(beta * (a[i] - a[i-1]) + (1 - beta) * b[i-1])
# #             s.append(gamma * (Y[i-1] - a[i-1] - b[i-1]) + (1-gamma) * s[i-1])
# #             y.append(a[i] + b[i] + s[i])
#         for i in range(len(Y)-1):
#             a.append(alpha * (Y[i+1] - s[i]) + (1 - alpha) * a[i])
#             s.append(gamma * (Y[i+1] - a[i]) + (1 - gamma) * s[i])
#             y.append(a[i + 1] + s[i + 1])
#
#         rmse = sqrt(sum([(c - d) ** 2 for c, d in zip(Y[:], y[:-1])]) / len(Y))
#         return y, alpha, gamma, rmse
#
#     smooth_data, alpha, gamma, rmse = additive(time_series, m=period, alpha=alpha, gamma=gamma)
#
#     pre_data = smooth_data[-2]
#     diff = []
#     for d1, d2 in zip(smooth_data[:-1], time_series[:-1]):
#         diff.append(d1-d2)
#     # diff = np.array(smooth_data[:-1] - time_series[:-1])
#     diff = np.array(diff)
#     high = pre_data + k * diff.std()
#     low = pre_data - k * diff.std()
#     return high, low, pre_data


def RMSE(params, *args):
    Y = args[0]
    type = args[1]
    if type == 'single':
        alpha = params
        a = [Y[0]]
        y = [a[0]]
        for i in range(1, len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * a[i-1])
            y.append(a[i])
        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[1:], y[:-1])]) / len(Y))
    elif type == 'linear':
        alpha, beta = params
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] + b[0]]
        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] + b[i + 1])
        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))
    else:
        alpha, beta, gamma = params
        m = args[2]
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
        if type == 'additive':
            s = [Y[i] - a[0] for i in range(m)]
            y = [a[0] + b[0] + s[0]]

            for i in range(len(Y)):
                a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
                y.append(a[i + 1] + b[i + 1] + s[i + 1])
        else:
            exit('Type must be either linear, trend or additive')
        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))
    return rmse


def single_exp(x, fc, alpha=None):
    Y = x[:]
    if (alpha == None):
        initial_values = array([0.3])
        boundaries = [(0, 1)]
        type = 'single'
        parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, type), bounds=boundaries, approx_grad=True)
        alpha = parameters[0][0]
    a = [(Y[0]+Y[1]+Y[2])/3]
    y = [a[0]]

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(a[-1])
        a.append(alpha * Y[i] + (1 - alpha) * a[i-1])
        y.append(a[i + 1])
    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
    print "alpha = ", alpha
    print "rmse = ", rmse
    print Y[-fc:]
    print y[-fc:]
    return Y[-fc:], alpha, rmse


def linear(x, fc, alpha=None, beta=None):
    Y = x[:]
    if (alpha==None or beta==None):
        initial_values = array([0.3, 0.1])
        boundaries = [(0, 1), (0, 1)]
        type = 'linear'
        parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, type), bounds=boundaries, approx_grad=True)
        alpha, beta = parameters[0]
    a = [Y[0]]
    b = [Y[1] - Y[0]]
    y = [a[0] + b[0]]

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(a[-1] + b[-1])
        a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        y.append(a[i + 1] + b[i + 1])
    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
    print "alpha = ", alpha
    print "beta = ", beta
    return Y[-fc:], alpha, beta, rmse


def additive(x, m, fc, alpha=None, beta=None, gamma=None):
    Y = x[:]
    if (alpha==None or beta==None or gamma==None):
        initial_values = array([0.3, 0.1, 0.1])
        boundaries = [(0, 1), (0, 1), (0, 1)]
        type = 'additive'
        parameters = fmin_l_bfgs_b(RMSE, x0=initial_values, args=(Y, type, m), bounds=boundaries, approx_grad=True)
        alpha, beta, gamma = parameters[0]
    a = [sum(Y[0:m]) / float(m)]
    b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
    s = [Y[i] - a[0] for i in range(m)]
    y = [a[0] + b[0] + s[0]]

    for i in range(len(Y) + fc):
        if i == len(Y):
            Y.append(a[-1] + b[-1] + s[-m])

        a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
        b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
        s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
        y.append(a[i + 1] + b[i + 1] + s[i + 1])
    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))
    return Y[-fc:], alpha, beta, gamma, rmse


if __name__ == "__main__":
    period = 1440
    data = pd.read_csv("tb_link_ad_result_815.csv", header=None,
                       names=['link_Id', 'log_time', 'flow_type', 'true_value', 'high', 'low', 'score'])
    log_time = []
    for item in data["log_time"]:
        log_time.append(datetime.strptime(item, "%d/%m/%Y %H:%M:%S"))
    data["log_time"] = log_time

    data1 = data[(data['link_Id'] == '6fbc00c2-0a91-44aa-bb8e-9df410ff7cd2') & (data['flow_type'] == 'speed_1_to_2') &
                 (data['log_time'] > datetime.strptime('5/8/2019 16:48:00', "%d/%m/%Y %H:%M:%S"))]
    data1.drop_duplicates(subset=['link_Id', 'log_time'], keep='first', inplace=True)
    data1.sort_values(by='log_time', inplace=True)
    data1.drop(['link_Id', 'flow_type', 'high', 'low', 'score'], axis=1, inplace=True)

    log_time = data1["log_time"]
    true_value = data1["true_value"]

    len_data = len(true_value)
    print "len_data = ", len_data

    pre_len = 3000
    win_data = list(true_value[:len_data-pre_len])
    print win_data[:20]

    # single_result, alpha0, rmse0 = single_exp(win_data, fc=pre_len)
    # linear_result, alpha1, beta1, rmse1 = linear(win_data, fc=pre_len)
    holt_result, alpha2, beta2, gamma2, rmse2 = additive(win_data, m=period, fc=pre_len)

    # for i in range(len_data - 14200):
    #     win_data = true_value[i:i + 14200]
    #     # pre_data, high, low = holt_winters(win_data, period=period, alpha=None, beta=None, gamma=None, k=3)
    #
    #
    plt.figure(figsize=[20, 10])
    plt.plot(log_time.values[:], true_value.values[:], color='b')
    # plt.plot(log_time.values[-pre_len:], single_result, color='r', label="single")
    # plt.plot(log_time.values[-pre_len:], linear_result, color='k', label="linear")
    plt.plot(log_time.values[-pre_len:], holt_result, color='g', label="holt_winter")
    plt.legend()
    plt.show()

