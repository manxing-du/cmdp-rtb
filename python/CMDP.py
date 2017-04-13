# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 11:11:57 2016

@author: sassi
"""

import numpy as np
from scipy.optimize import linprog
import config
import pandas as pd
import pickle as pickle
from sklearn.metrics.cluster import normalized_mutual_info_score


def lr_cmdp(pctr, bins, market_price_pdf, pctr_pdf, max_bid, alpha, budget):
    market_price_pdf = np.asarray(market_price_pdf)
    actions = np.asarray(range(0, max_bid+1), dtype=int)
    reward = np.zeros((bins, (max_bid + 1)))
    cost = np.zeros((bins, (max_bid + 1)))
    for i in range(0, bins):
        for j in range(0, max_bid + 1):
            # reward[i][j] = sum(market_price_pdf[0:j + 1]) * pctr[i]
            # cost[i][j] = np.dot(market_price_pdf[0:j + 1], actions[0:j + 1])
            if j == 0:
                reward[i][j] = sum(market_price_pdf[i][0:j+1]) * pctr[i]
                cost[i][j] = np.dot(market_price_pdf[i][0:j+1], actions[0:j+1])
            else:
                reward[i][j] = sum(market_price_pdf[i][0:j]) * pctr[i]
                cost[i][j] = np.dot(market_price_pdf[i][0:j], actions[0:j])

    reward = reward.reshape((1, bins * (max_bid + 1)))
    reward = reward.ravel()
    cost = cost.reshape((1, bins*(max_bid + 1)))
    cost = np.asmatrix(cost)
    beq = [0 for i in range(0, bins + 1)]
    beq[-1] = 1
    d = np.zeros((bins+1, bins*(max_bid + 1)))

    for i in range ( 0,bins):
        d[i][0:] = -pctr_pdf[i]
        d[i][i*(max_bid+1):(i+1)*(max_bid+1)]+=1
    d[bins][0:] = 1
    # pctr distribution, sum = 1
    # Minimize -reward == maximize reward
    rho = linprog(-reward, A_ub=cost, b_ub=budget, A_eq=d, b_eq=beq, options={"maxiter": 5000, "tol": 1e-8})
    policy = [0 for i in range(0, bins)]
    for i in range(0, bins):
        policy[i]= np.argmax(rho.x[i * (max_bid+1):(i+1) * (max_bid+1)])
    return policy


def calc_m_pdf(m_counter, laplace=1):
    m_pdf = [0] * len(m_counter)
    sum = 0
    for i in range(0, len(m_counter)):
        sum += m_counter[i]
    for i in range(0, len(m_counter)):
        m_pdf[i] = (m_counter[i] + laplace) / (sum + len(m_counter) * laplace)
    return m_pdf


def calc_pctr_pdf(train_data, pctr, bins):
    pctr_pdf = np.asarray([0 for i in range(0,bins)], dtype= float)
    for i in range(train_data.shape[0]):
        theta = train_data.iloc[i:i + 1, 2].values[0]
        index = np.where(pctr <= theta)[-1][-1]
        pctr_pdf[index] += 1
    return pctr_pdf/train_data.shape[0]


def run(auction_in, policy, pctr, B, bid_log):

    log = "{:>10}\t{:>8}\t{:>8}\t{:>10}".format("bid_price", "win_price", "click", "budget")
    bid_log.write(log + "\n")

    b = B
    imp = 0
    clk = 0
    cost = 0
    cpmt = 0
    for i in range(auction_in.shape[0]):
        winprice = auction_in. iloc[i:i + 1, 1].values[0]
        click = auction_in.iloc[i:i + 1, 0].values[0]
        theta = auction_in.iloc[i:i + 1, 2].values[0]
        index = np.where(pctr <= theta)[-1][-1]
        a = max(0, min(b, policy[index]))

        if click == 1:
            cpmt += 1
        if a >= winprice:
            b -= winprice
            imp += 1
            clk += click
            cost += winprice

        log = "{:>10}\t{:>8}\t{:>8}\t{:>10}" .format(a, winprice, click, cost)
        bid_log.write(log + "\n")

    bid_log.flush()
    bid_log.close()
    return imp, clk, cost

c0 = 1/32
alpha = 1
pctr_init = [0]
bins = [10**(-7), 10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1)]
zip_bin = zip(bins, bins[1:])
for i, j in zip_bin:
    pctr_bins = np. linspace(i, j, num=20)
    pctr_init = np.append(pctr_init, pctr_bins[:-1])
pctr_bins2 = np.linspace(0.1, 1, num=10)
pctr = np.append(pctr_init, pctr_bins2)
# print (pctr)
bins = len(pctr)-1

src = "ipinyou"
obj_type = "clk"
clk_vp = 1
train_file = 'train.theta.txt'

if src == "ipinyou":
    camps = config.ipinyou_camps
    data_path = config.ipinyouPath
    max_market_price = config.ipinyou_max_market_price

elif src == "ola":
    camps = config.ola_camps
    data_path = config.olaPath
    max_market_price = config.ola_max_market_price


log_in = open(config.logPath + 'bid-stat' + "/{}_c0={}_obj={}_clkvp={}.txt".format(src, c0, obj_type, clk_vp), "w")
print("logs in {}".format(log_in.name))
log = "{:<60}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>9}\t {:>8}\t {:>8}"\
    .format("setting", "objective", "auction", "impression", "click", "cost", "win-rate", "CPM", "eCPC")
log_in.write(log + "\n")

for camp in camps:

    auction_in = pd.read_csv(data_path + camp + "/" + 'test.theta.txt', header=None, index_col=False, sep=' ',
                             names=['click', 'winprice', 'pctr'])
    train = pd.read_csv(data_path + camp + "/" + train_file, header=None, index_col=False, sep=' ',
                        names=['click', 'winprice', 'pctr'])
    camp_info = pickle.load(open(data_path + camp + "/" + 'info.txt', "rb"))
    b = int(camp_info["cost_train"] / camp_info["imp_train"] * c0)
    m_pdf = calc_m_pdf(camp_info["price_counter_train"])
    max_bid = len(m_pdf) - 1

    # CMDP
    # Distritize pctr and get the conditional probability for market price: P(market_price| pctr)

    freq, pctr1 = np.histogram(train['pctr'], bins=pctr)
    price_range = [i for i in range(max_bid+2)]
    pctr_pdf = freq/(sum(freq))
    joint_pdf = np.histogram2d(train['pctr'], train['winprice'], bins=[pctr, price_range])
    cond_pdf = joint_pdf[0]/sum(sum(joint_pdf[0]))
    for i in range(bins):
        if pctr_pdf[i] == 0:
            cond_pdf[i][:] = m_pdf
        else:
            cond_pdf[i][:] = cond_pdf[i][:] / pctr_pdf[i]
    cond_pdf = np.nan_to_num(cond_pdf)
    m_pdf = cond_pdf

    # Log the bid price
    bid_log = open(
        config.logPath + "bid-log/{}/{}_camp={}_c0={}_obj={}.txt".format("CMDP", src, camp, c0, obj_type), "w")
    print("bid logs in {}".format(bid_log.name))

    b_total = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * auction_in.shape[0])
    print('Normalized Mutual Information Score is %.2f' % normalized_mutual_info_score(train['pctr'],
                                                                                       train['winprice']))

    policy = lr_cmdp(pctr, bins, m_pdf, pctr_pdf, max_bid, alpha, b)
    (imp, clk, cost) = run(auction_in, policy, pctr, b_total, bid_log)
    auction = auction_in.shape[0]
    win_rate = imp / auction * 100
    cpm = (cost / 1000) / imp * 1000
    ecpc = (cost / 1000) / clk
    # obj = clk
    obj = (auction_in['click'] == 1).sum()
    setting = "{}, camp={}, algo={}, c0={}".format(src, camp, "CMDP", c0)
    log = "{:<60}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}"\
        .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
    print(log)
    log_in.write(log + "\n")

    # Batch_CMDP
    groups = train.groupby(pd.cut(train.pctr, pctr)).mean()
    groups = groups.fillna(0)
    pctr_group = groups['click'].values

    # joint_pdf = np.histogram2d(train['pctr'], train['winprice'], bins=[pctr, price_range])
    # cond_pdf = joint_pdf[0] / sum(sum(joint_pdf[0]))
    # for i in range(bins):
    #     if pctr_pdf[i] == 0:
    #         cond_pdf[i][:] = m_pdf
    #     else:
    #         cond_pdf[i][:] = cond_pdf[i][:] / pctr_pdf[i]
    # cond_pdf = np.nan_to_num(cond_pdf)
    # m_pdf = cond_pdf

    # Log the bid price
    bid_log = open(
        config.logPath + "bid-log/{}/{}_camp={}_c0={}_obj={}.txt".format("batch-CMDP", src, camp, c0, obj_type), "w")
    print("bid logs in {}".format(bid_log.name))

    b_total = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * auction_in.shape[0])
    policy = lr_cmdp(pctr_group, bins, m_pdf, pctr_pdf, max_bid, alpha, b)
    (imp, clk, cost) = run(auction_in, policy, pctr, b_total, bid_log)
    auction = auction_in.shape[0]
    win_rate = imp / auction * 100
    cpm = (cost / 1000) / imp * 1000
    ecpc = (cost / 1000) / clk
    obj = (auction_in['click'] == 1).sum()
    setting = setting = "{}, camp={}, algo={}, c0={}".format(src, camp, "batch-CMDP", c0)
    log = "{:<70}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
        .format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
    print(log)
    log_in.write(log + "\n")

log_in.flush()
log_in.close()
