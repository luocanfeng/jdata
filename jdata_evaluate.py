# -*- coding: utf-8 -*-
'''
@data: 2017-03-29
@author: luocanfeng
'''
import pandas as pd


'''
评分
参赛者提交的结果文件中包含对所有用户购买意向的预测结果。对每一个用户的预测结果包括两方面：
1、该用户2016-04-16到2016-04-20是否下单P中的商品，提交的结果文件中仅包含预测为下单的用户，
  预测为未下单的用户，无须在结果中出现。若预测正确，则评测算法中置label=1，不正确label=0；
2、如果下单，下单的sku_id （只需提交一个sku_id），若sku_id预测正确，则评测算法中置pred=1，
  不正确pred=0。
对于参赛者提交的结果文件，按如下公式计算得分：
Score=0.4*F11 + 0.6*F12
此处的F1值定义为：
F11=6*Recall*Precise/(5*Recall+Precise)
F12=5*Recall*Precise/(2*Recall+3*Precise)
其中，Precise为准确率，Recall为召回率.
F11是label=1或0的F1值，F12是pred=1或0的F1值.

A榜阶段
F11
正确率=预测A榜user_id正确数量/提交数量，
召回率=预测A榜user_id正确数量/A榜数据总量；
F12
正确率=预测A榜user_id+sku_id正确数量/提交数量，
召回率=预测A榜user_id+sku_id正确数量/A榜数据总量；

B榜阶段
F11
正确率=预测B榜user_id正确数量/提交数量，
召回率=预测B榜user_id 正确数量/B榜数据总量；
F12
正确率=预测B榜user_id+sku_id正确数量/提交数量，
召回率=预测B榜user_id+sku_id正确数量/B榜数据总量；
'''
def evaluate(predict, reality):
    f11_tp = pd.merge(predict, reality, how='inner', on='user_id').size()
    f11_precision = f11_tp / predict.size()
    f11_recall = f11_tp / reality.size()
    f11 = 6 * f11_recall * f11_precision / (5 * f11_recall + f11_precision)
    
    f12_tp = pd.merge(predict, reality, how='inner', on=['user_id', 'sku_id'])
    f12_precision = f12_tp / predict.size()
    f12_recall = f12_tp / reality.size()
    f12 = 5 * f12_recall * f12_precision / (2 * f12_recall + 3 * f12_precision)
    
    return 0.4 * f11 + 0.6 * f12, f11, f12

