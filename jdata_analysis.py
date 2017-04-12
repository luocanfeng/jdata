# -*- coding: utf-8 -*-
'''
@data: 2017-04-12
@author: luocanfeng

TODO 用户行为+时间周期

用户活跃情况：
  统计用户近期的操作情况，以天为纬度统计，则产生6*3*30=540个属性
用户对于产品、品牌的喜好：
  统计用户、产品维度，以及用户、品牌纬度的各种操作总和
  用户在购买该款产品前，需要浏览多少次该产品，经历多长时间；
'''
import numpy as np
import pandas as pd
from sklearn import linear_model


file_encoding = 'gbk'
output_path = './out/'


file_tf_fsa_users = output_path + 'tf_fsa_users.csv'
file_tf_dd_fsa_users = output_path + 'tf_dd_fsa_users.csv'


'''
购买行为字段的相关性：
  buy_num >= 0
    addcart_num          0.349109
    cart_num             0.317944
    delcart_num          0.243275
    browse_num           0.192228
    click_num            0.186196
    user_lv_cd           0.147698
    cartnum2buy          0.130005
    browse2buy           0.092909
    favor_num            0.090517
    favor2buy            0.060134
    click2buy            0.010718
    cart2buy             0.007924
    
  buy_num > 0
    favor2buy            0.668391
    cart2buy             0.488339
    cartnum2buy          0.476147
    addcart_num          0.385979
    cart_num             0.330341
    browse2buy           0.326756
    browse_num           0.299422
    delcart_num          0.299187
    click_num            0.286777
    user_lv_cd           0.213969
    click2buy            0.137958
    favor_num            0.126097
    
  buy_num > 1
    addcart_num          0.317581
    cart_num             0.312401
    delcart_num          0.182803
    user_lv_cd           0.115500
    browse2buy           0.114848
    browse_num           0.112821
    click_num            0.111033
    cartnum2buy          0.074913
    favor2buy            0.064440
    favor_num            0.049805
    cart2buy             0.046835
    click2buy            0.024243
    
  buy_num > 2
    cart_num             0.332957
    addcart_num          0.317692
    delcart_num          0.144616
    browse2buy           0.126604
    user_lv_cd           0.122852
    favor2buy            0.079607
    click_num            0.070304
    browse_num           0.066657
    cart2buy             0.058010
    cartnum2buy          0.052795
    click2buy            0.025151
    favor_num            0.021965
    
  buy_num > 3
    cart_num             0.424964
    addcart_num          0.395907
    delcart_num          0.165649
    user_lv_cd           0.155897
    browse2buy           0.110761
    click_num            0.088284
    browse_num           0.077604
    favor2buy            0.054586
    favor_num            0.047694
    click2buy            0.035203
    sex                  0.022788
    cart2buy             0.011903
    cartnum2buy         -0.003087
    
  buy_num > 4
    cart_num             0.443522
    addcart_num          0.436586
    delcart_num          0.194716
    user_lv_cd           0.171691
    click_num            0.095146
    browse2buy           0.094493
    browse_num           0.088661
    favor2buy            0.051204
    click2buy            0.048258
    favor_num            0.044527
    sex                  0.034346
    cart2buy            -0.013321
    cartnum2buy         -0.017220
'''
def analysis_corrilation():
    df = pd.read_csv(file_tf_fsa_users, index_col='user_id')
    
    df = df[df['buy_num'] > 4]

    df['cart_num'] = df.apply(lambda r: r['addcart_num']-r['delcart_num'], axis=1)
    df['cart_num'].apply(lambda x:0 if x<0 else x)
    df['cartnum2buy'] = df.apply(lambda r: r['buy_num'] / \
            (1 if r['cart_num']==0 else r['cart_num']), axis=1)
    
#    print df.head()
    print df.groupby('buy_num').size()
    
    df['buy_lv'] = df['buy_num'].apply(buy_lv_map)
#    print df.groupby('buy_lv').size()
#    df.groupby('buy_lv').size().plot(kind='barh')
    
#    grp = df.groupby('buy_lv')
#    print grp.sum()
#    print grp.mean()
    
    corr = df.corr()['buy_num'].to_frame('corr')
    corr.sort_values('corr', ascending=False, inplace=True)
    print corr
    
def buy_lv_map(x):
    if x <= 10:
        return x
    elif x <= 20:
        return 15
    elif x <= 30:
        return 25
    elif x <=40:
        return 35
    elif x <= 50:
        return 45
    else:
        return 55
    
analysis_corrilation()







