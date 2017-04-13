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
import datetime, time, gc
import numpy as np
import pandas as pd
from sklearn import linear_model


file_encoding = 'gbk'
file_path = './data/'
output_path = './out/'
default_chunk_size = 100000

file_users = file_path + 'JData_User.csv'
file_products = file_path + 'JData_Product.csv'
file_comments = file_path + 'JData_Comment.csv'

file_actions_02 = file_path + 'JData_Action_201602.csv'
file_actions_03 = file_path + 'JData_Action_201603.csv'
file_actions_04 = file_path + 'JData_Action_201604.csv'
file_actions_arr = [file_actions_02, file_actions_03, file_actions_04]

file_tf_user_actions = output_path + 'tf_user_actions.csv'
file_tf_users = output_path + 'tf_users.csv'
file_tf_product_actions = output_path + 'tf_product_actions.csv'
file_tf_products = output_path + 'tf_products.csv'

file_tf_dd_user_actions = output_path + 'tf_dd_user_actions.csv'
file_tf_dd_users = output_path + 'tf_dd_users.csv'
file_tf_dd_product_actions = output_path + 'tf_dd_product_actions.csv'
file_tf_dd_products = output_path + 'tf_dd_products.csv'




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
file_tf_fsa_users = output_path + 'tf_fsa_users.csv'
file_tf_dd_fsa_users = output_path + 'tf_dd_fsa_users.csv'
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
    
#analysis_corrilation()











file_tf_actions_02 = output_path + 'tf_actions_02.csv'
file_tf_actions_03 = output_path + 'tf_actions_03.csv'
file_tf_actions_04 = output_path + 'tf_actions_04.csv'
file_tf_actions_arr = [file_tf_actions_02, file_tf_actions_03, file_tf_actions_04]

file_tf_actions = output_path + 'tf_actions.csv'
def count_actions():
    start = time.time()
    
    keep_cols = ['user_id','sku_id','time','type']
    for i in range(3):
#        reader = pd.read_csv(file_actions_arr[i], encoding=file_encoding, iterator=True)
#        df = reader.get_chunk(default_chunk_size)
        df = pd.read_csv(file_actions_arr[i], encoding=file_encoding, \
                         infer_datetime_format=True)[keep_cols]
        print df['time'].min()
        print df['time'].max()
        print len(df)
        df['type'] = df['type'].apply(lambda x:1 if x==6 else x)
        df = df.groupby(keep_cols).size().to_frame('count')
        df.reset_index(inplace=True)
        df.sort_values(['user_id', 'time'], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print len(df)
        
        df.to_csv(file_tf_actions_arr[i], index=False)
        
        del df
        gc.collect()
    
    end = time.time(); elapsed = (end - start); start = end;
    print 'count actions cost %ds'%elapsed
#count_actions()

'''
对比分析后，发现三份文件虽然有时间重叠，但没有数据重复
'''
def compare_cross_time():
    i=1
    for f in file_tf_actions_arr:
#        reader = pd.read_csv(f, encoding=file_encoding, iterator=True)
#        df = reader.get_chunk(default_chunk_size)
        df = pd.read_csv(f, encoding=file_encoding, infer_datetime_format=True)
        df['time'] = pd.to_datetime(df['time'])
#        print df.dtypes
        df.sort_values(['time', 'user_id'], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
#        print df['time'].min()
#        print df['time'].max()
        
        start_time = df['time'].min()
        end_time = start_time + np.timedelta64(60, 's')
        df_start = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        print '\n%s - %s'%(start_time, end_time)
#        print df_start
        df_start.to_csv(output_path+'%d.csv'%i, index=False)
        i+=1

        end_time = df['time'].max()
        start_time = end_time - np.timedelta64(60, 's')
        df_end = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        
        print '\n%s - %s'%(start_time, end_time)
#        print df_end
        df_end.to_csv(output_path+'%d.csv'%i, index=False)
        i+=1
        
        del df
        gc.collect()
#compare_cross_time()

def merge_actions():
    start = time.time()
    
    dfs = []
    for f in file_tf_actions_arr:
#        reader = pd.read_csv(f, encoding=file_encoding, iterator=True)
#        df = reader.get_chunk(default_chunk_size)
        df = pd.read_csv(f, encoding=file_encoding, infer_datetime_format=True)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
#    print df.head(20)
    df.sort_values(['user_id', 'time'], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
#    print df.head(20)
    df.to_csv(file_tf_actions, index=False)
    
    del df, dfs
    gc.collect()
    
    end = time.time(); elapsed = (end - start); start = end;
    print 'merge actions cost %ds'%elapsed
#merge_actions()

file_tf_b0_actions = output_path + 'tf_b0_actions.csv'
file_tf_b1_actions = output_path + 'tf_b1_actions.csv'
file_tf_bn_actions = output_path + 'tf_bn_actions.csv'
def split_actions_by_buy_num():
#    reader = pd.read_csv(file_tf_actions, encoding=file_encoding, iterator=True)
#    df = reader.get_chunk(default_chunk_size)
    df = pd.read_csv(file_tf_actions, encoding=file_encoding, infer_datetime_format=True)
    users = pd.read_csv(file_tf_users, encoding=file_encoding, index_col='user_id')
    users_0 = users[users['buy_num']<1]
    users_1 = users[users['buy_num']==1]
    users_n = users[users['buy_num']>1]
    
    df_0 = df[df['user_id'].isin(users_0.index.values)]
    df_1 = df[df['user_id'].isin(users_1.index.values)]
    df_n = df[df['user_id'].isin(users_n.index.values)]
    print df_0.head(20)
    print df_1.head(20)
    print df_n.head(20)
    
    df_0.to_csv(file_tf_b0_actions, index=False)
    df_1.to_csv(file_tf_b1_actions, index=False)
    df_n.to_csv(file_tf_bn_actions, index=False)
    
    del df, users, users_0, users_1, users_n, df_0, df_1, df_n
    gc.collect()
#split_actions_by_buy_num()



def find_lowest_visit_time():
#    reader = pd.read_csv(file_tf_actions, encoding=file_encoding, iterator=True)
#    df = reader.get_chunk(default_chunk_size)
    df = pd.read_csv(file_tf_actions, encoding=file_encoding, infer_datetime_format=True)
    temp = pd.DatetimeIndex(df['time'])
#    df['date'] = temp.date
    df['hour'] = temp.hour
    df['minute'] = temp.minute
#    df['second'] = temp.second
    grp = df.groupby(['hour','minute'])['count'].sum().to_frame('count')
    grp.reset_index(inplace=True)
    print grp
    
    grp.to_csv(output_path+'find_lowest_visit_time.csv', index=False)
    
    del df, temp
    gc.collect()
#find_lowest_visit_time()

def fix_user_id_as_int():
    files = [file_tf_user_actions, file_tf_dd_user_actions,\
             file_tf_actions_02, file_tf_actions_03, file_tf_actions_04, file_tf_actions,\
             file_tf_b0_actions, file_tf_b1_actions, file_tf_bn_actions]

    for f in files:
        df = pd.read_csv(f, encoding=file_encoding)
        df['user_id'] = df['user_id'].astype(int)
        df.to_csv(f, index=False)
        del df
        gc.collect()
#fix_user_id_as_int()

'''
以凌晨四点半为界，将用户行为以天为单位进行分组
'''
file_tf_gbd_b0_actions = output_path + 'tf_gbd_b0_actions.csv'
file_tf_gbd_b1_actions = output_path + 'tf_gbd_b1_actions.csv'
file_tf_gbd_bn_actions = output_path + 'tf_gbd_bn_actions.csv'
file_tf_gbd_actions = output_path + 'tf_gbd_actions.csv'
def group_actions_by_date(in_file, out_file):
#    reader = pd.read_csv(in_file, encoding=file_encoding, iterator=True)
#    df = reader.get_chunk(default_chunk_size)
    df = pd.read_csv(in_file, encoding=file_encoding, infer_datetime_format=True)
    
    df['time'] = pd.to_datetime(df['time'])
    base_datetime = np.datetime64('2016-01-30T16:30')
    df['date_diff'] = df['time'].apply(lambda x:\
                        int(round((x-base_datetime)/np.timedelta64(1, 'D'))))
#    print df[df['date_diff']<3]
    
    grp = df.groupby(['user_id', 'date_diff', 'type'])['count'].sum().to_frame('count')
    grp.reset_index(inplace=True)
    base_date = np.datetime64('2016-01-30')
    grp['date'] = grp['date_diff'].apply(lambda x:base_date+np.timedelta64(x, 'D'))
    print grp

    grp.to_csv(out_file, index=False)
    
    del df, grp
    gc.collect()
#group_actions_by_date(file_tf_b0_actions, file_tf_gbd_b0_actions)
#group_actions_by_date(file_tf_b1_actions, file_tf_gbd_b1_actions)
#group_actions_by_date(file_tf_bn_actions, file_tf_gbd_bn_actions)

def fill_date():
    base_date = np.datetime64('2016-01-30')
    for f in [file_tf_gbd_b0_actions, file_tf_gbd_b1_actions, file_tf_gbd_bn_actions]:
        df = pd.read_csv(f)
#        print df.head()
        df['date'] = df['date_diff'].apply(lambda x:base_date+np.timedelta64(x, 'D'))
        df['date'] = df['date'].apply(lambda x:str(x)[0:10])
        print df.head()
        df.to_csv(f, index=False)
        
        del df
        gc.collect()
#fill_date()

file_tf_buy_actions = output_path + 'tf_buy_actions.csv'
def find_buy_actions():
    src_files = [file_tf_b1_actions, file_tf_bn_actions]
    dfs = []
    base_datetime = np.datetime64('2016-01-30T16:30')
    base_date = np.datetime64('2016-01-30')
    for f in src_files:
        df = pd.read_csv(f)
        df = df[df['type']==4]

        df['time'] = pd.to_datetime(df['time'])
        df['date_diff'] = df['time'].apply(lambda x:\
                            int(round((x-base_datetime)/np.timedelta64(1, 'D'))))
        df['date'] = df['date_diff'].apply(lambda x:base_date+np.timedelta64(x, 'D'))
        df['date'] = df['date'].apply(lambda x:str(x)[0:10])
        print df.head()
        
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(file_tf_buy_actions, index=False)
    
    del df, dfs
    gc.collect()
#find_buy_actions()

















