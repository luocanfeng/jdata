# -*- coding: utf-8 -*-
'''
@data: 2017-04-12
@author: luocanfeng
'''
import time, gc
import numpy as np
import pandas as pd
#from sklearn import linear_model


file_path = './data/'
output_path = './out2/'
default_chunk_size = 100000

file_users = file_path + 'JData_User.csv'
file_products = file_path + 'JData_Product.csv'
file_comments = file_path + 'JData_Comment.csv'

file_actions_02 = file_path + 'JData_Action_201602.csv'
file_actions_03 = file_path + 'JData_Action_201603.csv'
file_actions_04 = file_path + 'JData_Action_201604.csv'
file_actions_arr = [file_actions_02, file_actions_03, file_actions_04]


out_users = output_path + 'users.csv'
out_user_actions = output_path + 'user_actions.csv'
out_products_all = output_path + 'products_all.csv'
out_product_actions = output_path + 'product_actions.csv'
out_users_models_to_order = output_path + 'umo.csv'
out_products_models_to_order = output_path + 'pmo.csv'


base_datetime = np.datetime64('2016-01-30T16:30')
base_date = np.datetime64('2016-01-30')


def analysis_user_actions_corrilation():
    df = pd.read_csv(out_user_actions, index_col='user_id')
    cols = ['b','a','d','o','f','c','b2o','c2o','a2o','f2o']
    df = df[cols]
    df.fillna(0, inplace=True)
    print '\n\nUser actions corrilation'
    print df.corr()
    print '\n\no==0'
    print df[df['o'] == 0].corr()
    print '\n\no==1'
    print df[df['o'] == 1].corr()
    print '\n\no>1'
    print df[df['o'] > 1].corr()
def analysis_product_actions_corrilation():
    df = pd.read_csv(out_product_actions, index_col='sku_id')
    df = df[['a1','a2','a3','comment_num','bad_comment_rate',\
             'b','a','d','o','f','c','b2o','c2o','a2o','f2o']]
    df.fillna(0, inplace=True)
    print '\n\nProduct actions corrilation'
    print df.corr()
    print '\n\no==0'
    print df[df['o'] == 0].corr()
    print '\n\no==1'
    print df[df['o'] == 1].corr()
    print '\n\no>1'
    print df[df['o'] > 1].corr()
    print '\n\ncomment_num==0'
    print df[df['comment_num'] == 0].corr()
    print '\n\ncomment_num==1'
    print df[df['comment_num'] == 1].corr()
    print '\n\ncomment_num==2'
    print df[df['comment_num'] == 2].corr()
    print '\n\ncomment_num==3'
    print df[df['comment_num'] == 3].corr()
    print '\n\ncomment_num==4'
    print df[df['comment_num'] == 4].corr()
#analysis_user_actions_corrilation()
#analysis_product_actions_corrilation()

def explore_product_actions():
    df = pd.read_csv(out_product_actions, index_col='sku_id')
    df = df['o'].sort_values(ascending=False)
    print df.head(200)
#explore_product_actions()





file_tf_actions_02 = output_path + 'tf_actions_02.csv'
file_tf_actions_03 = output_path + 'tf_actions_03.csv'
file_tf_actions_04 = output_path + 'tf_actions_04.csv'
file_tf_actions_arr = [file_tf_actions_02, file_tf_actions_03, file_tf_actions_04]

'''点击与浏览的相关性相当大，可以视为同一动作进行合并；同一时刻的相同行为进行合并累计'''
file_tf_actions = output_path + 'tf_actions.csv'
def count_actions():
    start = time.time()
    
    keep_cols = ['user_id','sku_id','time','type']
    for i in range(3):
#        reader = pd.read_csv(file_actions_arr[i], iterator=True)
#        df = reader.get_chunk(default_chunk_size)
        df = pd.read_csv(file_actions_arr[i], \
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

'''三个月份的行为数据进行合并'''
def merge_actions():
    start = time.time()
    
    dfs = []
    for f in file_tf_actions_arr:
#        reader = pd.read_csv(f, iterator=True)
#        df = reader.get_chunk(default_chunk_size)
        df = pd.read_csv(f, infer_datetime_format=True)
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

'''用户行为数据分割成多份文件，没有购买行为的一份，购买一次的一份，购买多次的一份'''
file_tf_b0_actions = output_path + 'tf_b0_actions.csv'
file_tf_b1_actions = output_path + 'tf_b1_actions.csv'
file_tf_bn_actions = output_path + 'tf_bn_actions.csv'
def split_actions_by_buy_num():
#    reader = pd.read_csv(file_tf_actions, encoding=file_encoding, iterator=True)
#    df = reader.get_chunk(default_chunk_size)
    df = pd.read_csv(file_tf_actions, infer_datetime_format=True)
    users = pd.read_csv(file_tf_users, index_col='user_id')
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

'''统计用户行为数据最少的时间点，依此进行“天”的物理分割'''
def find_lowest_visit_time():
#    reader = pd.read_csv(file_tf_actions, iterator=True)
#    df = reader.get_chunk(default_chunk_size)
    df = pd.read_csv(file_tf_actions, infer_datetime_format=True)
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

'''以凌晨四点半为界，将用户行为以天为单位进行分组统计'''
file_tf_gbd_b0_actions = output_path + 'tf_gbd_b0_actions.csv'
file_tf_gbd_b1_actions = output_path + 'tf_gbd_b1_actions.csv'
file_tf_gbd_bn_actions = output_path + 'tf_gbd_bn_actions.csv'
file_tf_gbd_actions = output_path + 'tf_gbd_actions.csv'
def group_actions_by_date(in_file, out_file):
#    reader = pd.read_csv(in_file, encoding=file_encoding, iterator=True)
#    df = reader.get_chunk(default_chunk_size)
    df = pd.read_csv(in_file, infer_datetime_format=True)
    
    df['time'] = pd.to_datetime(df['time'])
    df['date_diff'] = df['time'].apply(lambda x:\
                        int(round((x-base_datetime)/np.timedelta64(1, 'D'))))
#    print df[df['date_diff']<3]
    
    grp = df.groupby(['user_id', 'date_diff', 'type'])['count'].sum().to_frame('count')
    grp.reset_index(inplace=True)
    grp['date'] = grp['date_diff'].apply(lambda x:base_date+np.timedelta64(x, 'D'))
    print grp

    grp.to_csv(out_file, index=False)
    
    del df, grp
    gc.collect()
#group_actions_by_date(file_tf_b0_actions, file_tf_gbd_b0_actions)
#group_actions_by_date(file_tf_b1_actions, file_tf_gbd_b1_actions)
#group_actions_by_date(file_tf_bn_actions, file_tf_gbd_bn_actions)

'''用户以天为单位行为统计分组数据中添加date列'''
def fill_date():
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

'''用户购买行为'''
file_tf_buy_actions = output_path + 'tf_buy_actions.csv'
def find_buy_actions():
    src_files = [file_tf_b1_actions, file_tf_bn_actions]
    dfs = []
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

'''用户购买之前的行为统计'''
test_df = pd.DataFrame([[1,0],[2,0],[3,0],[5,0]], columns=['type','count'])
file_tf_actions_before_buy = output_path + 'tf_actions_before_buy.csv'
def actions_before_buy():
    src_files = [file_tf_b1_actions, file_tf_bn_actions]
    buy_actions = pd.DataFrame()
    for f in src_files:
#        reader = pd.read_csv(f, iterator=True)
#        df = reader.get_chunk(default_chunk_size)
        df = pd.read_csv(f)

        users = df['user_id'].drop_duplicates().values
#        print users
        
        df['time'] = pd.to_datetime(df['time'])
        df['date_diff'] = df['time'].apply(lambda x:\
                            int(round((x-base_datetime)/np.timedelta64(1, 'D'))))
#        df['date'] = df['date_diff'].apply(lambda x:base_date+np.timedelta64(x, 'D'))
#        df['date'] = df['date'].apply(lambda x:str(x)[0:10])
#        print df.head()
        
        for u in users:
            user_actions = df[df['user_id']==u]
            user_actions.reset_index(drop=True, inplace=True)
            
            user_buy_index = user_actions[user_actions['type']==4].index.values

            start = 0
            for i in user_buy_index:
                buy_action = cal_before_actions(user_actions, start, i)
#                print buy_action
                start = i
                buy_actions = buy_actions.append(buy_action)
        
        del df, users
        gc.collect()
    print buy_actions.head()
    buy_actions.to_csv(file_tf_actions_before_buy, index=False)
def cal_before_actions(user_actions, start_index, buy_index):
    buy_action = user_actions.loc[buy_index].to_frame().T
#    print type(buy_action)
#    print buy_action
    
    tmp = user_actions[start_index:buy_index][['type','count']]
    tmp = tmp.append(test_df, ignore_index=True)
    tmp = tmp.groupby('type').sum()
#    print type(tmp.loc[1]['count'])
#    print tmp.loc[1]['count']
    buy_action['before_view'] = tmp.loc[1]['count']
    buy_action['before_addcart'] = tmp.loc[2]['count']
    buy_action['before_delcart'] = tmp.loc[3]['count']
    buy_action['before_favor'] = tmp.loc[5]['count']
#    print buy_action
#    print tmp.head()
    return buy_action
#actions_before_buy()


'''
首先得到所有购买数据（是否需要加购/删购/收藏数据？）
然后按日期滑动计算各种行为在指定时间内的转化情况
    第一纬度：日期刻度（1/2/3/5/7/10/15/21/30天）
    第二纬度：用户、产品
    第三纬度：v,a,d,b,f, v2b,a2b,f2b,v2a,v2f, d, 是否购买

用户、产品聚类
计算用户分类、产品分类（中心点？）间的相关性（购买概率？）


计算用户最有可能购买的几个产品，小于一定概率直接忽略
用户近期对产品有过正向行为则加分，负面行为减分












浏览3/5/8/10次后购买/放弃购买（1/2/3/5/7/10/15/21/30天无购买行为）
加购/删购/收藏后购买/放弃购买



用户多长时间产生一次购买行为（用户消费倾向）
用户各种行为转化为购买行为的比例（随便逛逛/目标明确）
用户从对某款产品产生行为后多久会下单购买（用户时间概念）
用户产生各种行为的时间周期（活跃用户/惰性用户）
没有金额如何判断用户购买力？
用户是否在整个时间段内的行为有较大起伏？以周、十天、两周、月为单位统计用户行为，分析行为波动
用户浏览到加购/收藏的转化率，也是用户习惯的一个纬度数据

为用户进行聚类
给每类用户标注权重？


关注产品后是否会购买，多久会购买
按用户分类分析每类用户关注产品后的购买转化率，多久会购买




同用户在一段时间内产生行为的产品具有一定的相关性
用户的浏览、加购、删购、购买、收藏等操作具有一定的评分作用
统计产品得到的各种行为

对产品进行聚类（产品本身含有分类、品牌、attr等属性）


计算用户分类、产品分类的相关性、购买概率


计算用户最有可能购买的几个产品，小于一定概率直接忽略
用户近期对产品有过正向行为则加分，负面行为减分









计算用户各种行为转化为购买的转化比（含各种时间刻度的提前行为）
计算产品各种时间刻度的行为转化为购买的转化比
根据转化比对用户进行聚类
根据转化比对产品进行聚类
分析不同类用户对不同类产品的
根据用户已有行为数据分析其已有行为的产品是否会购买
计算用户相似度，
'''











