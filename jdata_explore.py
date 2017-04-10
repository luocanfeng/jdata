# -*- coding: utf-8 -*-
'''
@data: 2017-03-29
@author: luocanfeng
'''
import datetime, time, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn import linear_model


plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


file_encoding = 'gbk'
file_path = './data/'
out_file_path = './out/'

file_users = file_path + 'JData_User.csv'
file_products = file_path + 'JData_Product.csv'
file_comments = file_path + 'JData_Comment.csv'

file_actions_02 = file_path + 'JData_Action_201602.csv'
file_actions_03 = file_path + 'JData_Action_201603.csv'
file_actions_04 = file_path + 'JData_Action_201604.csv'
file_actions_arr = [file_actions_02, file_actions_03, file_actions_04]


gs = gridspec.GridSpec(2, 3)


'''
explore users
columns: user_id, age, sex, user_lv_cd, user_reg_tm
sex: 0表示男，1表示女，2表示保密

用户年龄：转换为数字（10、20、30、40、50、60）
'''
def explore_users():
    df = pd.read_csv(file_users, encoding=file_encoding, index_col='user_id', \
                     infer_datetime_format=True)
    print '\nUsers\'s length:%d'%len(df)
    print '\nUsers Header:'
    print df.head()
    print '\nUsers age distribution:'
    print df.groupby('age').size()
    df.groupby('age').size().plot(kind='bar', ax=plt.subplot(gs[0]))
    print '\nUsers sex distribution:'
    print df.groupby('sex').size()
    df.groupby('sex').size().plot(kind='bar', ax=plt.subplot(gs[1]))
    print '\nUsers level distribution:'
    print df.groupby('user_lv_cd').size()
    df.groupby('user_lv_cd').size().plot(kind='bar', ax=plt.subplot(gs[2]))
    
    #并没有NaN数据，为啥会报KeyError: nan？
    df['age'].fillna('-1',inplace=True)
    age_map = {'-1':0,u'15岁以下':10,u'16-25岁':20,u'26-35岁':30,\
               u'36-45岁':40,u'46-55岁':50,u'56岁以上':60}
    df['age'] = df['age'].apply(lambda x: age_map[x])
    
    df['sex'].fillna(2,inplace=True)
    sex_map = {2:0,0:1,1:-1}
    df['sex'] = df['sex'].apply(lambda x: sex_map[x])
    
    #根据注册日期计算"京东龄"
    df['_date'] = datetime.datetime(2016, 4, 16)
    df['user_reg_tm'] = pd.to_datetime(df['user_reg_tm'])
    df['user_reg_age_by_day'] = (df['_date'] - df['user_reg_tm'])\
                                .apply(lambda x: x/np.timedelta64(1,'D'))
    del df['_date']
    
    print df.head()
    df.to_csv(out_file_path + 'users.csv')
    return df
#explore_users()


'''
explore products
columns: sku_id, a1, a2, a3, cate, brand
'''
def explore_products():
    df = pd.read_csv(file_products, encoding=file_encoding, index_col='sku_id')
    print '\nProducts\'s length:%d'%len(df)
    print '\nProducts Header:'
    print df.head()
    print '\nProducts a1 distribution:'
    print df.groupby('a1').size()
    df.groupby('a1').size().plot(kind='bar', ax=plt.subplot(gs[3]))
    print '\nProducts a2 distribution:'
    print df.groupby('a2').size()
    df.groupby('a2').size().plot(kind='bar', ax=plt.subplot(gs[4]))
    print '\nProducts a3 distribution:'
    print df.groupby('a3').size()
    df.groupby('a3').size().plot(kind='bar', ax=plt.subplot(gs[5]))
    print '\nProducts brand distribution:'
    print df.groupby('brand').size()
    #df.groupby('brand').size().plot(kind='bar', ax=plt.subplot(gs[6]))


'''
explore comments
columns: dt, sku_id, comment_num, has_bad_comment, bad_comment_rate
comment_num：累计评论数分段
  0表示无评论，1表示有1条评论，2表示有2-10条评论，3表示有11-50条评论，4表示大于50条评论
'''
def explore_comments():
    df = pd.read_csv(file_comments, encoding=file_encoding, index_col=['dt','sku_id'])
    print '\nComments\'s length:%d'%len(df)
    print '\nComments Header:'
    print df.head()
    #df.groupby('sku_id')


'''
explore actions
columns: user_id, sku_id, time, model_id, type, cate, brand
type：1浏览,2加购,3删购,4下单,5关注,6点击
'''
def read_actions(file_name):
    start = time.time()
    df = pd.read_csv(file_name, encoding=file_encoding, infer_datetime_format=True)
    end = time.time(); elapsed = (end - start); start = end;
    print 'load actions cost %ds'%elapsed
    
    #df = df[df.type!=6]
    df.sort_values(['time', 'user_id'], ascending=True, inplace=True)
    temp = pd.DatetimeIndex(df['time'])
    df['Month'] = temp.month
#    df['Week'] = temp.week
#    df['Weekday'] = temp.weekday
    df['Date'] = temp.date
#    df['Time'] = temp.time
    del df['time']
    del temp
    gc.collect()
    print df.head()
    end = time.time(); elapsed = (end - start); start = end;
    print 'sort actions cost %ds'%elapsed
    return df

def group_actions_by_user_and_type(df):
    start = time.time()
    grp = df.groupby(['user_id','type']).size().to_frame('count')\
            .unstack(level=['type'], fill_value=0)
    print grp
    end = time.time(); elapsed = (end - start); start = end;
    print 'cost %ds'%elapsed
    return grp

def group_actions_by_date_and_type(df):
    start = time.time()
    grp = df.groupby(['Date','type']).size().to_frame('count')\
            .unstack(level=['type'], fill_value=0)
    print grp
    end = time.time(); elapsed = (end - start); start = end;
    print 'cost %ds'%elapsed
    return grp

def group_actions_by_month_and_type(df):
    start = time.time()
    grp = df.groupby(['Month','type']).size().to_frame('count')\
            .unstack(level=['type'], fill_value=0)
    print grp
    end = time.time(); elapsed = (end - start); start = end;
    print 'cost %ds'%elapsed
    return grp

'''
用户活跃情况：
  统计用户近期的操作情况，以天为纬度统计，则产生6*3*30=540个属性
  可以统计出用户浏览、点击、加购、下单的频率
  浏览周期、点击周期、加购周期、下单周期
用户对于产品、品牌的喜好：
  统计用户、产品维度，以及用户、品牌纬度的各种操作总和
用户购买力：购买次数
用户冲动指数：
  用户需要经过多少次浏览、对比，才会下单；
  用户在购买该款产品前，需要浏览多少次该产品，经历多长时间；
'''
def explore_actions():
#    for f in file_actions_arr:
#        df = read_actions(f)
#        group_actions_by_date_and_type(df)
#        del df
#        gc.collect()
    df = read_actions(file_actions_02)
    
    grp = group_actions_by_user_and_type(df)
    grp.to_csv(out_file_path + 'group_actions_by_user_and_type_02.csv')
    
    grp = group_actions_by_date_and_type(df)
    grp.to_csv(out_file_path + 'group_actions_by_date_and_type_02.csv')
    
#    grp = group_actions_by_month_and_type(df)
#    grp.to_csv(out_file_path + 'group_actions_by_month_and_type_02.csv')
    
    del grp
    del df
    gc.collect()


def explore():
    print '===================='
    explore_users()
    print '===================='
    explore_products()
    print '===================='
    explore_comments()
    print '===================='
    explore_actions()
#explore()
#gc.collect()


# test
def concate_user_and_action():
    file_02 = out_file_path + 'group_actions_by_user_and_type_m02.csv'
    file_03 = out_file_path + 'group_actions_by_user_and_type_m03.csv'
    file_04 = out_file_path + 'group_actions_by_user_and_type_m04.csv'
    
    df2 = pd.read_csv(file_02, index_col='user_id').fillna(0)
    df3 = pd.read_csv(file_03, index_col='user_id').fillna(0)
    df4 = pd.read_csv(file_04, index_col='user_id').fillna(0)
    df2.sort_index(axis=1, inplace=True)
    df3.sort_index(axis=1, inplace=True)
    df4.sort_index(axis=1, inplace=True)
    
    df = df2.add(df3, fill_value=0).add(df4, fill_value=0)
    
#    figure = plt.figure(figsize=(8, 8), dpi=200)
#    ax = plt.subplot(1, 1, 1)
#    ax.ticklabel_format(style='plain', axis='both')
#    ax.set_title(u'相关性')
#    ax.scatter(df['t1'], df['t6'], color='blue', marker='x')
#    figure.show()
    
#    df.sort_values(by='t4', inplace=True)
#    df = df[df['t4']>0]
#    df['t4'].plot.hist(bins=50)
#    df['rate'] = df.t4 / (df.t1 + df.t6)
    
    users = explore_users()
    df = df.join(users, how='left')
    del df['user_reg_tm']
    print df
    
    del df2, df3, df4, users
    gc.collect()
    return df
df = concate_user_and_action()

def process_unknow_age(df):
    sample = df[df.age!=0]
    msk = np.random.rand(len(sample)) < 0.8
    train = sample[msk]
    test = sample[~msk]
    print len(train)
    print len(test)
    
    y_train = train.age
    x_train = train.drop('age', axis=1)
    y_test = test.age
    x_test = test.drop('age', axis=1)
    
    clf = linear_model.LinearRegression()
    clf.fit(x_train, y_train)
    print clf.coef_
    
    y_predict = clf.predict(x_test)
    df = pd.DataFrame([y_test,y_predict], columns=['test','predict'])
    print 'y_test\n'
    print y_test
    print 'y_predict\n'
    print y_predict
    print 'df\n'
    print df
    df.tp = (df.predict - df.test).apply(lambda x:(x>=-5 or x<5))
    print df.tp
    print 'tp=%d/%d'%((df.tp==True).size, df.size)
    
process_unknow_age(df)

def predict_order(df):
    msk = np.random.rand(len(df)) < 0.8
    train = df[df.age!=0]
    test = df[~msk]
    print len(train)
    print len(test)
    
    y_train = train.t4
    x_train = train.drop('t4', axis=1)
    print y_train
    print x_train
#predict_order(df)


def test():
#    reader = pd.read_csv(actions_file, encoding=file_encoding, iterator=True)
#    df = reader.get_chunk(10000)
    return

