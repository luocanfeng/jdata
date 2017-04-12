# -*- coding: utf-8 -*-
'''
@data: 2017-04-11
@author: luocanfeng

√ 行为统计
√ 用户+行为
√ 商品+行为

√ 行为去重

TODO 用户行为+时间周期

用户活跃情况：
  统计用户近期的操作情况，以天为纬度统计，则产生6*3*30=540个属性
用户对于产品、品牌的喜好：
  统计用户、产品维度，以及用户、品牌纬度的各种操作总和
  用户在购买该款产品前，需要浏览多少次该产品，经历多长时间；
'''
import datetime, time, gc
import pandas as pd
import numpy as np


file_encoding = 'gbk'
file_path = './data/'
output_path = './out/'

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


def read_actions(filename, file_encoding='gbk', read_columns=None, \
                 deduplicate=False, chunck_read=True, chunck_size=100000):
    start = time.time()
    
    df = None
    if chunck_read:
        reader = pd.read_csv(filename, encoding=file_encoding, iterator=True)
        df = reader.get_chunk(chunck_size)
    else:
        df = pd.read_csv(filename, encoding=file_encoding)

    if deduplicate == True:
        df.drop_duplicates(inplace=True)

    if read_columns != None:
        df = df[read_columns]

    if not chunck_read:
        end = time.time(); elapsed = (end - start); start = end;
        print 'read actions cost %ds'%elapsed

    return df
    
'''
  browse_num(浏览数),
  addcart_num(加购数),
  delcart_num(删购数),
  buy_num(购买数),
  favor_num(收藏数),
  click_num(点击数),
  
  cart2buy(购买加购转化率),
  browse2buy(购买浏览转化率),
  click2buy(购买点击转化率),
  favor2buy(购买收藏转化率)
'''
def group_actions_by(filename, user_or_product, deduplicate=False):
    start = time.time()
    col = user_or_product
    
    df = read_actions(filename, read_columns=[col, "type"], chunck_read=False,\
                      deduplicate=deduplicate)

    r = df.groupby([col,'type']).size().to_frame('count')
    r = r.unstack(level=['type'], fill_value=0)
    r.fillna(0, inplace=True)
    r = r[sorted(r.columns.values)]
    r.columns = ['browse_num','addcart_num','delcart_num',\
                   'buy_num','favor_num','click_num']
    r.reset_index(inplace=True)
    r.sort_values(col, inplace=True)
    print r.head()
    
    del df
    gc.collect()
    
    end = time.time(); elapsed = (end - start); start = end;
    print 'load and group actions cost %ds'%elapsed
    return r

def transform_actions_by(user_or_product, output_file, deduplicate=False):
    col = user_or_product
    df2 = group_actions_by(file_actions_02, col, deduplicate)
    df3 = group_actions_by(file_actions_03, col, deduplicate)
    df4 = group_actions_by(file_actions_04, col, deduplicate)
    df = pd.concat([df2,df3,df4], ignore_index=True)
    df.sort_values(col, inplace=True)
    print df.head()
    df = df.groupby(col).sum()
#    print type(df)
    print df.head()
    
    df['buy_num'].astype(float)
    df['cart2buy'] = df.apply(lambda r: r['buy_num'] / \
            (1 if r['addcart_num']==0 else r['addcart_num']), axis=1)
    df['browse2buy'] = df.apply(lambda r: r['buy_num'] / \
            (1 if r['browse_num']==0 else r['browse_num']), axis=1)
    df['click2buy'] = df.apply(lambda r: r['buy_num'] / \
            (1 if r['click_num']==0 else r['click_num']), axis=1)
    df['favor2buy'] = df.apply(lambda r: r['buy_num'] / \
            (1 if r['favor_num']==0 else r['favor_num']), axis=1)
    df.ix[df['cart2buy'] > 1., 'cart2buy'] = 1.
    df.ix[df['browse2buy'] > 1., 'browse2buy'] = 1.
    df.ix[df['click2buy'] > 1., 'click2buy'] = 1.
    df.ix[df['favor2buy'] > 1., 'favor2buy'] = 1.
    print df.head()
    
    if output_file != None:
        df.to_csv(output_file)
    return df



'''
users_transformed:
  user_id(用户id),
  age(年龄),
  sex(性别),
  user_lv_cd(用户级别),
  user_reg_age_by_day(京东龄),
'''
def transform_users(user_actions, output_file):
    df = pd.read_csv(file_users, encoding=file_encoding, index_col='user_id', \
                     infer_datetime_format=True)
    print '\nUsers Header:'
    print df.head()
    
    '''
    age转数字：
    0：异常值，先保留为0，后续通过计算给其补值
    15岁以下：1
    16-25岁：2
    26-35岁：3
    36-45岁：4
    46-55岁：5
    56岁以上：6
    '''
    df['age'].fillna('-1',inplace=True)
    age_map = {'-1':0,u'15岁以下':1,u'16-25岁':2,u'26-35岁':3,\
               u'36-45岁':4,u'46-55岁':5,u'56岁以上':6}
    df['age'] = df['age'].apply(lambda x: age_map[x])
    
    '''
    sex：0表示男，1表示女，2表示保密
    转换为：1男，-1女，0无；
    后续通过计算为0进行补值
    '''
    df['sex'].fillna(2,inplace=True)
    sex_map = {2:0,0:1,1:-1}
    df['sex'] = df['sex'].apply(lambda x: sex_map[x])
    
    '''根据注册日期计算"京东龄"'''
    df['_date'] = datetime.datetime(2016, 6, 30)
    df['user_reg_tm'] = pd.to_datetime(df['user_reg_tm'])
    df['user_reg_age_by_day'] = (df['_date'] - df['user_reg_tm'])\
                                .apply(lambda x: x/np.timedelta64(1,'D'))
    del df['_date']
    del df['user_reg_tm']
    
    df = df.join(user_actions, how='left')
    print df.head()
    
    df.to_csv(output_file)
    return df



'''
items_transformed:
  sku_id(商品id),
  a1,
  a2,
  a3,
  cate,
  brand,
  
  browse_num,
  addcart_num,
  delcart_num,
  buy_num,
  favor_num,
  click_num,
  
  cart2buy,
  browse2buy,
  click2buy,
  favor2buy,
  
  comment_num(评论数),
  has_bad_comment(是否有差评),
  bad_comment_rate(差评率)
'''
def transform_products(product_actions, output_file):
    products = pd.read_csv(file_products, encoding=file_encoding, index_col='sku_id', \
                           infer_datetime_format=True)
#    print '\nProducts Header:'
#    print df.head()

    comments = pd.read_csv(file_comments, encoding=file_encoding, \
                           infer_datetime_format=True)
    comments.sort_values(['sku_id','dt'], inplace=True)
    grp = comments.groupby('sku_id').last()
    del grp['dt']
    
    print 'products\' length:%d'%len(products)
    print 'products with actions length:%d'%len(product_actions)
    print 'comments\' length:%d'%len(grp)
    
    df = products.join(product_actions, how='outer').join(grp, how='outer')
#    df['comment_num','has_bad_comment'].fillna(-1, inplace=True)
#    df.fillna(0, inplace=True)
    print 'join len:%d'%len(df)
    print df.head()

#    print 'nan len:%d'%len(df[df['cate']==0])
#    print '8 len:%d'%len(df[df['cate']==8])
    
    df.to_csv(output_file)
    return df



def transform(read_transformed_data_from_file=True, deduplicate=False):
    file_ua = file_tf_user_actions
    file_u = file_tf_users
    file_pa = file_tf_product_actions
    file_p = file_tf_products
    
    if deduplicate:
        file_ua = file_tf_dd_user_actions
        file_u = file_tf_dd_users
        file_pa = file_tf_dd_product_actions
        file_p = file_tf_dd_products

    user_actions = None
    if read_transformed_data_from_file:
        user_actions = pd.read_csv(file_ua, encoding=file_encoding, \
                                   index_col='user_id')
        print user_actions.head()
    else:
        user_actions = transform_actions_by('user_id', file_ua, \
                                            deduplicate=deduplicate)
        
    transform_users(user_actions, file_u)
    
    product_actions = None
    if read_transformed_data_from_file:
        product_actions = pd.read_csv(file_pa, encoding=file_encoding, \
                                      index_col='sku_id')
        print product_actions.head()
    else:
        product_actions = transform_actions_by('sku_id', file_pa, \
                                               deduplicate=deduplicate)
    transform_products(product_actions, file_p)

#transform(read_transformed_data_from_file=True, deduplicate=False)
#transform(read_transformed_data_from_file=True, deduplicate=True)









'''
def read_actions(filename):
    start = time.time()
    df = pd.read_csv(filename, encoding=file_encoding, infer_datetime_format=True)
    end = time.time(); elapsed = (end - start); start = end;
    print 'load actions cost %ds'%elapsed
    print df.head()
    
    df.sort_values(['time', 'user_id'], ascending=True, inplace=True)
#    temp = pd.DatetimeIndex(df['time'])
#    df['Month'] = temp.month
#    df['Week'] = temp.week
#    df['Weekday'] = temp.weekday
#    df['Date'] = temp.date
#    df['Time'] = temp.time
#    del df['time']
#    del temp
    gc.collect()
    print df.head()
    end = time.time(); elapsed = (end - start); start = end;
    print 'sort actions cost %ds'%elapsed
    return df

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




