# -*- coding: utf-8 -*-
'''
@data: 2017-04-15
@author: luocanfeng
'''
import datetime, time, gc, os
import pandas as pd
import numpy as np


gbk = 'gbk'
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



if not os.path.exists(output_path):
    os.makedirs(output_path)

    
    
def read_actions(filename, test=False):
    start = time.time()
    
    df = None
    if test:
        reader = pd.read_csv(filename, iterator=True)
        df = reader.get_chunk(default_chunk_size)
    else:
        df = pd.read_csv(filename)

    #actions表中的产品分类、品牌属性
    products_from_actions = df[['sku_id', 'cate', 'brand']]
    products_from_actions.drop_duplicates(inplace=True)

    #删除不必要的字段
    keep_cols = ['user_id','sku_id','time','type']
    df = df[keep_cols]
    #用户ID float -> int
    df['user_id'] = df['user_id'].astype(int)
    #点击与浏览相关性相当高，可以合并
    df['type'] = df['type'].apply(lambda x:1 if x==6 else x)
    #同一时刻的相同行为进行合并累计
    df = df.groupby(keep_cols).size().to_frame('count')
    df.reset_index(inplace=True)
#    df.sort_values(['user_id', 'time'], ascending=True, inplace=True)
#    df.reset_index(drop=True, inplace=True)
    
    if not test:
        end = time.time(); elapsed = (end - start); start = end;
        print 'read actions cost %ds'%elapsed

    return df, products_from_actions
    
'''
view_num(浏览/点击数), addcart_num(加购数), delcart_num(删购数), 
buy_num(购买数), favor_num(收藏数),
'''
def group_actions_by_user_and_type(filename, test=False):
    start = time.time()
    
    df, products_from_actions = read_actions(filename, test=test)

    actions_groupby_user = group_actions_by(df, 'user_id')
    actions_groupby_product = group_actions_by(df, 'sku_id')
    
    
    del df
    gc.collect()
    
    end = time.time(); elapsed = (end - start); start = end;
    print 'load and group actions cost %ds'%elapsed
    return actions_groupby_user, actions_groupby_product, products_from_actions
    
def group_actions_by(actions, col):
    r = actions.groupby([col,'type'])['count'].sum()
    r = r.unstack(level=['type'])
    r.fillna(0, inplace=True)
    r = r[sorted(r.columns.values)]
    r.columns = ['view_num','addcart_num','delcart_num','buy_num','favor_num']
    r.reset_index(inplace=True)
    r.sort_values(col, inplace=True)
    print '\nGrouped actions:'
    print r.head()
    return r

'''
cart2buy(购买加购转化率), view2buy(购买浏览转化率), favor2buy(购买收藏转化率)
'''
def transform_actions(test=False):
#    if output_files and os.path.exists(output_files[0]):
#        agu, agp, pfa = None, None, None
#        if test:
#            reader_u = pd.read_csv(output_files[0], iterator=True)
#            agu = reader_u.get_chunk(default_chunk_size)
#            reader_p = pd.read_csv(output_files[1], iterator=True)
#            agp = reader_p.get_chunk(default_chunk_size)
#        else:
#            agu = pd.read_csv(output_files[0])
#            agp = pd.read_csv(output_files[1])
#        pfa = pd.read_csv(output_files[2])
#        print '\nActions groupby user:'
#        print agu.head()
#        print '\nActions groupby product:'
#        print agp.head()
#        print '\nProducts from actions:'
#        print pfa.head()
#        return agu, agp, pfa
        
    agu2,agp2,pfa2 = group_actions_by_user_and_type(file_actions_02, test)
    agu3,agp3,pfa3 = group_actions_by_user_and_type(file_actions_03, test)
    agu4,agp4,pfa4 = group_actions_by_user_and_type(file_actions_04, test)
    
    agu = pd.concat([agu2,agu3,agu4], ignore_index=True)
    agu.sort_values('user_id', inplace=True)
    agu = agu.groupby('user_id').sum()
    agu['click2buy'] = agu.apply(lambda r: div(r['buy_num'], r['view_num']), axis=1)
    agu['cart2buy'] = agu.apply(lambda r: div(r['buy_num'], r['addcart_num']), axis=1)
    agu['favor2buy'] = agu.apply(lambda r: div(r['buy_num'], r['favor_num']), axis=1)
    print '\nActions groupby user:'
    print agu.head()
#    if not test:
#        agu.to_csv(output_files[0])
    
    agp = pd.concat([agp2,agp3,agp4], ignore_index=True)
    agp.sort_values('sku_id', inplace=True)
    agp = agp.groupby('sku_id').sum()
    agp['view2buy'] = agp.apply(lambda r: div(r['buy_num'], r['view_num']), axis=1)
    agp['cart2buy'] = agp.apply(lambda r: div(r['buy_num'], r['addcart_num']), axis=1)
    agp['favor2buy'] = agp.apply(lambda r: div(r['buy_num'], r['favor_num']), axis=1)
    print '\nActions groupby user:'
    print agp.head()
#    if not test:
#        agp.to_csv(output_files[1])
    
    pfa = pd.concat([pfa2,pfa3,pfa4], ignore_index=True)
    pfa.drop_duplicates(inplace=True)
    pfa.set_index('sku_id', inplace=True)
#    if not test:
#        pfa.to_csv(output_files[2])
        
    return agu, agp, pfa
    
def div(d1, d2):
    r = float(d1) / (1 if d2==0 else d2)
    return 1 if r>1 else r
    


out_users = output_path + 'users.csv'
out_user_actions = output_path + 'user_actions.csv'
out_products_all = output_path + 'products_all.csv'
out_product_actions = output_path + 'product_actions.csv'
def transform(test=False):
    if os.path.exists(out_user_actions):
        user_actions, product_actions = None, None
        if test:
            reader_u = pd.read_csv(out_user_actions, iterator=True)
            user_actions = reader_u.get_chunk(default_chunk_size)
            reader_p = pd.read_csv(out_product_actions, iterator=True)
            product_actions = reader_p.get_chunk(default_chunk_size)
        else:
            user_actions = pd.read_csv(out_user_actions)
            product_actions = pd.read_csv(out_product_actions)
        print '\nUser actions:'
        print user_actions.head()
        print '\nProduct Actions:'
        print product_actions.head()
        return user_actions, product_actions
        
    agu, agp, pfa = transform_actions(test)
    
    users = pd.read_csv(file_users, encoding=gbk, index_col='user_id')
    
    '''
    age转数字：
    0：异常值，15岁以下：1，16-25岁：2，26-35岁：3，36-45岁：4，46-55岁：5，56岁以上：6
    '''
    users['age'].fillna('-1', inplace=True)
    age_map = {'-1':0,u'15岁以下':1,u'16-25岁':2,u'26-35岁':3,\
               u'36-45岁':4,u'46-55岁':5,u'56岁以上':6}
    users['age'] = users['age'].apply(lambda x: age_map[x])
    
    '''
    sex：0表示男，1表示女，2表示保密
    转换为：1男，-1女，0无；后续通过计算为0进行补值
    '''
    users['sex'].fillna(2, inplace=True)
    sex_map = {2:0, 0:1, 1:-1}
    users['sex'] = users['sex'].apply(lambda x: sex_map[x])
    
    '''根据注册日期计算"京东龄"'''
    users['_date'] = np.datetime64('2016-06-30')
    users['user_reg_tm'] = pd.to_datetime(users['user_reg_tm'])
    users['user_reg_age_by_day'] = (users['_date'] - users['user_reg_tm'])\
            .apply(lambda x: x/np.timedelta64(1,'D'))
    users['user_reg_age_by_day'].fillna(0, inplace=True)
    users['user_reg_age_by_day'] = users['user_reg_age_by_day'].astype(int)
    del users['_date'], users['user_reg_tm']

    print users.head()
#    print users.dtypes
    if not test:
        users.to_csv(out_users)
    
    user_actions = users.join(agu, how='left')
    user_actions.fillna(0, inplace=True)
    for col in ['view_num','addcart_num','delcart_num','buy_num','favor_num']:
        user_actions[col] = user_actions[col].astype(int)
    print user_actions.head()
#    print user_actions.dtypes
    
    if not test:
        user_actions.to_csv(out_user_actions)
        
    products = pd.read_csv(file_products, index_col='sku_id')
    pfa.rename(columns={'cate': '_cate', 'brand': '_brand'}, inplace=True)
    products_all = pd.concat([products, pfa], axis=1)
    products_all.fillna(-1, inplace=True)
    products_all['cate'] = products_all.apply(lambda r: \
            r['_cate'] if r['_cate']!=-1 else r['cate'], axis=1)
    products_all['brand'] = products_all.apply(lambda r: \
            r['_brand'] if r['_brand']!=-1 else r['brand'], axis=1)
    for col in products_all.columns:
        products_all[col] = products_all[col].astype(int)
    del products_all['_cate'], products_all['_brand']
    print products_all.head()
#    print products_all.dtypes
    if not test:
        products_all.to_csv(out_products_all)

    comments = pd.read_csv(file_comments, infer_datetime_format=True)
    comments.sort_values(['sku_id','dt'], inplace=True)
    comments = comments.groupby('sku_id').last()
    del comments['dt']
    comments['comment_num'] = comments['comment_num'].astype(int)
    comments['has_bad_comment'] = comments['has_bad_comment'].astype(int)
    print comments.head()
#    print comments.dtypes
    
    product_actions = products_all.join(comments, how='outer').join(agp, how='outer')
    for col in ['a1','a2','a3']:
        product_actions[col].fillna(-1, inplace=True)
    product_actions.fillna(0, inplace=True)
    for col in ['a1','a2','a3','cate','brand','comment_num','has_bad_comment',\
                'view_num','addcart_num','delcart_num','buy_num','favor_num']:
        product_actions[col] = product_actions[col].astype(int)
    print product_actions.head()
#    print product_actions.dtypes

    if not test:
        product_actions.to_csv(out_product_actions)
    
    return user_actions, product_actions
transform(test=False)


