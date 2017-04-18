# -*- coding: utf-8 -*-
'''
@data: 2017-04-15
@author: luocanfeng
'''
import time, gc, os
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

    
    
'''
b(浏览数), c(点击数), a(加购数), d(删购数), f(关注数), o(下单数)
'''
def read_actions(filename, test=False):
    start = time.time()
    
    df = None
    if test:
        reader = pd.read_csv(filename, iterator=True)
        df = reader.get_chunk(default_chunk_size)
    else:
        df = pd.read_csv(filename)

    #actions表中的产品分类、品牌属性
    pfa = df[['sku_id', 'cate', 'brand']]
    pfa.drop_duplicates(inplace=True)

    #按模块ID统计
    model_keep_cols = ['user_id','sku_id','model_id']
    models = df[df['model_id'].notnull()][model_keep_cols]
    models = models.groupby(model_keep_cols).size().to_frame('count')
    models.reset_index(inplace=True)
    for c in models.columns:
        models[c] = models[c].astype(int)
#    mgu = group_by_model_and(models, 'user_id')
#    mgp = group_by_model_and(models, 'sku_id')

    #buy actions
    oas = df[df['type']==4][['user_id','sku_id']]
    oas = oas.groupby(['user_id','sku_id']).size().to_frame('count')
    oas.reset_index(inplace=True)
    
    #删除不必要的字段
    keep_cols = ['user_id','sku_id','type']
    df = df[keep_cols]
    #float -> int
    for c in keep_cols:
        df[c] = df[c].astype(int)
    #同一时刻的相同行为进行合并累计
    df = df.groupby(keep_cols).size().to_frame('count')
    df.reset_index(inplace=True)
    agu = group_actions_by(df, 'user_id')
    agp = group_actions_by(df, 'sku_id')
    
    del df
    gc.collect()
    
    if not test:
        end = time.time(); elapsed = (end - start); start = end;
        print 'read actions cost %ds'%elapsed
#    return agu, agp, pfa, mgu, mgp
    return agu, agp, pfa, models, oas
    
def group_actions_by(actions, col):
    r = actions.groupby([col,'type'])['count'].sum()
    r = r.unstack(level=['type'])
    r.fillna(0, inplace=True)
    r = r[sorted(r.columns.values)]
    r.columns = ['b', 'a', 'd', 'o', 'f', 'c']
    r.reset_index(inplace=True)
    r.sort_values(col, inplace=True)
    r.reset_index(drop=True, inplace=True)
    for c in r.columns:
        r[c] = r[c].astype(int)
    print '\nGrouped actions:'
    print r.head()
    return r

def group_by_model_and(actions, col):
    r = actions.groupby([col,'model_id'])['count'].sum()
    r = r.unstack(level=['model_id'])
    r.fillna(0, inplace=True)
    r = r[sorted(r.columns.values)]
    r.reset_index(inplace=True)
    r.sort_values(col, inplace=True)
    r.reset_index(drop=True, inplace=True)
    for c in r.columns:
        r[c] = r[c].astype(int)
    print '\nGrouped by %s and model_id:'%col
    print r.head()
    return r

'''
b2o(浏览下单转化率), c2o(点击下单转化率), a2o(加购下单转化率), f2o(关注下单转化率)
'''
def transform_actions(test=False):
#    agu2, agp2, pfa2, mgu2, mgp2 = read_actions(file_actions_02, test)
#    agu3, agp3, pfa3, mgu3, mgp3 = read_actions(file_actions_03, test)
#    agu4, agp4, pfa4, mgu4, mgp4 = read_actions(file_actions_04, test)
    agu2, agp2, pfa2, models2, oas2 = read_actions(file_actions_02, test)
    agu3, agp3, pfa3, models3, oas3 = read_actions(file_actions_03, test)
    agu4, agp4, pfa4, models4, oas4 = read_actions(file_actions_04, test)
    
    agu = pd.concat([agu2, agu3, agu4], ignore_index=True)
    agu.sort_values('user_id', inplace=True)
    agu = agu.groupby('user_id').sum()
    for c in ['b', 'c', 'a', 'f']:
        agu['%s2o'%c] = agu.apply(lambda r: div(r['o'], r[c]), axis=1)
    for c in ['b', 'a', 'd', 'o', 'f', 'c']:
        agu[c] = agu[c].astype(int)
    print '\nActions groupby user:'
    print agu.head()
    
    agp = pd.concat([agp2, agp3, agp4], ignore_index=True)
    agp.sort_values('sku_id', inplace=True)
    agp = agp.groupby('sku_id').sum()
    for c in ['b', 'c', 'a', 'f']:
        agp['%s2o'%c] = agp.apply(lambda r: div(r['o'], r[c]), axis=1)
    for c in ['b', 'a', 'd', 'o', 'f', 'c']:
        agp[c] = agp[c].astype(int)
    print '\nActions groupby product:'
    print agp.head()
    
    pfa = pd.concat([pfa2, pfa3, pfa4], ignore_index=True)
    pfa.drop_duplicates(inplace=True)
    pfa.set_index('sku_id', inplace=True)
    
#    mgu = pd.concat([mgu2, mgu3, mgu4], ignore_index=True)
#    mgu.fillna(0, inplace=True)
#    mgu.sort_values('user_id', inplace=True)
#    mgu = mgu.groupby('user_id').sum()
#    for c in mgu.columns:
#        mgu[c] = mgu[c].astype(int)
#    print '\nModels groupby user:'
#    print mgu.head()
#    
#    mgp = pd.concat([mgp2, mgp3, mgp4], ignore_index=True)
#    mgp.fillna(0, inplace=True)
#    mgp.sort_values('sku_id', inplace=True)
#    mgp = mgp.groupby('sku_id').sum()
#    for c in mgp.columns:
#        mgp[c] = mgp[c].astype(int)
#    print '\nModels groupby product:'
#    print mgp.head()
    
    models = pd.concat([models2, models3, models4], ignore_index=True)
    models = models.groupby(['user_id','sku_id','model_id']).sum()
    models.reset_index(inplace=True)
    for c in models.columns:
        models[c] = models[c].astype(int)
    print '\nModels:'
    print models.head()
    
    oas = pd.concat([oas2, oas3, oas4], ignore_index=True)
    oas = oas.groupby(['user_id','sku_id']).sum()
    oas.reset_index(inplace=True)
    for c in oas.columns:
        oas[c] = oas[c].astype(int)
    print '\nBuy actions:'
    print oas.head()
    
#    return agu, agp, pfa, mgu, mgp
    return agu, agp, pfa, models, oas
    
def div(d1, d2):
    return 0. if d1==0 else (1. if d2==0 else min(float(d1)/d2, 1.))
    


out_users = output_path + 'users.csv'
out_user_actions = output_path + 'user_actions.csv'
out_products_all = output_path + 'products_all.csv'
out_product_actions = output_path + 'product_actions.csv'
out_users_models_to_order = output_path + 'umo.csv'
out_products_models_to_order = output_path + 'pmo.csv'
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
        return
        
#    agu, agp, pfa, mgu, mgp = transform_actions(test)
    agu, agp, pfa, models, oas = transform_actions(test)
    
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
    
#    user_actions = users.join(agu, how='outer').join(mgu, how='outer')
    user_actions = users.join(agu, how='outer')
    user_actions.fillna(0, inplace=True)
    for c in user_actions.columns:
        if '2o' not in str(c):
            user_actions[c] = user_actions[c].astype(int)
    print user_actions.head()
#    print user_actions.dtypes
    
    if not test:
        user_actions.to_csv(out_user_actions)
        
    comments = pd.read_csv(file_comments, infer_datetime_format=True)
    comments.sort_values(['sku_id','dt'], inplace=True)
    comments = comments.groupby('sku_id').last()
    del comments['dt']
    comments['comment_num'] = comments['comment_num'].astype(int)
    comments['has_bad_comment'] = comments['has_bad_comment'].astype(int)
    print comments.head()
#    print comments.dtypes
    
    products = pd.read_csv(file_products, index_col='sku_id')
    pfa.rename(columns={'cate': '_cate', 'brand': '_brand'}, inplace=True)
    products_all = pd.concat([products, pfa], axis=1)
    products_all.fillna(-1, inplace=True)
    products_all['cate'] = products_all.apply(lambda r: \
            r['_cate'] if r['_cate']!=-1 else r['cate'], axis=1)
    products_all['brand'] = products_all.apply(lambda r: \
            r['_brand'] if r['_brand']!=-1 else r['brand'], axis=1)
    del products_all['_cate'], products_all['_brand']
    products_all = products_all.join(comments, how='outer')
    products_all.fillna(0, inplace=True)
    for c in products_all.columns:
        if c != 'bad_comment_rate':
            products_all[c] = products_all[c].astype(int)
    print products_all.head()
#    print products_all.dtypes
    if not test:
        products_all.to_csv(out_products_all)

#    product_actions = products_all.join(agp, how='outer').join(mgp, how='outer')
    product_actions = products_all.join(agp, how='outer')
    for c in ['a1','a2','a3']:
        product_actions[c].fillna(-1, inplace=True)
    product_actions.fillna(0, inplace=True)
    for c in product_actions.columns:
        if '2o' not in str(c) and c != 'bad_comment_rate':
            product_actions[c] = product_actions[c].astype(int)
    print product_actions.head()
#    print product_actions.dtypes

    if not test:
        product_actions.to_csv(out_product_actions)
    
    models = models[['user_id','sku_id','model_id','count']]
    models.columns = ['user_id','sku_id','model_id','mcount']
    oas = oas[['user_id','sku_id','count']]
    oas.columns = ['user_id','sku_id','ocount']
    mo = pd.merge(models, oas, on=['user_id', 'sku_id'], how='left')
    mo.sort_values(['user_id','sku_id','model_id'], inplace=True)
    mo.reset_index(drop=True, inplace=True)
    mo['ocount'].fillna(0, inplace=True)
    mo['ocount'] = mo['ocount'].astype(int)
    print mo.head()
#    print mo.dtypes
    
    umo = mo[['user_id','model_id','mcount','ocount']]
    umo = umo.groupby(['user_id','model_id']).sum()
    umo.reset_index(inplace=True)
    umo.sort_values(['user_id','model_id'], inplace=True)
    umo.reset_index(drop=True, inplace=True)
    umo['m2o'] = umo.apply(lambda row: div(row['ocount'], row['mcount']), axis=1)
    print umo.head()
#    print umo.dtypes
    if not test:
        umo.to_csv(out_users_models_to_order, index=False)
    
    pmo = mo[['sku_id','model_id','mcount','ocount']]
    pmo = pmo.groupby(['sku_id','model_id']).sum()
    pmo.reset_index(inplace=True)
    pmo.sort_values(['sku_id','model_id'], inplace=True)
    pmo.reset_index(drop=True, inplace=True)
    pmo['m2o'] = pmo.apply(lambda row: div(row['ocount'], row['mcount']), axis=1)
    print pmo.head()
#    print pmo.dtypes
    if not test:
        pmo.to_csv(out_products_models_to_order, index=False)
transform(test=False)


