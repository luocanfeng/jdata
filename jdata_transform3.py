# -*- coding: utf-8 -*-
'''
@data: 2017-04-17
@author: luocanfeng
'''
import os,time
import numpy as np
import pandas as pd


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


out_test = output_path + 'test.csv'



if not os.path.exists(output_path):
    os.makedirs(output_path)

    
'''
删除重复数据，删除属于商品的分类与品牌属性，输出；
#同时输出用户与商品的购买次数
'''
out_actions = output_path + 'actions.csv'
#out_user_order_count = output_path + 'user_order_count.csv'
#out_product_order_count = output_path + 'product_order_count.csv'
def transform_actions(test=False):
    start = time.time()
    if os.path.exists(out_actions):
        df = None
        if test:
            reader = pd.read_csv(out_actions, iterator=True)
            df = reader.get_chunk(default_chunk_size)
        else:
            df = pd.read_csv(out_actions)
        print df.head(20)
        return df
        
    dfs = []
    for f in file_actions_arr:
        df = None
        if test:
            reader = pd.read_csv(f, iterator=True)
            df = reader.get_chunk(default_chunk_size)
        else:
            df = pd.read_csv(f)
            
        del df['cate'], df['brand']
        df.drop_duplicates(inplace=True)
        df['user_id'] = df['user_id'].astype(int)
        df.reset_index(drop=True, inplace=True)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.sort_values(['time','user_id','sku_id'], inplace=True)
    
    print df.head(10)
#    print df.dtypes
    if not test:
        df.to_csv(out_actions, index=False)
    
#    order_actions = df[df['type']==4][['user_id','sku_id']]
#    user_order_count = order_actions.groupby('user_id').size().to_frame('count')
#    print user_order_count.head()
#    product_order_count = order_actions.groupby('sku_id').size().to_frame('count')
#    print product_order_count.head()
#    if not test:
#        user_order_count.to_csv(out_user_order_count)
#        product_order_count.to_csv(out_product_order_count)
    
    end = time.time(); elapsed = (end - start); start = end;
    print 'transform actions cost %ds'%elapsed
#    return df, user_order_count, product_order_count
    return
#transform_actions(True)

'''
一个商品在浏览的同时，如果一直都会同时产生相同的点击，则认为是错误的统计，该点击无效
以商品、时间排序，检测出商品浏览连带触发的点击动作的错误统计情况，依此删除连带动作记录
'''
def detect_fault_actions(test=False):
    start = time.time()
    
#    df, user_order_count, product_order_count = transform_actions()
    df = transform_actions(test)
    df = df[df['type'].isin([1,6])]
#    df = df.groupby(['user_id','sku_id','time'], as_index=False)
    df['time'] = pd.to_datetime(df['time'])
    df['timedelta'] = df['time'].diff() / np.timedelta64(1, 's')
    df['timedelta'] = df['timedelta'].fillna(0).astype(int)
    df.sort_values(['sku_id','user_id','time','type'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print df.head(10)
    print len(df)
#    print df.dtypes
    del df['time']
    
    products = pd.read_csv(out_products_all, index_col='sku_id', \
                           usecols=['sku_id']).index.values
    print len(products)
    
    df_len = len(df)
    ba = df[df['type'] == 1]
    pa_loop = {}
    for p in products:
        pa_loop[p] = []
        pb = ba[ba['sku_id'] == p]
        for i1, r1 in pb.iterrows():
            loop = []
            pc = df[min(i1+1, df_len):min(i1+10, df_len)]
            timedelta = 0
            for i2, r2 in pc.iterrows():
                timedelta += r2['timedelta']
                if timedelta > 1 or r2['user_id'] != r1['user_id'] \
                        or r2['type'] != 6 or np.isnan(r2['model_id']):
                    break
                else:
                    loop.append(int(r2['model_id']))
            pa_loop[p].append(','.join(str(n) for n in sorted(loop)))
    
    r = pd.DataFrame(columns=('sku_id', 'model_ids', 'count'))
    i = 0
    for k, v in pa_loop.items():
        if len(v) == 0:
            continue
        loop_dict = {i: v.count(i) for i in v}
#        print '%d: %d'%(k, len(v))
        for k2, v2 in loop_dict.items():
            r.loc[i] = [k, k2, v2]
            i += 1
#            print '    %s    %d'%(k2,v2)
    
    print r.head(20)
    if not test:
        r.to_csv(out_test, index=False)
    
    end = time.time(); elapsed = (end - start); start = end;
    print 'detect fault actions cost %ds'%elapsed
#detect_fault_actions(True)



'''
2、以用户、时间排序，依次分析每个用户在一个时间周期内的活动情况
3、期间可能需要根据商品的销量/用户购买次数进行分类，将中间数据落盘
'''




def test():
    user_actions = pd.read_csv(out_user_actions, index_col='user_id')
    product_actions = pd.read_csv(out_product_actions, index_col='sku_id')
    
    #找出访问量极小的用户/产品
    user_actions['all'] = user_actions.apply(lambda r: \
            sum([r['b'],r['a'],r['d'],r['o'],r['f'],r['c']]), axis=1)
    print user_actions['all'].describe([.01,.02,.03,.05,.1,.25,.5,.75])
    freqence = 6
    lf_users = user_actions[user_actions['all'] <= freqence]
    lf_users_len = len(lf_users)
    lf_users_len_ordered = len(lf_users[lf_users['o'] > 0])
    print '%d / %d = %f'%(lf_users_len_ordered, lf_users_len, \
                          float(lf_users_len_ordered)/lf_users_len)
#test()


