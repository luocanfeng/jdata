# -*- coding: utf-8 -*-
'''
@data: 2017-04-17
@author: luocanfeng
'''
import os,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


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
    return df
#transform_actions(True)



#TODO threshold value
def find_exception_users(threshold=500):
    user_actions = pd.read_csv(out_user_actions, index_col='user_id')
    print user_actions.head(10)
    print len(user_actions)
    
    exception_users = []

    mask = user_actions['b']==0
    temp = user_actions[mask]
    print 'o>0/b=0: %d / %d'%(len(temp[temp['o']>0]), len(temp))
    exception_users.extend(temp.index.values)

    # remove users with 0 browses
    user_actions = user_actions[~mask]
    print len(user_actions)

    mask = ((user_actions['b'] + user_actions['c'])>=threshold) \
            & (user_actions['o'])==0 & (user_actions['user_reg_age_by_day']>90)
    temp = user_actions[mask]
#    temp.sort_values('user_reg_age_by_day', inplace=True)
    print temp.head(10)
    print 'zero order users:%d'%len(temp)
    exception_users.extend(temp.index.values)
    
    # remove users with zero order
    user_actions = user_actions[~mask]
    print len(user_actions)

#    temp = user_actions[user_actions['c']==0]
#    print 'o>0/c=0: %d / %d'%(len(temp[temp['o']>0]), len(temp))
#    temp = user_actions[user_actions['a']==0]
#    print 'o>0/a=0: %d / %d'%(len(temp[temp['o']>0]), len(temp))
#    temp = user_actions[user_actions['f']==0]
#    print 'o>0/f=0: %d / %d'%(len(temp[temp['o']>0]), len(temp))
    
#    grp = user_actions.groupby('b').size().to_frame('count')
#    grp.hist(column='count', bins=100)
#    grp = user_actions.groupby('c').size().to_frame('count')
#    grp.hist(column='count', bins=100)
    
#    mask = ((user_actions['b'] + user_actions['c'])>1000) & (user_actions['o']<2)
#    temp = user_actions[mask]
##    temp.sort_values('user_reg_age_by_day', inplace=True)
#    print temp.head(50)
#    print 'low order users:%d'%len(temp)
    
#    print sorted(exception_users)
    return sorted(exception_users)
#find_exception_users()



def remove_exception_users_actions(threshold=500, test=False):
    outfile = output_path + 'actions_%d.csv'%threshold
    start = time.time()
    if os.path.exists(outfile):
        df = None
        if test:
            reader = pd.read_csv(outfile, iterator=True)
            df = reader.get_chunk(default_chunk_size)
        else:
            df = pd.read_csv(outfile)
        return df
        
    df = transform_actions(test)
    print len(df)
    exception_users = find_exception_users(threshold)
    df = df[~df['user_id'].isin(exception_users)]
    print len(df)
    if not test:
        df.to_csv(outfile, index=False)
        
    end = time.time(); elapsed = (end - start); start = end;
    print 'transform actions cost %ds'%elapsed
    return df
#remove_exception_users_actions(500, False)
#remove_exception_users_actions(100, False)
#remove_exception_users_actions(0, False)



#def find_low_percent_conversion_user(bc_threshold, bc_pc, af_pc):
def find_low_percent_conversion_user(bc_threshold, o_threshold, bc_pc=.005, af_pc=.01):
    user_actions = pd.read_csv(out_user_actions, index_col='user_id')
    print user_actions.head(10)
    print len(user_actions)
    
    exception_users = []

    mask = ((user_actions['b'] + user_actions['c'])>=bc_threshold) \
            & (user_actions['user_reg_age_by_day']>90)
    mask = mask & ((((user_actions['b2o'] + user_actions['c2o'])<bc_pc) \
            & (user_actions['a2o']<af_pc) & (user_actions['f2o']<af_pc)) \
            | (user_actions['o']<=o_threshold))
    temp = user_actions[mask]
    exception_users.extend(temp.index.values)
    print temp.head(10)
    
    return exception_users
#find_low_percent_conversion_user(1000, 2)

def remove_low_percent_conversion_users_actions(\
        threshold, bc_threshold, o_threshold, test=False):
    outfile = output_path + 'actions_%d_%d_%d.csv'%(threshold, bc_threshold, o_threshold)
    start = time.time()
    if os.path.exists(outfile):
        df = None
        if test:
            reader = pd.read_csv(outfile, iterator=True)
            df = reader.get_chunk(default_chunk_size)
        else:
            df = pd.read_csv(outfile)
        return df
        
    df = remove_exception_users_actions(threshold, test)
    print len(df)
    exception_users = find_low_percent_conversion_user(bc_threshold, o_threshold)
    df = df[~df['user_id'].isin(exception_users)]
    print len(df)
    if not test:
        df.to_csv(outfile, index=False)
        
    end = time.time(); elapsed = (end - start); start = end;
    print 'transform actions cost %ds'%elapsed
    return df
#remove_low_percent_conversion_users_actions(0, 1000, 2, False)



def pca(std_data):
    start = time.time()
    pca = PCA()
    pca.fit(std_data)
    
    end = time.time(); elapsed = (end - start); start = end;
    print 'PCA cost %ds'%elapsed
    print 'components_: \n', pca.components_, '\n'
    print 'explained_variance_: \n', pca.explained_variance_, '\n'
    print 'explained_variance_ratio_: \n', pca.explained_variance_ratio_, '\n'
    print 'mean_: \n', pca.mean_, '\n'
    print 'noise_variance_: \n', pca.noise_variance_, '\n'
def kmeans_cluster(std_data, k, threshold=200):
    start = time.time()
    
    model = KMeans(n_clusters=k, init='k-means++', verbose=1)
    model.fit(std_data)
    
    #标准化数据及其类别
    r = pd.concat([std_data, pd.Series(model.labels_, index = std_data.index)], \
                   axis = 1)  #每个样本对应的类别
    r.columns = list(std_data.columns) + ['classification'] #重命名表头
    
    norm = []
    for i in range(k): #逐一处理
        norm_tmp = r[std_data.columns][r['classification'] == i] \
                - model.cluster_centers_[i]
        norm_tmp = norm_tmp.apply(np.linalg.norm, axis = 1) #求出绝对距离
        norm.append(norm_tmp / norm_tmp.median()) #求相对距离并添加
    norm = pd.concat(norm) #合并
    print norm.head()

    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    norm[norm <= threshold].plot(style = 'go') #正常点

    discrete_points = norm[norm > threshold] #离群点
    if len(discrete_points) > 0:
        discrete_points.plot(style = 'ro')

    #for i in range(len(discrete_points)): #离群点做标记
    #    id = discrete_points.index[i]
    #    n = discrete_points.iloc[i]
    #    plt.annotate('(%s, %0.2f)'%(id, n), xy = (id, n), xytext = (id, n))

    plt.xlabel(u'编号')
    plt.ylabel(u'相对距离')
    plt.show()
    
    #print 'labels_: \n', model.labels_, '\n'
    #print 'cluster_centers_: \n', model.cluster_centers_, '\n'
    print 'inertia_: \n', model.inertia_, '\n'
    end = time.time(); elapsed = (end - start); start = end;
    print 'KMeans cost %ds'%elapsed
def get_actions(threshold, bc_threshold, o_threshold, test=False):
    return remove_low_percent_conversion_users_actions(\
            threshold, bc_threshold, o_threshold, test)
def cluster_by_percent_conversion(threshold, bc_threshold, o_threshold, test=False):
#    df = get_actions(threshold, bc_threshold, o_threshold, test)
    user_actions = pd.read_csv(out_user_actions, index_col='user_id')
#    print df.head(10)
    print user_actions.head(10)
    
    exclude_user_ids = []
    user_ids1 = find_exception_users(threshold)
#    print len(user_ids1)
    exclude_user_ids.extend(user_ids1)
    user_ids2 = find_low_percent_conversion_user(bc_threshold, o_threshold)
#    print len(user_ids2)
    exclude_user_ids.extend(user_ids2)
#    print len(exclude_user_ids)
    exclude_user_ids = list(set(exclude_user_ids))
#    print len(exclude_user_ids)
#    print len(user_actions)
    user_actions = user_actions[~user_actions.index.isin(exclude_user_ids)]
    print len(user_actions)
    
    users_lv1 = user_actions[user_actions['user_lv_cd']==1]
    print users_lv1
    
    # remove lv1 users
    user_actions = user_actions[user_actions['user_lv_cd']>1]
    print len(user_actions)
    
    # recover sex
    sex_map = {0:2, 1:0, -1:1}
    user_actions['sex'] = user_actions['sex'].apply(lambda x: sex_map[x])
    
    # ad2o
    user_actions['ad2o'] = user_actions.apply(lambda r: \
            float(r['o']) / max(r['a'] - r['d'], 1), axis=1)
    print user_actions.head(10)
    
    # dummy attributes
#    dummy_age = pd.get_dummies(user_actions['age'], prefix='a')
#    for col in dummy_age.columns:
#        dummy_age[col] = dummy_age[col].astype(int)
#    dummy_sex = pd.get_dummies(user_actions['sex'], prefix='s')
#    for col in dummy_sex.columns:
#        dummy_sex[col] = dummy_sex[col].astype(int)
#    dummy_lv = pd.get_dummies(user_actions['user_lv_cd'], prefix='lv')
#    for col in dummy_lv.columns:
#        dummy_lv[col] = dummy_lv[col].astype(int)
#    
#    users = pd.concat([user_actions[['b','a','d','o','f','c','b2o','c2o','a2o','f2o']], \
#                      dummy_age, dummy_sex, dummy_lv], axis=1)
#    print users.head(10)
    
    # corrilation
#    corr = users.corr()
#    for col in corr.columns:
#        corr[col] = corr[col].apply(lambda x: round(x, 2))
#    print corr
#    out_corr = output_path + 'corr.csv'
#    corr.to_csv(out_corr)
    
    # corrilation by user attributes
#    ages = sorted(user_actions['age'].drop_duplicates().values)
#    sexes = sorted(user_actions['sex'].drop_duplicates().values)
#    levels = sorted(user_actions['user_lv_cd'].drop_duplicates().values)
#    print ages
#    print sexes
#    print levels
#    for a in ages:
#        for s in sexes:
#            for lv in levels:
#                temp = user_actions[(user_actions['age']==a) \
#                                    & (user_actions['sex']==s) \
#                                    & (user_actions['user_lv_cd']==lv)]
#                del temp['age'],temp['sex'],temp['user_lv_cd']
#                temp.rename(columns={'user_reg_age_by_day': 'reg_days'}, inplace=True)
#                
#                length = len(temp)
#                print length
#                if length > 0:
##                    print temp.mean().to_frame('mean').T
#                    print temp.describe()
#                    corr = temp.corr()
#                    for col in corr.columns:
#                        corr[col] = corr[col].apply(lambda x: round(x, 2))
##                    print 'corr age=%d, sex=%d, lv=%d:'%(a, s, lv)
##                    print corr
#                    corr.to_csv(output_path + 'corr_%d_%d_%d.csv'%(a, s, lv))
    
    #数据标准化
    std_data = 1.0 * (user_actions - user_actions.mean()) / user_actions.std()
    std_data.fillna(0, inplace=True)
    
    # PCA
#    pca(std_data)
    
    # cluster by percent conversion
    kmeans_cluster(std_data, 10)
cluster_by_percent_conversion(0, 1000, 2, False)

























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


