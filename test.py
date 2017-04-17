# -*- coding: utf-8 -*-
'''
@data: 2017-04-17
@author: luocanfeng
'''
import numpy as np
import pandas as pd


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


out_actions = output_path + 'actions.csv'
def transform_actions(test=False):
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
    
    '''
    #一个商品在浏览的同时，如果一直都会同时产生相同的点击，则认为是错误的统计，该点击无效
    1、以商品、时间排序，检测出商品一个动作连带触发其他动作的错误统计情况，依此删除连带动作记录
    2、以用户、时间排序，依次分析每个用户在一个时间周期内的活动情况
    3、期间可能需要根据商品的销量/用户购买次数进行分类，将中间数据落盘
    '''
    
    print df.head(20)
    print df.dtypes
    if not test:
        df.to_csv(out_actions, index=False)
transform_actions(test=False)






