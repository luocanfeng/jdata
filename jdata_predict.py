# -*- coding: utf-8 -*-
'''
@data: 2017-04-18
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



def test():
    user_actions = pd.read_csv(out_user_actions, index_col='user_id')
    product_actions = pd.read_csv(out_product_actions, index_col='sku_id')
    
    user_actions_order_ge5 = user_actions[user_actions['o'] >= 5][['o']]
    users_order_ge5 = user_actions_order_ge5.index.values
    print users_order_ge5
    print users_order_ge5.dtype
    print user_actions_order_ge5.groupby('o').size().plot(kind='bar')
    
    product_orders = sorted(list(set(product_actions['o'].values)))
    product_orders_length = len(product_orders)
    quarter_o = product_orders[3*product_orders_length/4]
    print product_orders
    print quarter_o
    product_actions_top_quarter = product_actions[product_actions['cate'] == 8]
    product_actions_top_quarter = \
            product_actions[product_actions['o'] >= quarter_o][['o']]
    print product_actions_top_quarter.head(10)
    
    products = pd.read_csv(file_products, index_col='sku_id')
    sku_ids = products.index.values
    sku_ids_top_quarter = product_actions_top_quarter.index.values
    sku_ids = [val for val in sku_ids if val in sku_ids_top_quarter]
    sku_ids_len = len(sku_ids)

    user_order_predict = user_actions_order_ge5
    user_order_predict['sku_id'] = sku_ids[np.random.randint(0,sku_ids_len)]
    del user_order_predict['o']
    print user_order_predict
    user_order_predict.to_csv(out_test, encoding='utf-8')


