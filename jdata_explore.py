# -*- coding: utf-8 -*-
'''
@data: 2017-03-29
@author: luocanfeng
'''
import pandas as pd
import matplotlib.pyplot as plt


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


figure = plt.figure(figsize=(10, 8), dpi=200)
ax1 = figure.add_subplot(2, 3, 1)
ax2 = figure.add_subplot(2, 3, 2)
ax3 = figure.add_subplot(2, 3, 3)
ax4 = figure.add_subplot(2, 3, 4)
ax5 = figure.add_subplot(2, 3, 5)
ax6 = figure.add_subplot(2, 3, 6)


'''
explore users
columns: user_id, age, sex, user_lv_cd, user_reg_tm
sex: 0表示男，1表示女，2表示保密
'''
def explore_users():
    df = pd.read_csv(file_users, encoding=file_encoding, index_col='user_id', \
                     infer_datetime_format=True)
    print '\nUsers\'s length:%d'%len(df)
    print '\nUsers Header:'
    print df.head()
    print '\nUsers age distribution:'
    print df.groupby('age').size()
    df.groupby('age').size().plot(kind='bar', ax=plt.subplot(ax1))
    print '\nUsers sex distribution:'
    print df.groupby('sex').size()
    df.groupby('sex').size().plot(kind='bar', ax=plt.subplot(ax2))
    print '\nUsers level distribution:'
    print df.groupby('user_lv_cd').size()
    df.groupby('user_lv_cd').size().plot(kind='bar', ax=plt.subplot(ax3))
explore_users()


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
    df.groupby('a1').size().plot(kind='bar', ax=plt.subplot(ax4))
    print '\nProducts a2 distribution:'
    print df.groupby('a2').size()
    df.groupby('a2').size().plot(kind='bar', ax=plt.subplot(ax5))
    print '\nProducts a3 distribution:'
    print df.groupby('a3').size()
    df.groupby('a3').size().plot(kind='bar', ax=plt.subplot(ax6))
    #print '\nProducts brand distribution:'
    #print df.groupby('brand').size()
    #df.groupby('brand').size().plot(kind='bar', ax=plt.subplot(ax6))
explore_products()


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
explore_comments()


'''
explore actions
columns: user_id, sku_id, time, model_id, type, cate, brand
type：1浏览,2加购,3删购,4下单,5关注,6点击
'''
def explore_actions(file_name):
    reader = pd.read_csv(file_name, encoding=file_encoding, \
                         infer_datetime_format=True, iterator=True)
    df = reader.get_chunk(100000)
    print '\nActions Header:'
    print df.head()
    
    return df
explore_actions(file_actions_02)

