# -*- coding: utf-8 -*-
'''
@data: 2017-04-11
@author: luocanfeng

√ 行为统计
√ 用户+行为
TODO 商品+行为
TODO 性别插值
TODO 年龄插值
TODO 商品属性插值
TODO 预测用户购买行为
TODO 预测用户购买商品

TODO 用户行为+时间周期
'''
import numpy as np
import pandas as pd
from sklearn import linear_model


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

file_transformed_user_actions = out_file_path + 'transformed_user_actions.csv'
file_transformed_users = out_file_path + 'transformed_users.csv'
file_transformed_product_actions = out_file_path + 'transformed_product_actions.csv'
file_transformed_products = out_file_path + 'transformed_products.csv'



def load_standard_users():
    df = pd.read_csv(file_transformed_users, index_col='user_id')
    df.fillna(0, inplace=True)
    
    #数据标准化
    df_zs = 1.0 * (df - df.mean()) / df.std()
    df_zs.fillna(0, inplace=True)
    df_zs['age'] = df['age']
    df_zs['sex'] = df['sex']
    df_zs['cart2buy'] = df['cart2buy']
    df_zs['browse2buy'] = df['browse2buy']
    df_zs['click2buy'] = df['click2buy']
    df_zs['favor2buy'] = df['favor2buy']
    
    print 'Concate Users and Actions:'
    print df_zs.head()
    
    return df_zs
df = load_standard_users()



'''
LinearRegression 准确率84.9704%
LogisticRegression 准确率84.8805%
LogisticRegressionCV 准确率84.9010%
'''
def guess_sex(df):
    sample = df[df.sex!=0]
#    sample = sample[sample.age!=0]
    msk = np.random.rand(len(sample)) < 0.8
    train = sample[msk]
    test = sample[~msk]
#    print '\ntrain\'s len:%d'%len(train)
#    print 'test\'s len:%d'%len(test)
    
    y_train = train.sex
    x_train = train.drop('sex', axis=1)
    y_test = test.sex
    x_test = test.drop('sex', axis=1)
    
    clf = linear_model.LinearRegression()
#    clf = linear_model.LogisticRegression()
#    clf = linear_model.LogisticRegressionCV()
    clf.fit(x_train, y_train)
#    print '\nclf.coef:'
#    print clf.coef_
    
    y_predict = clf.predict(x_test)
#    print '\npredict\'s len:%d'%len(y_predict)
    
    t = pd.DataFrame(y_test, index = y_test.index)
    p = pd.DataFrame(y_predict, index = y_test.index, columns=['predict'])
    r = pd.concat([t, p], axis = 1, join='inner')
    r.columns = ['test', 'predict']
#    print r.head()
    r['predict_round'] = r['predict'].apply(lambda x:1 if x>=0 else -1)
    r['tp'] = r.predict_round == r.test
#    print '\nr:'
#    print r.head()
    r_len = len(r)
    tp_len = len(r[r.tp==True])
    print 'tp=%d/%d, %f'%(tp_len, r_len, float(tp_len) / r_len)
    return float(tp_len) / r_len
precision_arr = [guess_sex(df) for i in range (100)]
print 'precision=%f'%(np.mean(precision_arr))



'''
LinearRegression 准确率51.5214%
LogisticRegression 准确率54.6331%
LogisticRegressionCV 准确率54.2840
'''
def guess_age(df):
    sample = df[df.age!=0]
    sample = sample[sample.sex!=0]
    msk = np.random.rand(len(sample)) < 0.8
    train = sample[msk]
    test = sample[~msk]
#    print '\ntrain\'s len:%d'%len(train)
#    print 'test\'s len:%d'%len(test)
    
    y_train = train.age
    x_train = train.drop('age', axis=1)
    y_test = test.age
    x_test = test.drop('age', axis=1)
    
#    clf = linear_model.LinearRegression()
#    clf = linear_model.LogisticRegression()
    clf = linear_model.LogisticRegressionCV()
    clf.fit(x_train, y_train)
#    print '\nclf.coef:'
#    print clf.coef_
    
    y_predict = clf.predict(x_test)
#    print '\npredict\'s len:%d'%len(y_predict)
    
    t = pd.DataFrame(y_test, index = y_test.index)
    p = pd.DataFrame(y_predict, index = y_test.index, columns=['predict'])
    r = pd.concat([t, p], axis = 1, join='inner')
    r.columns = ['test', 'predict']
#    print r.head()
    r['predict_round'] = r['predict'].apply(lambda x:int(round(x/10))*10)
    r['tp'] = r.predict_round == r.test
#    print '\nr:'
#    print r.head()
    r_len = len(r)
    tp_len = len(r[r.tp==True])
    print 'tp=%d/%d, %f'%(tp_len, r_len, float(tp_len) / r_len)
    return float(tp_len) / r_len
#precision_arr = [guess_age(df) for i in range (10)]
#print 'precision=%f'%(np.mean(precision_arr))

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

