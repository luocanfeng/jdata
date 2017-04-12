# -*- coding: utf-8 -*-
'''
@data: 2017-04-11
@author: luocanfeng

√ 性别插值
√ 年龄插值
TODO 商品属性插值
TODO 预测用户购买行为
TODO 预测用户购买商品

TODO 用户行为+时间周期
'''
import numpy as np
import pandas as pd
from sklearn import linear_model


file_encoding = 'gbk'
output_path = './out/'


file_tf_users = output_path + 'tf_users.csv'
file_tf_products = output_path + 'tf_products.csv'
file_tf_dd_users = output_path + 'tf_dd_users.csv'
file_tf_dd_products = output_path + 'tf_dd_products.csv'

file_tf_fs_users = output_path + 'tf_fs_users.csv'
file_tf_dd_fs_users = output_path + 'tf_dd_fs_users.csv'
file_tf_fsa_users = output_path + 'tf_fsa_users.csv'
file_tf_dd_fsa_users = output_path + 'tf_dd_fsa_users.csv'



def load_std_users(filename):
    df = pd.read_csv(filename, index_col='user_id')
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
    
    print df_zs.head()
    return df_zs



def guess(df, guess_col, exception_val, clf=None):
    sample = df[df[guess_col]!=exception_val]
#    sample = sample[sample.age!=0]
    msk = np.random.rand(len(sample)) < 0.8
    train = sample[msk]
    test = sample[~msk]
#    print '\ntrain\'s len:%d'%len(train)
#    print 'test\'s len:%d'%len(test)
    
    y_train = train[guess_col]
    x_train = train.drop(guess_col, axis=1)
    y_test = test[guess_col]
    x_test = test.drop(guess_col, axis=1)
    
    if clf == None:
        clf = linear_model.LinearRegression()
    clf.fit(x_train, y_train)
#    print '\nclf.coef:'
#    print clf.coef_
    
    y_predict = clf.predict(x_test)
#    print '\npredict\'s len:%d'%len(y_predict)
    
    t = pd.DataFrame(y_test, index = y_test.index)
    p = pd.DataFrame(y_predict, index = y_test.index, columns=['predict'])
    r = pd.concat([t, p], axis = 1, join='inner')
    r.columns = ['reality', 'predict']
#    print r.head()
    r['predict_round'] = r['predict'].apply(\
                            lambda x:reflect_predict_to_reality(guess_col, x))
    r['tp'] = r['predict_round'] == r['reality']
#    print '\nr:'
#    print r.head()
    r_len = len(r)
    tp_len = len(r[r.tp==True])
#    print 'tp=%d/%d, %f'%(tp_len, r_len, float(tp_len) / r_len)
    return float(tp_len) / r_len

def reflect_predict_to_reality(predict_col, val):
    if predict_col == 'sex':
        return 1 if val>=0 else -1
    elif predict_col == 'age':
        return int(round(val))
    else:
        return np.NaN


'''
before deduplicated
    guess sex by LinearRegression, precision=0.846415
    guess sex by LogisticRegression, precision=0.846857
    guess sex by LogisticRegressionCV, precision=0.846297
    guess age by LinearRegression, precision=0.515205
    guess age by LogisticRegression, precision=0.545580
    guess age by LogisticRegressionCV, precision=0.546151
after deduplicated
    guess sex by LinearRegression, precision=0.846691
    guess sex by LogisticRegression, precision=0.846178
    guess sex by LogisticRegressionCV, precision=0.849279
    guess age by LinearRegression, precision=0.515921
    guess age by LogisticRegression, precision=0.544717
    guess age by LogisticRegressionCV, precision=0.544684
行为数据去重并不能帮助提高年龄插值正确率
性别插值并不能帮助提高年龄插值正确率
'''
def precision_of_guess(filename):
    df = load_std_users(filename)
    
    clf = linear_model.LinearRegression()
    precision_arr = [guess(df, 'sex', 0, clf) for i in range(100)]
    print 'guess sex by LinearRegression, precision=%f'%(np.mean(precision_arr))
    
    clf = linear_model.LogisticRegression()
    precision_arr = [guess(df, 'sex', 0, clf) for i in range(50)]
    print 'guess sex by LogisticRegression, precision=%f'%(np.mean(precision_arr))
    
    clf = linear_model.LogisticRegressionCV()
    precision_arr = [guess(df, 'sex', 0, clf) for i in range(10)]
    print 'guess sex by LogisticRegressionCV, precision=%f'%(np.mean(precision_arr))
    
    clf = linear_model.LinearRegression()
    precision_arr = [guess(df, 'age', 0, clf) for i in range(100)]
    print 'guess age by LinearRegression, precision=%f'%(np.mean(precision_arr))
    
    clf = linear_model.LogisticRegression()
    precision_arr = [guess(df, 'age', 0, clf) for i in range(20)]
    print 'guess age by LogisticRegression, precision=%f'%(np.mean(precision_arr))
    
    clf = linear_model.LogisticRegressionCV()
    precision_arr = [guess(df, 'age', 0, clf) for i in range(5)]
    print 'guess age by LogisticRegressionCV, precision=%f'%(np.mean(precision_arr))

#precision_of_guess(file_tf_fs_users)
#precision_of_guess(file_tf_dd_fs_users)


def fill(filename, guess_col, exception_val, output_file, clf=None):
    df = load_std_users(file_tf_dd_users)
    train = df[df[guess_col]!=exception_val]
    test = df[df[guess_col]==exception_val]
#    print '\ntrain\'s len:%d'%len(train)
#    print 'test\'s len:%d'%len(test)
    
    y_train = train[guess_col]
    x_train = train.drop(guess_col, axis=1)
    y_test = test[guess_col]
    x_test = test.drop(guess_col, axis=1)
    
    if clf == None:
        clf = linear_model.LinearRegression()
    clf.fit(x_train, y_train)
#    print '\nclf.coef:'
#    print clf.coef_
    
    y_predict = clf.predict(x_test)
#    print '\npredict\'s len:%d'%len(y_predict)
    
    p = pd.DataFrame(y_predict, index = y_test.index, columns=['predict'])
    p['predict_round'] = p['predict'].apply(\
                            lambda x:reflect_predict_to_reality(guess_col, x))
    print len(df)
    print len(p)
    
    df2 = pd.concat([df, p], axis = 1, join='outer')
    print len(df2)
    df2['predict_round'].fillna(exception_val, inplace=True)
    df2[guess_col] = df2.apply(lambda df2: df2['predict_round'] \
            if df2[guess_col]==exception_val else df2[guess_col], axis=1)
    print df2
    print len(df2[df2[guess_col]==exception_val])
    

    df3 = pd.read_csv(filename, index_col='user_id')
    df3.fillna(0, inplace=True)
    
    df4 = df2[guess_col].to_frame('fill')
    print df4.head()
    
    r = pd.concat([df3, df4], axis = 1, join='outer')
    r[guess_col] = r['fill']
    del df,df2,df3,df4,r['fill']
    print r
    print len(r[r[guess_col]==exception_val])

    r.to_csv(output_file)
    return r
    
def fill_sex_and_age():
    clf = linear_model.LogisticRegressionCV()
    fill(file_tf_users, 'sex', 0, file_tf_fs_users, clf)
    fill(file_tf_dd_users, 'sex', 0, file_tf_dd_fs_users, clf)
    fill(file_tf_fs_users, 'age', 0, file_tf_fsa_users, clf)
    fill(file_tf_dd_fs_users, 'age', 0, file_tf_dd_fsa_users, clf)
    
fill_sex_and_age()


