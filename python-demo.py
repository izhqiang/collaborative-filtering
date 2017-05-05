# 引用 http://www.jianshu.com/p/ae10bd629f74

python3
import numpy as np
import pandas as pd

# pandas数据结构DataFrame
# 指定了列名。可以在read_csv函数中指定参数，不一定非得是csv文件
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('/Users/zhangqiang/Downloads/ml-100k/u.data', sep = '\t', names = header)
[100000 rows x 4 columns]

# help函数查询文档
>>> help (pd.read_csv)

943个用户以及精选的1682部电影的评分
>>> n_users = df.user_id.unique().shape[0]
>>> n_users
943
>>> n_items = df.item_id.unique().shape[0]
>>> n_items
1682
>>> print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
Number of users = 943 | Number of movies = 1682

# 将整个10万条数据分割为训练集和测试集
>>> from sklearn import cross_validation as cv
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
train_data,test_data = cv.train_test_split(df, test_size = 0.25)
train_data  [75000 rows x 4 columns]
test_data   [25000 rows x 4 columns]


train_data_matrix = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]
>>> train_data_matrix # 生成稀疏矩阵
array([[ 5.,  3.,  4., ...,  0.,  0.,  0.],
       [ 4.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ...,
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  5.,  0., ...,  0.,  0.,  0.]])


test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
>>> test_data_matrix
array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ...,
       [ 5.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]])


使用sklearn的pairwise_distances函数来计算余弦相似性
创建相似性矩阵：user_similarity和item_similarity
--------------------------------------------------------------------------------


上述表述有误，计算的是用户a和k的余弦相似性
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric = "cosine")
>>> user_similarity
array([[ 0.        ,  0.8863552 ,  0.98423974, ...,  0.91472172,
         0.86809843,  0.69558477],
       [ 0.8863552 ,  0.        ,  0.91805274, ...,  0.94744722,
         0.9033864 ,  0.93838303],
       [ 0.98423974,  0.91805274,  0.        , ...,  0.89341273,
         0.91936166,  0.96619137],
       ...,
       [ 0.91472172,  0.94744722,  0.89341273, ...,  0.        ,
         0.89657324,  0.8877209 ],
       [ 0.86809843,  0.9033864 ,  0.91936166, ...,  0.89657324,
         0.        ,  0.82354982],
       [ 0.69558477,  0.93838303,  0.96619137, ...,  0.8877209 ,
         0.82354982,  0.        ]])

item_similarity = pairwise_distances(train_data_matrix.T, metric = "cosine")
>>> item_similarity
array([[ 0.        ,  0.66561359,  0.73777729, ...,  1.        ,
         0.94581716,  1.        ],
       [ 0.66561359,  0.        ,  0.8354188 , ...,  1.        ,
         0.91318014,  1.        ],
       [ 0.73777729,  0.8354188 ,  0.        , ...,  1.        ,
         1.        ,  1.        ],
       ...,
       [ 1.        ,  1.        ,  1.        , ...,  0.        ,
         1.        ,  1.        ],
       [ 0.94581716,  0.91318014,  1.        , ...,  1.        ,
         0.        ,  1.        ],
       [ 1.        ,  1.        ,  1.        , ...,  1.        ,
         1.        ,  0.        ]])


通过基于用户的CF应用下面的公式做出预测：
def predict(rating, similarity, type = 'user'):
def predict(rating, similarity, type):
    if type == 'user':
        mean_user_rating = rating.mean(axis = 1)
        rating_diff = (rating - mean_user_rating[:,np.newaxis])
        pred = mean_user_rating[:,np.newaxis] + similarity.dot(rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
>>> predict
<function predict at 0x108453f28>


item_prediction = predict(train_data_matrix, item_similarity, type = 'item')
>>> item_prediction
array([[ 0.34481547,  0.358818  ,  0.37444401, ...,  0.42220669,
         0.41240821,  0.41939322],
       [ 0.08679741,  0.09979364,  0.09477536, ...,  0.09807142,
         0.09991887,  0.09994051],
       [ 0.0634707 ,  0.06721181,  0.0646438 , ...,  0.06319638,
         0.06558284,  0.06603212],
       ...,
       [ 0.033509  ,  0.03989134,  0.03794674, ...,  0.04408703,
         0.04330975,  0.04402142],
       [ 0.12172354,  0.12931056,  0.13714852, ...,  0.1440286 ,
         0.14231994,  0.14455681],
       [ 0.23118851,  0.22553317,  0.24994862, ...,  0.28814202,
         0.27831831,  0.28554432]])

user_prediction = predict(train_data_matrix, user_similarity, type = 'user')
>>> user_prediction
array([[ 1.55260765,  0.55991435,  0.4526597 , ...,  0.27040616,
         0.27032443,  0.26792643],
       [ 1.32232235,  0.30091886,  0.14223304, ..., -0.06757093,
        -0.0661181 , -0.06939686],
       [ 1.33122763,  0.26135862,  0.1133517 , ..., -0.10212414,
        -0.10045826, -0.10364132],
       ...,
       [ 1.23970751,  0.22945977,  0.07984277, ..., -0.12017565,
        -0.11926819, -0.12237002],
       [ 1.35435636,  0.3174529 ,  0.1942385 , ..., -0.01373719,
        -0.01304687, -0.01587944],
       [ 1.44107623,  0.41760266,  0.3196194 , ...,  0.1353847 ,
         0.1352251 ,  0.13286135]])


评估
这里采用均方根误差(RMSE)来度量预测评分的准确性
可以使用sklearn的mean_square_error(MSE)函数，其中RMSE仅仅是MSE的平方根。
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
>>> rmse
<function rmse at 0x1084662f0>

>>> print ('User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
User based CF RMSE: 3.1273678526361195

>>> print ('Item based CF RMSe: ' + str(rmse(item_prediction, test_data_matrix)))
Item based CF RMSe: 3.4539988883984893

可以看出，基于Memory的算法很容易实现并产生合理的预测质量。



# ==================基于模型的协同过滤================== #
基于模型的协同过滤是基于矩阵分解(MF)的，矩阵分解广泛应用于推荐系统中，它比基于内存的CF有更好的扩展性和稀疏性。MF的目标是从已知的评分中学习用户的潜在喜好和产品的潜在属性，随后通过用户和产品的潜在特征的点积来预测未知的评分。

计算MovieLens数据集的稀疏度：
>>> sparsity = round(1.0 - len(df) / float(n_users*n_items),3)
>>> print ('The sparsity level of MovieLen100K is ' + str(sparsity * 100) + '%')
The sparsity level of MovieLen100K is 93.7%

import scipy.sparse as sp
from scipy.sparse.linalg import svds


help (svds)
    k : int, optional
        Number of singular values and vectors to compute.
        Must be 1 <= k < min(A.shape).

u, s, vt = svds(train_data_matrix, k = 20)
# 对角矩阵
s_diag_matrix = np.diag(s)
x_pred = np.dot(np.dot(u,s_diag_matrix),vt)
>>> x_pred
array([[  6.64428301e+00,   2.36413266e+00,   9.33062799e-01, ...,
         -9.36800523e-03,   4.23871002e-02,   0.00000000e+00],
       [  1.71666123e+00,   9.55351531e-02,  -8.43998179e-02, ...,
          1.35122979e-02,   6.62376678e-04,   0.00000000e+00],
       [  9.24839078e-02,  -9.00980527e-02,   8.24917603e-02, ...,
          2.16237902e-02,   1.38838891e-03,   0.00000000e+00],
       ...,
       [  1.20198561e+00,  -9.82642404e-02,   3.22488421e-01, ...,
         -4.62491323e-04,   6.82632117e-03,   0.00000000e+00],
       [  1.95137420e+00,   2.11361288e-01,  -2.73468658e-01, ...,
          1.24369115e-02,   2.78306051e-03,   0.00000000e+00],
       [  1.30166002e+00,   2.17589128e+00,   7.34259322e-01, ...,
         -7.48226136e-03,   3.53510211e-02,   0.00000000e+00]])

>>> print ('User-based CF MSE: ' + str(rmse(x_pred, test_data_matrix)))
User-based CF MSE: 2.716049697692806