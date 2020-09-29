# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:32:07 2020

@author: USER
"""

# In[1]:パッケージのインポート
# coding=utf-8
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score # モデル評価用(正答率)
from sklearn.metrics import log_loss # モデル評価用(logloss)     
from sklearn.metrics import roc_auc_score # モデル評価用(auc)

import IPython
def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)
        

all_train = pd.read_csv('train_DATA.csv')
all_train = all_train.iloc[:,1:]

all_train = all_train[all_train['Result']!='*']

res = all_train["Result"].astype(int)
all_train['Result'] = res

# all_train = all_train[all_train["league"]=="J1"]

# In[]: 学習用データに変換していく

train = all_train.iloc[:,8:]
non_use_col = [
                # 'home_getScore',"home_lossScore","away_getScore","away_lossScore",
                # 'home_getCount_0','home_getCount_1','home_getCount_2','home_getCount_3',
                # 'away_getCount_0','away_getCount_1','away_getCount_2','away_getCount_3',
                # "home_RecentScore","away_RecentScore","home_RecentLoss","away_RecentLoss",
               "Result","Home_Score","Away_Score"]

use_col = [col for col in train.columns if col not in non_use_col]
# use_col = [	
#  	"boteHome",
#  	"boteAway",
#  	"home_rank",
#  	"away_rank",
#  	"boteDraw",
#  	"home_shootNum",
#  	"away_shootNum",
#  	"home_shootedNum",
#  	"home_AvegetScore",
#  	"away_AvegetScore",
#  	"away_shootedNum",
#  	"away_draw"
# ]
X_train = train[use_col]
# X_train  = train[['boteHome','boteAway',"home_rank","away_rank"]]


Y_train = all_train['Result']

# In[]: 学習してみる
from sklearn.model_selection import train_test_split

X_train = X_train.values
Y_train = Y_train.values

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, random_state=28, test_size=0.1)

# In[]:
    
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=1000, n_features=3,
#                            n_informative=2, n_redundant=0,
#                             random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)
y_pred_clf = clf.predict(x_test)
# acc : 正答率64
acc = accuracy_score(y_test,y_pred_clf)
print('Acc :', acc)


 # In[]:
# 必要ライブラリのロード
import lightgbm as lgb
# import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 学習に使用するデータを設定
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# LightGBM parameters
params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass', # 目的 : 多クラス分類 
        'num_class': 3,            # クラス数 : 3
        'metric': {'multi_error'}, # 評価指標 : 誤り率(= 1-正答率) 
        #　他には'multi_logloss'など
}

# モデルの学習
model = lgb.train(params,
                  train_set=lgb_train, # トレーニングデータの指定
                  valid_sets=lgb_eval, # 検証データの指定
                  )

# 予測の実施
y_pred_rate = model.predict(x_test)
y_pred = np.argmax(y_pred_rate,axis=1)

# モデル評価



# acc : 正答率64
acc = accuracy_score(y_test,y_pred)
print('Acc :', acc)

# 特徴量重要度の算出 (データフレームで取得)
# cols = list(df.drop('target',axis=1).columns)       # 特徴量名のリスト(目的変数target以外)
f_importance = np.array(model.feature_importance(importance_type='gain')) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':use_col, 'importance':f_importance})
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
display(df_importance)

# In[]:
df_result = pd.DataFrame(x_test)
df_result.columns = use_col
df_result = df_result[["home_rank","away_rank"]]
df_result['y_test'] = y_test
df_result['y_pred'] = y_pred
df_result["y_pred_rf"]=y_pred_clf