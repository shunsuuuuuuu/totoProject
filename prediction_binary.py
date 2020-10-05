# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:32:07 2020

@author: USER
"""

# In[1]:パッケージのインポート
# coding=utf-8
import pandas as pd
import numpy as np

from sklearn.svm import SVC
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

# In[]: 勝ち負けでのみ分類する

all_train = all_train[(all_train["Result"]!=0)] 
# all_train = all_train[(all_train["Result"]==2)]

    
    
    
# In[]: 学習用データに変換していく

train = all_train.iloc[:,8:]
non_use_col = [
                'home_getScore',"home_lossScore","away_getScore","away_lossScore",
                'home_getCount_0','home_getCount_1','home_getCount_2','home_getCount_3',
                'away_getCount_0','away_getCount_1','away_getCount_2','away_getCount_3',
                "home_RecentScore","away_RecentScore","home_RecentLoss","away_RecentLoss",
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

all_train['Result'] = all_train['Result'].replace({2:0})
Y_train = all_train['Result']

#クラスの比率
n_target0, n_target1 = len(all_train[all_train['Result'] == 0]), len(all_train[all_train['Result'] == 1])
n_all = n_target0+n_target1
print('負け の割合 :',n_target0/n_all) # target0(健常者)の割合
print('勝ち の割合 :',n_target1/n_all) # target1(がん患者)の割合

# In[]: 学習してみる
from sklearn.model_selection import train_test_split

X_train = X_train.values
Y_train = Y_train.values

x_train, x_test, y_train, y_test =\
    train_test_split(X_train, Y_train, random_state=28, test_size=0.2)


# In[]: SVC

model = SVC(probability=True)
model.fit(x_train, y_train)
y_pred_prob_svm = model.predict_proba(x_test)
y_pred_prob_svm = y_pred_prob_svm[:,1]
y_pred_svm = np.where(y_pred_prob_svm > 0.5, 1, 0)
    
acc = accuracy_score(y_test,y_pred_svm)
print('Acc_svm :', acc)

# In[]:Random Forest
    
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# X, y = make_classification(n_samples=1000, n_features=3,
#                            n_informative=2, n_redundant=0,
#                             random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(x_train, y_train)
y_pred_prob_rf = clf.predict_proba(x_test)
y_pred_prob_rf=y_pred_prob_rf[:,1]
y_pred_rf = np.where(y_pred_prob_rf > 0.5, 1, 0)
# acc : 正答率64
acc = accuracy_score(y_test,y_pred_rf)
print('Acc_rf :', acc)


# In[]: XGBoost

import xgboost as xgb
# XGBoost が扱うデータセットの形式に直す
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
# 学習用のパラメータ
xgb_params = {
    # 二値分類問題
    'objective': 'binary:logistic',
    # 'min_child_weight ':10,
    # 'ete':0.001,
    "gamma":10,
    "max_depth":2,
    # 評価指標
    'eval_metric': 'logloss',
}
# モデルを学習する
bst = xgb.train(xgb_params,
                dtrain,
                num_boost_round=100,  # 学習ラウンド数は適当
                )
# 検証用データが各クラスに分類される確率を計算する
y_pred_prob_xgb = bst.predict(dtest)
# しきい値 0.5 で 0, 1 に丸める
y_pred_xgb = np.where(y_pred_prob_xgb > 0.5, 1, 0)
# 精度 (Accuracy) を検証する
acc = accuracy_score(y_test, y_pred_xgb)
print('Acc_xgb :', acc)
    
    

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
        # 'boosting_type': 'gbdt',
        'objective': 'binary', # 目的 : 多クラス分類 
        # 'num_class': 2,            # クラス数 : 3
        # 'metric': {'auc'}, # 評価指標 : 誤り率(= 1-正答率) 
        'max_depth':2,
        'min_child_samples':100,
        # 'learning_rate':0.1
        #　他には'multi_logloss'など
}

# モデルの学習
model = lgb.train(params,
                  train_set=lgb_train, # トレーニングデータの指定
                  valid_sets=lgb_eval, # 検証データの指定
                  verbose_eval=-1
                  )

# 予測の実施
y_pred_prob_lgbm = model.predict(x_test)
y_pred_lgbm = np.where(y_pred_prob_lgbm > 0.5, 1, 0)
# y_pred = np.argmax(y_pred_rate,axis=1)


# モデル評価
# acc : 正答率64
acc = accuracy_score(y_test,y_pred_lgbm.astype(int))
print('Acc_lgb :', acc)

# 特徴量重要度の算出 (データフレームで取得)
# cols = list(df.drop('target',axis=1).columns)       # 特徴量名のリスト(目的変数target以外)
f_importance = np.array(model.feature_importance(importance_type='gain')) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':use_col, 'importance':f_importance})
df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
display(df_importance)

# In[]:Ensemble
y_pred_prob = (y_pred_prob_svm + y_pred_prob_rf + y_pred_prob_lgbm + y_pred_prob_xgb)/4
y_pred_ens = np.where(y_pred_prob > 0.5, 1, 0)
acc = accuracy_score(y_test, y_pred_ens)
print('Acc_ens :', acc)

# In[]:
df_result = pd.DataFrame(x_test)
df_result.columns = use_col
df_result = df_result[["home_rank","away_rank"]]
df_result['y_test'] = y_test
df_result['y_pred'] = y_pred_lgbm
df_result["y_pred_rf"]=y_pred_rf