# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:15:01 2020

@author: shunsuke

スクレイピングしたサイト　↓
# https://www.totoone.jp/blog/datawatch/

"""
# In[1]:パッケージのインポート
# coding=utf-8
import pandas as pd
import numpy as np
import requests
#import datetime
import re
from bs4 import BeautifulSoup


# In[Loop-A]: 一回分のtotoをループ
df_base=pd.DataFrame()
urls=[]

url = "https://www.totoone.jp/blog/datawatch/index.php?id=488"
resp = requests.get(url)
soup = BeautifulSoup(resp.text)

# totoの開催番号を取得
No = (soup.find_all("div", class_="txt_lead1"))
No = No[0].text
No = int(re.findall(r'\d+', No)[0])

# 各試合のURLを取得 （45、57は変動の恐れあり）
texts = soup.find_all("a")
for i in np.arange(45,57,1): 
    urls.append(texts[i].get("href"))

for url in urls:
    print(url)
    # In[2]:指定したURLからスクレイピングにより文字列を抽出する
    target_url = "https://www.totoone.jp" + url
       
    resp = requests.get(target_url)
    soup = BeautifulSoup(resp.text)
    
    
    texts_home=(soup.find_all("td", class_="home tR")) #home tR というクラスをすべて取得
    texts_away=(soup.find_all("td", class_="away")) #awayというクラスをすべて取得、リーグ戦績以外にも存在するので、homeとは処理が少し異なる。
    
    # In[]: 試合日程を追加
    sche = (soup.find_all("td", colspan="2"))
    sche = sche[0].text
    sche = re.findall(r'\d+', sche)
    
    league = "J" + str(sche[0])
    section = sche[1]
    year = sche[2]
    mon = sche[3]
    day = sche[4]
    
    if sche[0] != "1":
        print('not J1, HTML format isnt supported')
        continue
    # In[3]: リーグでの戦績を抽出

    # リーグ全体での順位や勝ち点、勝利数などを抽出　例）4位 勝ち点 17　4勝 5分 10敗
    s = texts_home[0].text
    stringList = re.findall(r'\d+', s)
    home_rank   = int(stringList[0])
    home_point  = int(stringList[1])
    home_win    = int(stringList[2])
    home_draw   = int(stringList[3])
    home_lose   = int(stringList[4])
    
    s = texts_away[8].text
    stringList = re.findall(r'\d+', s)
    away_rank   = int(stringList[0])
    away_point  = int(stringList[1])
    away_win    = int(stringList[2])
    away_draw   = int(stringList[3])
    away_lose   = int(stringList[4])
    
    
    # リーグ(home/away)での戦績を抽出　例）　ホーム成績 1勝 3分 6敗
    s = texts_home[1].text
    stringList = re.findall(r'\d+', s)
    home_win_onHome   = int(stringList[0])
    home_draw_onHome  = int(stringList[1])
    home_lose_onHome    = int(stringList[2])
    
    s = texts_away[9].text
    stringList = re.findall(r'\d+', s)
    away_win_onAway   = int(stringList[0])
    away_draw_onAway  = int(stringList[1])
    away_lose_onAway    = int(stringList[2])
    
    
    # リーグでの総得点と１試合平均得点
    s = texts_home[3].text
    stringList = re.findall(r'\d+', s)
    home_getScore   = int(stringList[0])
    try: 
        home_AvegetScore   = int(stringList[1])+int(stringList[2])/100.0
    except IndexError:
        home_AvegetScore   = int(stringList[1])
        
    s = texts_away[11].text
    stringList = re.findall(r'\d+', s)
    away_getScore   = int(stringList[0])
    try: 
        away_AvegetScore   = int(stringList[1])+int(stringList[2])/100.0
    except IndexError:
        away_AvegetScore   = int(stringList[1])
    
    
    # リーグでの総失点数と１試合平均失点数
    s = texts_home[4].text
    stringList = re.findall(r'\d+', s)
    home_lossScore   = int(stringList[0])
    try: 
        home_AvelossScore   = int(stringList[1])+int(stringList[2])/100.0
    except IndexError:
        home_AvelossScore   = int(stringList[1])
    
    s = texts_away[12].text
    stringList = re.findall(r'\d+', s)
    away_lossScore   = int(stringList[0])
    try: 
        away_AvelossScore   = int(stringList[1])+int(stringList[2])/100.0
    except IndexError:
        away_AvelossScore   = int(stringList[1])
    
    
    # リーグ中のシュート数と被シュート数
    s = texts_home[5].text
    stringList = re.findall(r'\d+', s)
    home_shootNum   = int(stringList[0])
    
    s = texts_home[6].text
    stringList = re.findall(r'\d+', s)
    home_shootedNum   = int(stringList[0])
    
    
    s = texts_away[13].text
    stringList = re.findall(r'\d+', s)
    away_shootNum   = int(stringList[0])
    
    s = texts_away[14].text
    stringList = re.findall(r'\d+', s)
    away_shootedNum   = int(stringList[0])
    
    
    # リーグでのそれぞれ得点数の回数
    s = texts_home[7].text
    stringList = re.findall(r'\d+', s)
    home_getCount_0 = int(stringList[1])
    home_getCount_1 = int(stringList[3])
    home_getCount_2 = int(stringList[5])
    home_getCount_3 = int(stringList[7])
    
    s = texts_away[15].text
    stringList = re.findall(r'\d+', s)
    away_getCount_0 = int(stringList[1])
    away_getCount_1 = int(stringList[3])
    away_getCount_2 = int(stringList[5])
    away_getCount_3 = int(stringList[7])
    
    # リーグでのそれぞれ失点数の回数
    s = texts_home[8].text
    stringList = re.findall(r'\d+', s)
    home_lossCount_0 = int(stringList[1])
    home_lossCount_1 = int(stringList[3])
    home_lossCount_2 = int(stringList[5])
    home_lossCount_3 = int(stringList[7])
    
    s = texts_away[16].text
    stringList = re.findall(r'\d+', s)
    away_lossCount_0 = int(stringList[1])
    away_lossCount_1 = int(stringList[3])
    away_lossCount_2 = int(stringList[5])
    away_lossCount_3 = int(stringList[7])
    
    # 直近3試合での得点数と失点数
    s = texts_home[9].text
    stringList = re.findall(r'\d+', s)
    home_RecentScore   = int(stringList[0])
    try: 
        home_RecentAveScore   = int(stringList[1])+int(stringList[2])/100.0
    except IndexError:
        home_RecentAveScore   = int(stringList[1])
    
    
    s = texts_away[17].text
    stringList = re.findall(r'\d+', s)
    away_RecentScore   = int(stringList[0])
    try: 
        away_RecentAveScore   = int(stringList[1])+int(stringList[2])/100.0
    except IndexError:
        away_RecentAveScore   = int(stringList[1])
    
    s = texts_home[10].text
    stringList = re.findall(r'\d+', s)
    home_RecentLoss   = int(stringList[0])
    try: 
        home_RecentAveLoss   = int(stringList[1])+int(stringList[2])/100.0
    except IndexError:
        home_RecentAveLoss   = int(stringList[1])
    
    s = texts_away[18].text
    stringList = re.findall(r'\d+', s)
    away_RecentLoss   = int(stringList[0])
    try: 
        away_RecentAveLoss   = int(stringList[1])+int(stringList[2])/100.0
    except IndexError:
        away_RecentAveLoss   = int(stringList[1])
    

    
    # In[4]: データフレームの構造に
    
    data = []
    name = []
    data.append(league)
    data.append(section)
    data.append(year)
    data.append(mon)
    data.append(day)
    name.append("league")
    name.append("section")
    name.append("year")
    name.append("mon")
    name.append("day")
        
    data.append(home_rank)
    data.append(home_point)
    data.append(home_win)
    data.append(home_draw)
    data.append(home_lose)
    data.append(home_win_onHome)
    data.append(home_draw_onHome)
    data.append(home_lose_onHome)
    data.append(home_getScore)
    data.append(home_AvegetScore)
    data.append(home_lossScore)
    data.append(home_AvelossScore)
    data.append(home_shootNum)
    data.append(home_shootedNum)
    data.append(home_getCount_0)
    data.append(home_getCount_1)
    data.append(home_getCount_2)
    data.append(home_getCount_3)
    data.append(home_lossCount_0)
    data.append(home_lossCount_1)
    data.append(home_lossCount_2)
    data.append(home_lossCount_3)
    data.append(home_RecentScore)
    data.append(home_RecentAveScore)
    data.append(home_RecentLoss)
    data.append(home_RecentAveLoss)
    
    name.append("home_rank")
    name.append("home_point")
    name.append("home_win")
    name.append("home_draw")
    name.append("home_lose")
    name.append("home_win_onHome")
    name.append("home_draw_onHome")
    name.append("home_lose_onHome")
    name.append("home_getScore")
    name.append("home_AvegetScore")
    name.append("home_lossScore")
    name.append("home_AvelossScore")
    name.append("home_shootNum")
    name.append("home_shootedNum")
    name.append("home_getCount_0")
    name.append("home_getCount_1")
    name.append("home_getCount_2")
    name.append("home_getCount_3")
    name.append("home_lossCount_0")
    name.append("home_lossCount_1")
    name.append("home_lossCount_2")
    name.append("home_lossCount_3")
    name.append("home_RecentScore")
    name.append("home_RecentAveScore")
    name.append("home_RecentLoss")
    name.append("home_RecentAveLoss")
    
    data.append(away_rank)
    data.append(away_point)
    data.append(away_win)
    data.append(away_draw)
    data.append(away_lose)
    data.append(away_win_onAway)
    data.append(away_draw_onAway)
    data.append(away_lose_onAway)
    data.append(away_getScore)
    data.append(away_AvegetScore)
    data.append(away_lossScore)
    data.append(away_AvelossScore)
    data.append(away_shootNum)
    data.append(away_shootedNum)
    data.append(away_getCount_0)
    data.append(away_getCount_1)
    data.append(away_getCount_2)
    data.append(away_getCount_3)
    data.append(away_lossCount_0)
    data.append(away_lossCount_1)
    data.append(away_lossCount_2)
    data.append(away_lossCount_3)
    data.append(away_RecentScore)
    data.append(away_RecentAveScore)
    data.append(away_RecentLoss)
    data.append(away_RecentAveLoss)
    
    name.append("away_rank")
    name.append("away_point")
    name.append("away_win")
    name.append("away_draw")
    name.append("away_lose")
    name.append("away_win_onAway")
    name.append("away_draw_onAway")
    name.append("away_lose_onAway")
    name.append("away_getScore")
    name.append("away_AvegetScore")
    name.append("away_lossScore")
    name.append("away_AvelossScore")
    name.append("away_shootNum")
    name.append("away_shootedNum")
    name.append("away_getCount_0")
    name.append("away_getCount_1")
    name.append("away_getCount_2")
    name.append("away_getCount_3")
    name.append("away_lossCount_0")
    name.append("away_lossCount_1")
    name.append("away_lossCount_2")
    name.append("away_lossCount_3")
    name.append("away_RecentScore")
    name.append("away_RecentAveScore")
    name.append("away_RecentLoss")
    name.append("away_RecentAveLoss")
    
    df_data = pd.DataFrame(np.array(data), index=name)
    df_data = df_data.transpose()
    
    # In[5]: 対戦チームを加える
    texts = soup.find_all("img")
    home_teamName = texts[4].get("alt")
    away_teamName = texts[5].get("alt")
    df_data["HomeTeam"]=home_teamName
    df_data["Awayteam"]=away_teamName
    
    # In[6]: チーム名を最初の方に
    col_res = list(df_data.columns[:-2])
    col_name = list(df_data.columns[-2:])
    df_data = df_data[col_name+col_res]
    

        
    # In[7]: データフレームを合体させる
    df_base = pd.concat([df_base,df_data])
    
    
# In[]:
print(df_base)