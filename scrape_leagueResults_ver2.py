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


# In[]: totoの開催番号でループ
df_base=pd.DataFrame()
lot_number =  np.arange(569,200,-1)
for lotNum in lot_number:
    
    # In[Loop-A]: 一回分のtotoをループ
    urls=[]
    
    url = "https://www.totoone.jp/blog/datawatch/index.php?id=" +str(lotNum)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text)
    
    # totoの開催番号を取得
    try:
        No = (soup.find_all("div", class_="txt_lead1"))
        No = No[0].text
        No = int(re.findall(r'\d+', No)[0])
        print("\nLottery Number: ",No)
    except IndexError:
        print("\nGO TO /index.php?id=" +str(lotNum))
        print('⇒ Skip: Linked to Deleted site')
        continue
    
    # # toto開催のくじの種類を判別　mini-toto,GOAL3のみは除外
    texts = soup.find_all("img")
    figName = texts[3].get("src")
    #object6は未確認
    if (figName == "/blog/datawatch/img/object8.gif")\
        or (figName == "/blog/datawatch/img/object7.gif")\
            or (figName == "/blog/datawatch/img/object6.gif")\
               or (figName == "/blog/datawatch/img/object5.gif")\
                    or (figName == "/blog/datawatch/img/object4.gif"): #GOAL3の画像で判別
        print("⇒ Skip: Not the site for toto")
        continue
    
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
        # print(league,section,len(texts_away))
        
        # 5節以内の試合は考慮しない
        if int(section) <= 5:
            print('⇒ Skip: Game-Data is insufficient yet (section < 5)')
            continue
        
        # J1,2,3以外の試合は除外する
        if (sche[0] != "1") and (sche[0] != "2") and (sche[0] != "3"):
            print('⇒ Skip: Not the game-Data of J-league')
            continue
        # In[]: texts_away の長さが異なる時があるので、その時を対処
        awayInfo = 8
        # for t in range(len(texts_home)):
        #     print(t,texts_home[t])
            
        if len(texts_away)!=20:
            awayInfo = len(texts_away)-12 #len=18なら6　len⁼17なら5 なので
        
        if len(texts_home)<11:
            print('⇒ Skip: HTML format is not supported cuz of Wcup or England-league or...etc')
            continue
        
        #末尾が数字の場合、通常あるツイッタータイムラインがない場合なので例外処理
        # s = texts_home[-1].text
        # if 'Tweets' not in s:
        #     print(s)
        #     print("The page not have Twitter Timeline ")
        #     continue

        s = texts_away[-1].text
        if 'Tweets' not in s:
            print("The page not have Twitter Timeline ")
            continue
        
        
        # In[3]: リーグでの戦績を抽出
        # リーグ全体での順位や勝ち点、勝利数などを抽出　例）4位 勝ち点 17　4勝 5分 10敗
        s = texts_home[0].text
        stringList = re.findall(r'\d+', s)
        # print(s,stringList,len(texts_home))
        home_rank   = int(stringList[0])
        home_point  = int(stringList[1])
        home_win    = int(stringList[2])
        home_draw   = int(stringList[3])
        home_lose   = int(stringList[4])
        
        # 試合戦績がないものは除外する 海外チームの試合が対象
        if (home_win+home_draw+home_lose) == 0:
            print('⇒ Skip: Overseas team has no Game-Data')
            continue
        
        s = texts_away[awayInfo].text
        stringList = re.findall(r'\d+', s)
        # print(s,stringList,len(texts_away))
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
        home_lose_onHome  = int(stringList[2])
        
        s = texts_away[awayInfo].text
        stringList = re.findall(r'\d+', s)
        away_win_onAway   = int(stringList[0])
        away_draw_onAway  = int(stringList[1])
        away_lose_onAway  = int(stringList[2])
        
        
        # リーグでの総得点と１試合平均得点
        s = texts_home[3].text
        stringList = re.findall(r'\d+', s)
        home_getScore   = int(stringList[0])
        try: 
            home_AvegetScore   = int(stringList[1])+int(stringList[2])/100.0
        except IndexError:
            home_AvegetScore   = int(stringList[1])
            
        s = texts_away[awayInfo+3].text
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
        
        s = texts_away[awayInfo+4].text
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
        
        
        s = texts_away[awayInfo+5].text
        stringList = re.findall(r'\d+', s)
        away_shootNum   = int(stringList[0])
        
        s = texts_away[awayInfo+6].text
        stringList = re.findall(r'\d+', s)
        away_shootedNum   = int(stringList[0])
        
        
        # リーグでのそれぞれ得点数の回数
        s = texts_home[7].text
        stringList = re.findall(r'\d+', s)
        home_getCount_0 = int(stringList[1])
        home_getCount_1 = int(stringList[3])
        home_getCount_2 = int(stringList[5])
        home_getCount_3 = int(stringList[7])
        
        s = texts_away[awayInfo+7].text
        stringList = re.findall(r'\d+', s)
        # print(s,stringList,len(texts_away))
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
        
        s = texts_away[awayInfo+8].text
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
        
        
        s = texts_away[awayInfo+9].text
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
        
        s = texts_away[awayInfo+10].text
        stringList = re.findall(r'\d+', s)
        away_RecentLoss   = int(stringList[0])
        try: 
            away_RecentAveLoss   = int(stringList[1])+int(stringList[2])/100.0
        except IndexError:
            away_RecentAveLoss   = int(stringList[1])
        
    
        
        # In[4]: データフレームの構造に
        
        data = []
        name = []
        
        data.append(No)                         #totoの開催番号
        name.append("LotteryNo")
        
        data.append(league)                     #リーグの種類
        data.append(section)                    #節数
        data.append(year)                       #年
        data.append(mon)                        #月
        data.append(day)                        #日
        name.append("league")
        name.append("section")
        name.append("year")
        name.append("mon")
        name.append("day")
            
        data.append(home_rank)                  #リーグ順位
        data.append(home_point)                 #勝ち点
        data.append(home_win)                   #勝ち数
        data.append(home_draw)                  #引き分け数
        data.append(home_lose)                  #負け数
        data.append(home_win_onHome)            #ホームでの勝ち数
        data.append(home_draw_onHome)           #ホームでの分け数
        data.append(home_lose_onHome)           #ホームでの負け数
        data.append(home_getScore)              #得点数
        data.append(home_AvegetScore)           #平均得点数
        data.append(home_lossScore)             #失点数
        data.append(home_AvelossScore)          #平均失点数
        data.append(home_shootNum)              #シュート数
        data.append(home_shootedNum)            #被シュート数
        data.append(home_getCount_0)            #得点が0点の試合数
        data.append(home_getCount_1)            #得点が1点の試合数
        data.append(home_getCount_2)            #得点が2点の試合数
        data.append(home_getCount_3)            #得点が3点の試合数
        data.append(home_lossCount_0)           #失点が0点の試合数
        data.append(home_lossCount_1)           #失点が1点の試合数
        data.append(home_lossCount_2)           #失点が2点の試合数
        data.append(home_lossCount_3)           #失点が3点の試合数
        data.append(home_RecentScore)           #直近3試合の得点数
        data.append(home_RecentAveScore)        #直近3試合の平均得点数
        data.append(home_RecentLoss)            #直近3試合の失点数
        data.append(home_RecentAveLoss)         #直近3試合の平均失点数
        
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

df_base = df_base.reset_index(drop=True)
df_base.to_csv('TeamInfo.csv')
print(df_base)

