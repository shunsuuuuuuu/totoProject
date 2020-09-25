# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:15:01 2020

@author: shunsuke

スクレイピングしたサイト　↓
# https://www.totoone.jp/blog/datawatch/

"""
# In[1]:パッケージのインポート
# coding=utf-8
import requests
#import datetime
from bs4 import BeautifulSoup


# In[2]:指定したURLからスクレイピングにより文字列を抽出する
target_url = "https://www.totoone.jp/blog/datawatch/detail.php?mid=17036&tid=705"
   
resp = requests.get(target_url)
soup = BeautifulSoup(resp.text)

tables = soup.find_all("tbody") #tbodyによって表が抽出される
#    print(tables[1])
#    print(tables[2])
#    print(tables[3])

table = tables[4] #サイトの上から順にtables[1] [2]...となっている。[4]は試合までのリーグ成績

data = []
for i in range(len(table.find_all("td"))):
    string=(table.find_all("td")[i]).text
#    print(string)
    data.append(string)

# with open('test.csv', 'w') as f:
#     writer = csv.writer(f, lineterminator='\n')
#     writer.writerows(data)

# In[]:テキストから必要な数値を抽出し、データフレームに変換
print(data[1][:27])
home_rank = int(data[1][3:5].replace(" ",""))
home_point=int(data[1][11:13].replace(" ",""))
home_win = int(data[1][13:15].replace(" ",""))
home_draw= int(data[1][16:18].replace(" ",""))
home_win = int(data[1][19:21].replace(" ",""))

print(data[2][:27])
away_rank = int(data[2][3:5].replace(" ",""))
away_point=int(data[2][11:13].replace(" ",""))
away_win = int(data[2][13:15].replace(" ",""))
away_draw= int(data[2][16:18].replace(" ",""))
away_win = int(data[2][19:21].replace(" ",""))