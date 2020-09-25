# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:15:01 2020

@author: shunsuke
"""

# coding=utf-8
import requests
#import datetime
from bs4 import BeautifulSoup
import csv

#    Year = "2006"
#    day = datetime.datetime.today()
#    thisyear = day.year

#    while True:
target_url = 'https://totoone.jp/blog/datawatch/detail.php?mid=3453&tid=131'
   
resp = requests.get(target_url)
soup = BeautifulSoup(resp.text)

tables = soup.find_all("tbody")
#    print(tables[1])
#    print(tables[2])
#    print(tables[3])

table = tables[4]

data = []
for i in range(len(table.find_all("td"))):
    string=(table.find_all("td")[i]).text
#    print(string)
    data.append(string)

print(data[1])
with open('test.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(data)

# In[]:
print(data[1][:27])
home_rank = int(data[1][3:5].replace(" ",""))
home_point=int(data[1][11:13].replace(" ",""))
home_win = int(data[1][13:15].replace(" ",""))
home_draw= int(data[1][16:18].replace(" ",""))
home_win = int(data[1][19:21].replace(" ",""))
