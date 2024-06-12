#divide different group by user names
import requests, json
import os
from pprint import pprint
import csv
import sys
import time
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import *
import random
import argparse
if __name__ == "__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument('--keyword',required=True)
  args=parser.parse_args()
  repo_list=args.keyword
  dic=repo_list[-11:-9]
  inputname='suspicious_'+dic+'.csv'
  email_cache=[]
  contributor={'user':[0],'commitname':[0],'email':[0],'sha':[0],'id':[0],'datenum':[0],'groupid':[0],'reponame':[0]}
  df_leak=pd.DataFrame(contributor)
  df_sus=pd.read_csv(inputname)
  df_leak.drop(axis=0,index=0,inplace=True)
  g=0
  for i in tqdm(range(len(df_sus))):
    if(i==len(df_sus)-1):
      next_name=df_sus.iloc[i,1]
    else:
      next_name=df_sus.iloc[i+1,1]
    name=df_sus.iloc[i,1]
    email=df_sus.iloc[i,2]
    if(name==next_name and (email not in email_cache) and type(email)==str and email[-10:]!="github.com"):
      email_cache.append(email)
    if(name!=next_name):
      if(len(email_cache)>=2):
 #       print(email_cache)
        df_relate=df_sus[df_sus.commitname==name]
        group_list=[g for _ in range(len(df_relate))]
        df_relate['groupid']=group_list
        df_leak=pd.concat([df_leak,df_relate])
        g+=1
      email_cache=[]
  outputname='filter_email'+dic+'.csv'
  df_leak.to_csv(outputname,index=False)
    
