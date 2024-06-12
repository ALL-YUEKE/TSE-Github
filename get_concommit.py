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
  inputname='leak_email_'+dic+'.csv'
  df_leak=pd.read_csv(inputname)
#  df_leak.drop(['emailname'],axis=1,inplace=True)
  emailname=[]
  verifylist=[]
  user_cache=[]
  for i in tqdm(range(len(df_leak))):
    reponame=df_leak.iloc[i,8]
    sha=df_leak.iloc[i,3]
    email=df_leak.iloc[i,2]
    commit_url='https://api.github.com/repos/'+reponame+'/commits/'+sha
    headers={"Authorization":"token "+token_list[random.randint(0,5)]}
    if(len(user_cache)<=1):
      user_cache.append(email)
    else:
      del(user_cache[0])
      user_cache.append(email)
    if(len(user_cache)!=1 and user_cache[0]==user_cache[1]):
      user_byemail=emailname[-1]
      verify=verifylist[-1]
      verifylist.append(verify)
      emailname.append(user_byemail)
    else:
      event_req = requests.get(commit_url, headers=headers)
      commit=event_req.json()
      try:
        userbyemail=commit['author']['login']
        verify=commit['commit']['verification']['verified']
        emailname.append(userbyemail)
        verifylist.append(verify)
      except (KeyError, IndexError,TypeError) as e:
        emailname.append('not_found')
        verifylist.append('not found')
        continue
  df_leak['emailname']=emailname
  df_leak['verify']=verifylist
  outputname="fullinfo_"+dic+".csv"
  df_leak.to_csv(outputname,index=False)

