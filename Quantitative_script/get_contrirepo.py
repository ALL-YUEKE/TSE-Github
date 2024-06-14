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

def get_contributor_event(json_path,cid): # c is a contributor json obj
    f=open(json_path,'r')
    events=json.loads(f.read())
    headers={"Authorization":"token "+token_list[random.randint(0,3)]}
    for event in tqdm(events):
        ctype=event['type']
        uid=event['id']
        if(uid==str(cid)):
          conname=event['repo']['name']
          return conname
        else:
          continue
"""
        if(ctype=='PullRequestEvent'):
          sha=event['payload']['pull_request']['head']['sha']
          date=event['payload']['pull_request']['created_at']
          datenum=''
          for s in date:
            if(s.isdigit()):
              datenum=datenum+s
          datenum=int(datenum)
          commit_url="https://api.github.com/repos/"+reponame+"/commits/"+sha
          try:
            commit_req=requests.get(commit_url, headers=headers)
            commit_page=commit_req.json()
            user_byemail=commit_page['commit']['author']['login']
            email=commit_page['commit']['author']['email']
            df_con.loc[len(df_con7b.index)]=[user_byemail,user_byemail,email,sha,cid,datenum]
          except (KeyError) as e:
            continue
        else:
          try:
            datenum=''
            date=event['created_at']
            for s in date:
              if(s.isdigit()):
                datenum=datenum+s
            datenum=int(datenum)
            for commit in event['payload']['commits']:
               sha=commit['sha']
               email=commit['author']['email']
               name=commit['author']['name']
               df_con.loc[len(df_con.index)]=[username,name,email,sha,cid,datenum]
          except (KeyError) as e:
            continue
"""
def get_contributor_path(df_con,df_json,reponame):
  df_getevent=df_json[df_json.reponame==reponame]
  for i in range(len(df_getevent)):
    json_path=df_getevent.iloc[i,1]
    username=df_getevent.iloc[i,2]
    get_contributor_event(df_con,reponame,json_path,username)

if __name__ == "__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument('--keyword',required=True)
  args=parser.parse_args()
  repo_list=args.keyword
  dic=repo_list[-11:-9]
  inputname='filter_email'+dic+'.csv'
  df_leak=pd.read_csv(inputname)
  con_list=[]
  for i in tqdm(range(len(df_leak))):
    reponame=df_leak.iloc[i,7]
    user=df_leak.iloc[i,0]
    cid=df_leak.iloc[i,4]
    json_path="../contributor_json/"+reponame.replace("/","_")+"_"+user+".json"
    conname=get_contributor_event(json_path,cid)
    con_list.append(conname)
  df_leak['conrepo']=con_list
  outputname='leak_email'+dic+'.csv'
  df_leak.to_csv(outputname,index=False)
