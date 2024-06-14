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
def get_repo_event(events,df,req_time,token_list): # c is a contributor json obj
  for event in tqdm(events):
    try:
      email=event['commit']['author']['email']
      author=event['commit']['author']['name']
      committer=event['commit']['committer']['name']
      date=event['commit']['author']['date']
      name=event['commit']['author']['name']
      user_byemail=event['author']['login']
      sha=event['sha']
      datenum=''
      for s in date:
        if(s.isdigit()):
          datenum=datenum+s
      datenum=int(datenum)
      df.loc[len(df.index)]=[name,email,sha,user_byemail,datenum,author,committer]
    except (KeyError, IndexError,TypeError) as e:
      continue

if __name__ == "__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument('--keyword',required=True)
  parser.add_argument('--start',required=True)
  parser.add_argument('--end',required=True)
  args=parser.parse_args()
  repo_list=args.keyword
  dic=repo_list[-11:-9]
  df_reponame=pd.read_csv(repo_list)
  req_time=0
  for i in tqdm(range(int(args.start),int(args.end))):
    reponame=df_reponame.iloc[i,0]
    commit={'name':[0],'email':[0],'sha':[0],'user_byemail':[0],'date':[0],'author':[0],'committer':[0]}
    df=pd.DataFrame(commit)
    headers={"Authorization":"token "+token_list[random.randint(0,5)]}
    event_url='https://api.github.com/repos/'+reponame+'/commits?per_page=100'        #per_page controls the number of items that api returns perpage, we set it as max number 100. 
    event_req = requests.get(event_url, headers=headers)
    event_json=event_req.json()
    count=0
    while 'next' in event_req.links.keys():         #
      count+=1                                    
      if(count<=20):                               #the max number of page you want to check,in this code, we check max 2100 commits.
        event_req=requests.get(event_req.links['next']['url'],headers=headers)
        event_json.extend(event_req.json())          #add the page more than 1 to the page1 json
      else:
        break
    json_name='commits_json/'+reponame.replace('/','_')+'_event.json'
    with open(json_name,'w') as f:
      json.dump(event_json,f)         #now the event_json contains all items from pages we search
    get_repo_event(event_json,df,req_time,token_list)
    repo_rec='repo_commit/'+reponame.replace('/','_')+"_event.csv"
    df.drop(axis=0,index=0,inplace=True)
    df.to_csv(repo_rec,index=False)

