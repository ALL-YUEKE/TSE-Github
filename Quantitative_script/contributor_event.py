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
def get_contributor_event(c,df,token_list,reponame,df_json): # c is a contributor json obj
    c_uname = c["login"]
    headers={"Authorization":"token "+token_list[random.randint(0,3)]}
    event_url = "https://api.github.com/users/"+c_uname+'/events/public?per_page=100'
    event_req = requests.get(event_url, headers=headers)
    json_name='contributor_json/'+reponame.replace('/','_')+'_'+c_uname+'.json'
    df_json.loc[len(df_json.index)]=[reponame,json_name,c_uname]
    events = event_req.json()
    while 'next' in event_req.links.keys():
      event_req=requests.get(event_req.links['next']['url'],headers=headers)
      events.extend(event_req.json())
    with open(json_name,'w') as f:
      json.dump(events,f)
    #pprint(events)
    email_l =[] 
    name_l = []
    sha_l=[]
    id_l=[]
    #print("!!!!!!!UNAME:"+c_uname)
    try:
      for event in tqdm(events):
        ctype=event['type']
        cid=event['id']
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
            df.loc[len(df.index)]=[user_byemail,user_byemail,email,sha,cid,datenum]
          except (KeyError) as e:
            continue
        try:
          datenum=''
          date=event['created_at']
          for s in date:
            if(s.isdigit()):
              datenum=datenum+s
          datenum=int(datenum)
          for commit in event['payload']['commits']:
#             print(commit)
            username = c["login"]
            sha=commit['sha']
            email=commit['author']['email']
            name=commit['author']['name']
            df.loc[len(df.index)]=[username,name,email,sha,cid,datenum]
#              email = event['payload']['commits'][0]['author']['email'] 
#              name = event['payload']['commits'][0]['author']['name']
#               email_l.append(email)
#               name_l.append(name) 
#               sha_l.append(sha)
#               id_l.append(cid)
#            if email != '' and name != '':
#                print(email)
#                print(name)
#                break
        except (KeyError, IndexError,TypeError) as e:
#               print(e)
            continue
    except (IndexError,TypeError) as e:
      print(events)
def minid(df,df_id):
  for i in range(len(df)):
    user=df.iloc[i,0]
    cid=int(df.iloc[i,4])
    date=int(df.iloc[i,5])
    if(i!=len(df)-1):
      ad_user=df.iloc[i+1,0]
    else:
      df_id.loc[len(df_id.index)]=[user,cid,date]
      continue
    if(ad_user!=user and i!=len(df)-1):
      df_id.loc[len(df_id.index)]=[user,cid,date]
      

  
if __name__ == "__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument('--keyword',required=True)
  parser.add_argument('--start',required=True)
  parser.add_argument('--end',required=True)
  args=parser.parse_args()
  repo_list=args.keyword
  dic=repo_list[-11:-9]
  df_reponame=pd.read_csv(repo_list)
  idname='miniid_'+dic+'.csv'
  df_id=pd.read_csv(idname)
  jsonname='contributor_json_'+dic+'.csv'
  df_json=pd.read_csv(jsonname)
  for i in tqdm(range(int(args.start),int(args.end))):
    contributor={'user':[0],'commitname':[0],'email':[0],'sha':[0],'id':[0],'datenum':[0]}
#  repo={'name':[0],'email':[0],'sha':[0]}
    df_con=pd.DataFrame(contributor)

#  df_repo=pd.DataFrame(repo)
    headers={"Authorization":"token "+token_list[random.randint(0,3)]}
    reponame=df_reponame.iloc[i,0]
    contributor_url ="https://api.github.com/repos/"+reponame+"/contributors?per_page=100"
    req = requests.get(contributor_url,  headers=headers)
    contributors = req.json() 
    while 'next' in req.links.keys():
      req=requests.get(req.links['next']['url'],headers=headers)
      contributors.extend(req.json())
    for c in tqdm(contributors):
      get_contributor_event(c,df_con,token_list,reponame,df_json)
    contributor_rec='contributor/'+reponame.replace('/','_')+"_contributor.csv"
    df_con.drop(axis=0,index=0,inplace=True)
    df_id.loc[len(df_id.index)]=[i,i,i]
    minid(df_con,df_id)
    df_con.to_csv(contributor_rec,index=False)
    df_json.to_csv(jsonname,index=False)
    df_id.to_csv(idname,index=False)

  


