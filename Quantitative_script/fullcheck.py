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
import numpy as np
if __name__ == "__main__":
  df_repo=pd.read_csv("allrepo.csv")
  parser=argparse.ArgumentParser()
  parser.add_argument('--start',required=True)
  parser.add_argument('--end',required=True)
  args=parser.parse_args()
  rec=pd.read_csv("commit_rec.csv")
  for i in tqdm(range(int(args.start),int(args.end))):
    reponame=df_repo.iloc[i,1]
    repo=reponame.replace('/','_')
    emailname=[]
    verifylist=[]
    try:
      repo_path="contributor_contri/"+repo+".csv"
      json_name="commit_json/"+repo+".json"
      df_commit=pd.read_csv(repo_path)
      for j in tqdm(range(len(df_commit))):
        sha=df_commit.iloc[j,3]
        conrepo=df_commit.iloc[j,6]
        commit_url='https://api.github.com/repos/'+conrepo+'/commits/'+sha
        headers={"Authorization":"token "+token_list[random.randint(0,9)]}
        event_req = requests.get(commit_url, headers=headers)
        try:
          commit=event_req.json()
        except(simplejson.errors.JSONDecodeError):
          continue
        with open(json_name,'a') as f:
          json.dump(commit,f)
        try:
          userbyemail=commit['author']['login']
          emailname.append(userbyemail)
        except (KeyError, IndexError,TypeError) as e:
          emailname.append('not_found')
        try:
          verify=commit['commit']['verification']['verified']
          verifylist.append(verify)
        except (KeyError, IndexError,TypeError) as e:
          verifylist.append('not found')
      df_commit['emailname']=emailname
      df_commit['verify']=verifylist
      outputname="contributor_verify/fullinfo_"+repo+".csv"
      df_commit.to_csv(outputname,index=False)
      rec.loc[len(rec)]=[i,reponame]
      rec.to_csv("commit_rec.csv",index=False)
    except(NameError,FileNotFoundError):
      continue
