import csv
import sys
import time
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import *
import argparse
if __name__ == "__main__": 
  parser=argparse.ArgumentParser()
  parser.add_argument('--keyword',required=True)
  args=parser.parse_args()
  repo_list=args.keyword
  df_reponame=pd.read_csv(repo_list)
  total_leak=[]
  dic=repo_list[-11:-9]
  leaktable={'project':[0],'username':[0],'email':[0],'namebyemail':[0],'sha':[0]}
  df_leak=pd.DataFrame(leaktable)
  df_leak.drop(axis=0,index=0,inplace=True)
  leak=0
  idname='miniid_'+dic+'.csv'
  df_id=pd.read_csv(idname)
  for j in tqdm(range(len(df_reponame))):
    reponame=df_reponame.iloc[j,0]
    try:
      contributor_rec='contributor/'+reponame.replace('/','_')+"_contributor.csv"
      repo_rec='repo_commit/'+reponame.replace('/','_')+"_event.csv"
      df_con=pd.read_csv(contributor_rec)
      df_repo=pd.read_csv(repo_rec)
    except:
      continue
    for i in range(len(df_repo)):
      namebyemail=df_repo.iloc[i,3]
      sha=df_repo.iloc[i,2]
      username=df_repo.iloc[i,0]
      email=df_repo.iloc[i,1]
      date=df_repo.iloc[i,4]
      if(df_id[df_id.user==namebyemail].empty==False):
        oldest_date=df_id[df_id.user==namebyemail].iloc[0,2]
      else:
        continue
      if(date<oldest_date):
        continue
      df_verify=df_con[((df_con.user==namebyemail)&(df_con.sha==sha))|((df_con.commitname==namebyemail)&(df_con.sha==sha))]
      try:
        if(df_verify.empty==True and username[-5:]!="[bot]" and namebyemail!='0'):
          leak+=1
          dic=repo_list[-11:-9]
          outputname='leak_info_'+dic+'.csv'
          df_leak.loc[len(df_leak.index)]=[reponame,username,email,namebyemail,sha]
          df_leak.to_csv(outputname,index=False)
      except:
        continue



