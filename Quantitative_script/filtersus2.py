#analyze leak_email and get the number of each group
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
  contributor={'user':[0],'commitname':[0],'email':[0],'datenum':[0],'reponame':[0],'verify':[0]}
  df_leak=pd.DataFrame(contributor)
  inputname='check_name_'+dic+'.csv'
  df_sus=pd.read_csv(inputname)
  commitl=list(set(df_sus.iloc[:,1].to_list()))

  df_leak.drop(axis=0,index=0,inplace=True)
  user_cache=[]
  for i in tqdm(commitl):
    df_commitname=df_sus[df_sus.commitname==i]
    user_l=list(set(df_commitname.iloc[:,0].to_list())) 
    for j in range(len(user_l)):
      df_user=df_commitname[df_commitname.user==user_l[j]]
      email_l=list(set(df_user.iloc[:,2].to_list())) 
      for k in range(len(email_l)):
        df_email=df_user[df_user.email==email_l[k]]
        verify_l=list(set(df_email.iloc[:,10].to_list())) 
        for l in range(len(verify_l)):
          df_verify=df_email[df_email.verify==verify_l[l]]
          num=len(df_verify)
          user=df_verify.iloc[0,0]
          commitname=df_verify.iloc[0,1]
          email=df_verify.iloc[0,2]
          reponame=df_verify.iloc[0,7]
          verify=df_verify.iloc[0,10]
          df_leak.loc[len(df_leak.index)]=[user,commitname,email,num,reponame,verify]
  outputname='leak_user_'+dic+'.csv'
  df_leak.to_csv(outputname,index=False)

