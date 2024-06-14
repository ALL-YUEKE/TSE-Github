#check verify condition 
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
  inputname='check_verify_'+dic+'.csv'
  df_leak=pd.read_csv(inputname)
#  df_leak.drop(['index'],axis=1,inplace=True)
#  df_leak.to_csv("leak_email_cs.csv",index=False)
  contributor={'user':[0],'commitname':[0],'email':[0],'sha':[0],'id':[0],'datenum':[0],'groupid':[0],'reponame':[0],'conrepo':[0],'emailname':[0],'verify':[0]}
#  repo={'name':[0],'email':[0],'sha':[0]}
  df_con=pd.DataFrame(contributor)
  df_con.drop(axis=0,index=0,inplace=True)
  for i in tqdm(range(len(df_leak))):
    user=df_leak.iloc[i,0]
    emailuser=df_leak.iloc[i,9]
    if(user!=emailuser):
      df_con.loc[len(df_con.index)]=df_leak.iloc[i].to_list()
    else:
      continue
  outputname='check_name_'+dic+'.csv'
  df_con.to_csv(outputname,index=False)
      
