import requests, json
import os
from pprint import pprint
import csv
import sys
import time
import pandas as pd
import random
from tqdm import *
from pandas.core.frame import DataFrame
import argparse
def getrepo(df,keyword):
  headers={"Authorization":"token "+token_list[random.randint(0,3)]}
  keyword=args.keyword
  for i in tqdm(range(11)):
    sorturl="https://api.github.com/search/repositories?q="+keyword+"&sort=stars&page="+str(i)+"&per_page=100"
    req = requests.get(sorturl,  headers=headers)
    repos = req.json() 
    try:
      for repo in (repos['items']):
         reponame=repo['full_name']
         url=repo['url']    
         df.loc[len(df.index)]=[reponame,url]
    except (KeyError, IndexError) as e:
      print(e)
      continue
if __name__ == "__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument('--keyword',required=True)
  args=parser.parse_args()
  keyword=args.keyword
  repoinfo={'repo':[0],'url':[0]}
  df=pd.DataFrame(repoinfo)
  getrepo(df,keyword)
  df.drop(axis=0,index=0,inplace=True)
  name='repo_url/'+keyword+'_repo.csv'
  df.to_csv(name,index=False)
