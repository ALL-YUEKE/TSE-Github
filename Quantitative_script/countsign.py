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
if __name__ == "__main__":
  filename=[]
  true=0
  false=0
  unfound=0
  for root,dirs,files in os.walk('contributor_verify'):
    for f in files:
      f_path=os.path.join(root, f)
      filename.append(f_path)
  for repo in tqdm(filename):
    try:
      df=pd.read_csv(repo)
    except(pd.errors.EmptyDataError):
      continue
    verifylist=df.iloc[:,8].to_list()
    verifylist=df['verify'].to_list()
    for v in verifylist:
      if(v=="False"):
        false+=1
      if(v=="True"):
        true+=1
  print("True:"+str(true)+" False:"+str(false))
  
