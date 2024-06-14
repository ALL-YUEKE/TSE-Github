import csv
import sys
import time
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import *
import argparse
import os
if __name__ == "__main__":
  filename=[]
  for root,dirs,files in os.walk('cross_check_result'):
    for f in files:
      if (f[:4]=='leak'):
        f_path=os.path.join(root, f)
        filename.append(f_path)
  leak_list=[]
  user_list=[]
  for leak in filename:
    df_leak=pd.read_csv(leak)
    leak_list.extend(df_leak.iloc[:,0].to_list())
  leak_list=list(set(leak_list))
  print("leak_user: ",len(leak_list))
  filename=[]
  for root,dirs,files in os.walk('contributor_list'):
    for f in files:
      f_path=os.path.join(root, f)
      filename.append(f_path)
  print(filename)
  for user in filename:
    df_user=pd.read_csv(user)
    user_list.extend(list(set(df_user.iloc[:,2].to_list())))
  user_list=list(set(user_list))
  print("total user: ",len(user_list))

