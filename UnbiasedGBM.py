from glob import glob
from pickle import GLOBAL
from unittest.case import _AssertRaisesContext
import numpy as np 
import pandas as pd 
import random, os

from utils import *
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import subprocess

# import sys
# sys.setrecursionlimit(10000)

GLOBAL_THREADS = 1

debug_time = 0
import time

class UnbiasedTree:
    def __init__(self, booster, p):
        self.booster = booster
        self.child = []
        txt = p.stdout.readline().decode()
        txt = txt.split(' ')
        if int(txt[0]):
            self.split_col = txt[1]
            self.split_imp = float(txt[2])
            if self.split_imp < -1e8: self.split_imp = 0
            return
        else:
            self.split_col = txt[1]
            self.split_imp = float(txt[2])
            
            self.split_type = txt[3]
            self.split_val = float(txt[4])
            self.split_avg_l = float(txt[5])
            self.split_avg_r = float(txt[6])
            self.split_avg_x = float(txt[7])
            self.split_dict = eval(txt[8])
            if self.split_type=='cat': self.split_dict[np.nan] = self.split_dict[-1]
            self.child = [UnbiasedTree(booster, p), UnbiasedTree(booster, p)]

    def apply_split(self, df):
        if self.split_type == 'cat':
            if self.split_dict is None:
                return df[self.split_col] == self.split_val
            return df[self.split_col].apply(lambda x: (self.split_dict[x] if x in self.split_dict else self.split_dict[np.nan])) <= self.split_val
        else:
            return df[self.split_col].apply(lambda x: (self.split_dict[-1] if pd.isna(x) else x)) <= self.split_val
        
    def predict_single(self, df):
        if len(self.child)==0: return 0.
        is_right = self.apply_split(df).sum() == 0
        return self.child[is_right].predict_single(df) + (self.split_avg_r if is_right else self.split_avg_l)
    
    def predict(self, df, dep=0):
        if len(df)==0: return []
        if len(self.child)==0: return [0.]*len(df)
        idx_l = self.apply_split(df)
        res = pd.Series(index=df.index, dtype=float)
        res[idx_l==True] = self.child[0].predict(df[idx_l==True], dep+1)
        res[idx_l==True] += self.split_avg_l
        res[idx_l==False] = self.child[1].predict(df[idx_l==False], dep+1)
        res[idx_l==False] += self.split_avg_r
        if dep==0: res += self.split_avg_x
        return res.values.tolist()
    
    def calc_self_imp(self, mp):
        if len(self.child)==0: return
        self.child[0].calc_self_imp(mp)
        self.child[1].calc_self_imp(mp)
        if self.split_col not in mp:
            mp[self.split_col] = 0
        mp[self.split_col] += self.split_imp

class UnbiasedBoost:
    def __init__(self, losstool, n_est, min_leaf, thresh=0, n_data_split=4, lr=0.1, seed=0, n_leaf=32):
        self.losstool = losstool
        self.n_est = n_est
        self.min_leaf = min_leaf
        self.thresh = thresh
        self.Trees = []
        self.feat_imp = {}
        self.n_data_split = n_data_split
        self.lr = lr
        self.seed = seed
        self.n_leaf = n_leaf

    def predict(self, df):
        res = np.array([0 for i in df.index]).astype('float')
        for i in range(len(self.Trees)):
            res += self.predict_one_tree(df, i)
        return res

    def predict_one_tree(self, df, idx):
        res = np.array(self.Trees[idx].predict(df)).astype('float')*self.lr
        return res

    def df_to_str(self, df):
        res = f'{df.shape[1]}\n'
        for c in df.columns:
            line = f'{c} {"cat" if "int" in str(df[c].dtype) else "num"} {df.shape[0]}'
            for v in df[c]: line += f' {v}'
            res += line + '\n'
        return res

    def fit(self, df, label, n_jobs=1, debug=True, valset=None, testset=None, score_type='advance',
            mono_h=1, large_leaf_reward=0.5, return_pred=True):
        for c in df.columns:
            if ' ' in c:
                print('No blank space!')
                exit(0)
        
        if score_type not in ['origin', 'chaos', 'fair', 'advance', 'full']:
            print('Unknow score_type')
            exit(0)
        
        txt = None
        if self.losstool=='logloss':
            txt = f'classification logloss {self.n_est} {self.min_leaf} {self.thresh} {self.lr}\n'
        if self.losstool=='MSE':
            txt = f'regression MSE {self.n_est} {self.min_leaf} {self.thresh} {self.lr}\n'
        if txt is None:
            print('Error, what is the task type?')
                
        txt += self.df_to_str(df)
        
        line = None
        if self.losstool=='logloss':
            line = f'label cat {df.shape[0]}'
            for v in label: line += f' {int(v)}'
        if self.losstool=='MSE':
            line = f'label num {df.shape[0]}'
            for v in label: line += f' {float(v)}'
        if line is None:
            print('Error, what is the task type?')
        
        txt += line + '\n'
        
        if valset is not None:
            txt += '1' + '\n'
            txt += self.df_to_str(valset[1])
        else:
            txt += '0' + '\n'   
        
        if testset is not None:
            txt += '1' + '\n'
            txt += self.df_to_str(testset[1])
        else:
            txt += '0' + '\n'            
        
        p = subprocess.Popen([f'{os.environ["HOME"]}/cache/ugb',f'{n_jobs}',f'{self.seed}',score_type,
                              f'{mono_h}', f'{large_leaf_reward}', f'{self.n_leaf-1}'],
            stdout = subprocess.PIPE,
            stdin = subprocess.PIPE)
        p.stdin.write(txt.encode())
        p.stdin.close()
        
        for _ in range(self.n_est):
            self.Trees.append( UnbiasedTree(self, p) )
        
        history = {}
        mx, mxid = -1e10, self.n_est
        
        if valset is not None:
            for _ in range(self.n_est):
                it = int(p.stdout.readline().decode())
                
                pred = ''
                las = p.stdout.readline().decode()
                while ']' not in las:
                    pred += las
                    las = p.stdout.readline().decode()
                pred += las
                pred = eval(pred)
                pred = [float(o) for o in pred]
                now_score = valset[0](valset[2], pred)
                history[it]=now_score
                
                if now_score > mx: mx, mxid = now_score, it
                
                if debug and it%10==0:
                    print(f' [validset] {it}-th iter,  valid loss {now_score}')
                if it==self.n_est: break
        
        if testset is not None:
            for _ in range(self.n_est):
                it = int(p.stdout.readline().decode())
                
                pred = ''
                las = p.stdout.readline().decode()
                while ']' not in las:
                    pred += las
                    las = p.stdout.readline().decode()
                pred += las
                pred = eval(pred)
                pred = [float(o) for o in pred]
                now_score = testset[0](testset[2], pred)
                
                if debug and it%10==0:
                    print(f' [testset] {it}-th iter,  test loss {now_score}')
                
                if it==mxid:
                    if return_pred:
                        return now_score, pred
                    else:
                        return now_score
                if it==self.n_est: exit(1)

        return None
    
    def calc_self_imp(self):
        self.feat_imp = {}
        for T in self.Trees:
            T.calc_self_imp(self.feat_imp)
        return self.feat_imp
