import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np 
from copy import deepcopy
import random
import math

KEY_WORDS = ['_label', '_pred', '_istrain', '_G', '_H']
global_feat_list = []
global_approx = 100
global_min_samples = 5
global_debug = False

class MSE_tool:
    def calc_g(self, y, fpre):
        return -(y-fpre)
    def calc_h(self, y, fpre):
        return 1

class logloss_tool:
    def calc_g(self, y, fpre):
        fpre = min(fpre, 350)
        if y==1: return -1./(math.exp(fpre)+1)
        if y==0: return math.exp(fpre)/(math.exp(fpre)+1)
        print('Unknown label. binary label must be 0/1')
    def calc_h(self, y, fpre):
        fpre = min(fpre, 350)
        if y==1: return math.exp(fpre)/(math.exp(2*fpre) + 2*math.exp(fpre) + 1)
        if y==0: return math.exp(fpre)/(math.exp(2*fpre) + 2*math.exp(fpre) + 1)
        print('Unknown label. binary label must be 0/1')

def test_model(model): # model:LGBMmodel   df:pd.DataFrame   label:pd.Series
    info = model._Booster.dump_model()["tree_info"]
    print(type(info), len(info))
    print(type(info[0]), type(info[0]['tree_structure']))
    struct = info[0]['tree_structure']
    print([a for a in struct])
    for a in struct:
        if 'child' not in a:
            print(' ',a,struct[a])
    while 'left_child' in struct:
        struct = struct['left_child']
    print([a for a in struct])
    for a in struct:
        if 'child' not in a:
            print(' ',a,struct[a])

def get_left_index(data, dtype, thresh, aim):
    if isinstance(thresh,str):
        thresh = [int(t) for t in thresh.split('||')]
        aim=(not aim)
        if dtype=='==': return (data.apply(lambda x: x in thresh))==aim
        else: print(f'Error : unknown str with not ==')
    if dtype=='==': return (data==thresh)==aim
    if dtype=='<=': return (data<=thresh)==aim
    if dtype=='>=': return (data>=thresh)==aim
    if dtype=='<': return (data<thresh)==aim
    if dtype=='>': return (data>thresh)==aim
    print(f'Error : unknown type {dtype}')
    exit(0)

def calcGH(data, samples = -1):
    if samples == -1:
        return np.array([data['_G'][data['_istrain']==True].sum(), data['_G'][data['_istrain']==False].sum()]), np.array([data['_H'][data['_istrain']==True].sum(), data['_H'][data['_istrain']==False].sum()])
    else:
        if samples < global_min_samples: return 0., 0., 0.
        arrayG = data['_G'][data['_istrain']==False].values
        arrayH = data['_H'][data['_istrain']==False].values
        results = []
        for notvariable in range(global_approx):
            idx = random.sample(range(len(arrayG)), samples)
            results.append( arrayG[idx].sum() / arrayH[idx].sum() )
        return data['_G'][data['_istrain']==True].sum(), data['_H'][data['_istrain']==True].sum(), np.array(results).mean()

def gogogo(tree, feat_imp, data, over_all_data, biased):
    Gx, Hx = calcGH(data)
    if 'left_child' in tree:
        idx_l = get_left_index(data[global_feat_list[tree['split_feature']]], tree['decision_type'], tree['threshold'], tree['default_left'])
        data_l, data_r = data[idx_l==True], data[idx_l==False]
        samples = min(len(data_l[data_l['_istrain']==False]), len(data_r[data_r['_istrain']==False]))
        gl, hl, wlp = calcGH(data_l, samples)
        gr, hr, wrp = calcGH(data_r, samples)
        gx, hx, wxp = calcGH(data, samples)
        score = - gx*wxp + gl*wlp + gr*wrp if samples >= global_min_samples else 0.
        if biased:
            Gl, Hl = calcGH(data_l)
            Gr, Hr = calcGH(data_r)
            score = -Gx[0]**2/Hx[0] +Gl[0]**2/Hl[0] +Gr[0]**2/Hr[0]
        feat_imp[tree['split_feature']] += score
        if global_debug: print('Internal : ', tree['split_feature'], tree['decision_type'], tree['threshold'], tree['default_left'])
        if global_debug: print(f'  samples={samples}', Gx, Hx, tree['internal_value'], tree['internal_weight'], tree['internal_count'])
        gogogo(tree['left_child'], feat_imp, data_l, over_all_data, biased)
        gogogo(tree['right_child'], feat_imp, data_r, over_all_data, biased)
    else:
        if global_debug: print('Leaf :', Gx, Hx, tree['leaf_value'], tree['leaf_weight'], tree['leaf_count'])
        over_all_data.loc[data.index,'_pred'] += [tree['leaf_value']]*len(data)


def convert_str_to_tree(s):
    global global_feat_list
    # a node should be {'split_feature':?, 'decision_type':'<', 'threshold':?, 'default_left':True}
    # if leaf, 'leaf_value':?;  else, 'left_child':?,'right_child':?
    stack = []
    def pop_back(stack):
        if 'left_child' in stack[-2][1]: stack[-2][1]['right_child'] = stack[-1][1]
        else: stack[-2][1]['left_child'] = stack[-1][1]
        return stack[:-1]
    
    for line in s.split('\n'):
        if len(line)<1: continue
        cnt = sum([int(o=='\t') for o in line])
        while len(stack) and cnt<=stack[-1][0]:
            stack = pop_back(stack)
        if '[' in line:
            keys = line.split('[')[-1].split(']')[0]
            node = {'split_feature':global_feat_list.index(keys.split('<')[0]), 'decision_type':'<', 'threshold':float(keys.split('<')[1]), 'default_left':True}
        else:
            value = float(line.split('=')[-1])
            node = {'leaf_value':value, 'leaf_weight':None, 'leaf_count':None}
        stack.append((cnt, node))
    
    while len(stack)>=2:
        stack = pop_back(stack)
    
    return stack[0][1]


def calc_gain(model, dataT, labelT, dataV, labelV, losstool, biased=False):
    global global_feat_list
    global_feat_list = list(dataT.columns)
    data = deepcopy(pd.concat([dataT,dataV])).reset_index(drop=True)
    feat_imp = [0.]*data.shape[1]
    data['_label'] = labelT.values.tolist() + labelV.values.tolist()
    data['_istrain'] = [True]*len(labelT) + [False]*len(labelV)
    data['_pred'] = [0.]*len(data)
    if isinstance(model, xgb.sklearn.XGBRegressor) or isinstance(model, xgb.sklearn.XGBClassifier):
        if isinstance(model, xgb.sklearn.XGBRegressor):
            data['_pred'] = [0.5]*len(data)
        dmp = model.get_booster().get_dump()
        treeinfo = [{'tree_structure':convert_str_to_tree(tree)} for tree in dmp]
    else:
        treeinfo = model._Booster.dump_model()["tree_info"]
    for now, tree in enumerate(treeinfo):
        if global_debug: print('now :',now)
        listG, listH = [], []
        for y, p in zip(data['_label'], data['_pred']):
            listG.append(losstool.calc_g(y, p))
            listH.append(losstool.calc_h(y, p))
        data['_G'] = listG
        data['_H'] = listH
        gogogo(tree['tree_structure'], feat_imp, data, data, biased)
    for f, s in zip(global_feat_list, feat_imp):
        if global_debug: print(f,s)
    # print(data['_pred'])
    return feat_imp
        
        


