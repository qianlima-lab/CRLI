# coding: utf-8

import os
import re
import numpy as np
import pandas as pd
import json as json
import pdb 

patient_ids = []

for filename in os.listdir('./set-a'):
    # the patient data in PhysioNet contains 6-digits
    match = re.search('\d{6}', filename)
    if match:
        id_ = match.group()
        patient_ids.append(id_)


out = pd.read_csv('./set-a/Outcomes-a.txt').set_index('RecordID')['In-hospital_death']

# we select 35 attributes which contains enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']

# mean and std of 35 attributes
mean = np.array([59.540976152469405, 86.72320413227443, 139.06972964987443, 2.8797765291788986, 58.13833409690321,
                 147.4835678885565, 12.670222585415166, 7.490957887101613, 2.922874149659863, 394.8899400819931,
                 141.4867570064675, 96.66380228136883, 37.07362841054398, 505.5576196473552, 2.906465787821709,
                 23.118951553526724, 27.413004968675743, 19.64795551193981, 2.0277491155660416, 30.692432164676188,
                 119.60137167841977, 0.5404785381886381, 4.135790642787733, 11.407767149315339, 156.51746031746032,
                 119.15012244292181, 1.2004983498349853, 80.20321011673151, 7.127188940092161, 40.39875518672199,
                 191.05877024038804, 116.1171573535279, 77.08923183026529, 1.5052390166989214, 116.77122488658458])

std = np.array(
    [13.01436781437145, 17.789923096504985, 5.185595006246348, 2.5287518090506755, 15.06074282896952, 85.96290370390257,
     7.649058756791069, 8.384743923130074, 0.6515057685658769, 1201.033856726966, 67.62249645388543, 3.294112002091972,
     1.5604879744921516, 1515.362517984297, 5.902070316876287, 4.707600932877377, 23.403743427107095, 5.50914416318306,
     0.4220051299992514, 5.002058959758486, 23.730556355204214, 0.18634432509312762, 0.706337033602292,
     3.967579823394297, 45.99491531484596, 21.97610723063014, 2.716532297586456, 16.232515568438338, 9.754483687298688,
     9.062327978713556, 106.50939503021543, 170.65318497610315, 14.856134327604906, 1.6369529387005546,
     133.96778334724377])


def to_time_bin(x):
    h, m = map(int, x.split(':'))
    return h


def parse_data(x):
    x = x.set_index('Parameter').to_dict()['Value']
    values = []
    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(48):
        if h == 0:
            deltas.append(np.ones(35))
        else:
            deltas.append(np.ones(35) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).as_matrix()

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()

    return rec


def parse_id(id_):
    data = pd.read_csv('./set-a/{}.txt'.format(id_))
    # accumulate the records within one hour
    data['Time'] = data['Time'].apply(lambda x: to_time_bin(x))

    evals = []

    # merge all the metrics within one hour
    for h in range(48):
        dt = data[data['Time'] == h]
        evals.append(parse_data(dt))

    evals = (np.array(evals) - mean) / std


    # randomly eliminate 10% values as the imputation ground-truth
    masks = np.where(~np.isnan(evals),1,0)
    evals = np.where(np.isnan(evals),0,evals)
    label = int(out.loc[int(id_)])

    evals_rsp = np.reshape(evals,-1)
    masks_rsp = np.reshape(masks,-1)
    
    evals_rsp = np.hstack((label,evals_rsp))
    masks_rsp = np.hstack((label,masks_rsp))
    
    return evals_rsp,masks_rsp

#length = np.ones([4000],np.int32) * 48
#lengthmark = np.ones([4000,1680],np.int32)
#np.savetxt('length_Physionet 2012_TRAIN',length,'%d',delimiter=',')
#np.savetxt('lengthmark_Physionet 2012_TRAIN',lengthmark,'%d',delimiter=',')

zero=[]
data = []
masks = []
for id_ in patient_ids:
    print('Processing patient {}'.format(id_))
    values,mask = parse_id(id_)
    if np.sum(mask[1:]) == 0:
        zero.append(id_)
        continue
    data.append(values)
    masks.append(mask)
data = np.reshape(data,[len(patient_ids)-len(zero),-1])
masks = np.reshape(masks,[len(patient_ids)-len(zero),-1])

np.savetxt('Physionet 2012_TRAIN',data,'%f',delimiter=',')
np.savetxt('mask_Physionet 2012_TRAIN',masks,'%d',delimiter=',')

