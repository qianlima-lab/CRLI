# -*- coding: utf-8 -*-

import numpy as np
miss_value = -1
with open('house-votes-84.data') as f:
    lines = f.readlines()
    lines = [i.replace('republican','1')
    .replace('democrat','0')
    .replace('y','1')
    .replace('n','0')
    .replace('?',str(miss_value))
    .split(',') for i in lines]
    lb_dt = np.array(lines,dtype=np.int32)
    mask = np.where(lb_dt != miss_value,np.ones_like(lb_dt),np.zeros_like(lb_dt))
    length = np.ones(lb_dt.shape[0])*(lb_dt.shape[1]-1)
    lengthmark = np.ones([lb_dt.shape[0],lb_dt.shape[1]-1])
    np.savetxt('HouseVote_TRAIN',lb_dt,'%d',',')
    np.savetxt('mask_HouseVote_TRAIN',mask,'%d',',')
    np.savetxt('length_HouseVote_TRAIN',length,'%d',',')
    np.savetxt('lengthmark_HouseVote_TRAIN',lengthmark,'%d',',')