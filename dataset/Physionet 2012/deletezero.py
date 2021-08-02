# -*- coding: utf-8 -*-

import numpy as np

mtrain=np.sum(np.loadtxt('mask_Physionet 2012_TRAIN',delimiter=',')[:,1:],axis=1)
mtest=np.sum(np.loadtxt('mask_Physionet 2012_TEST',delimiter=',')[:,1:],axis=1)

trainidx = np.where(mtrain==0)
testidx=np.where(mtest==0)