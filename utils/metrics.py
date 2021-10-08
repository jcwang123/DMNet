import numpy as np

def compute_dice(pre,ref,return_all=False):
    # n*c*w*h
    pre = pre>0.5
    ref = ref>0.5
    assert pre.shape==ref.shape
    class_num = pre.shape[1]
    dice = np.zeros((class_num-1,))
    for c in range(1, class_num):
        index = list(np.sum(ref[:,c],axis=(1,2))>10)
        p = pre[:,c][index]
        r = ref[:,c][index]
        dice[c-1] = np.mean(2*np.sum(p*r,axis=(1,2))/(np.sum(p,axis=(1,2))+np.sum(r,axis=(1,2))))
    if return_all:
        return dice
    else:
        return np.mean(dice[dice>-1])     

def compute_iou(pre,ref, return_all=False):
    pre = pre>0.5
    ref = ref>0.5
    assert pre.shape==ref.shape
    class_num = pre.shape[1]
    iou = np.zeros((class_num-1,))
    for c in range(1, class_num):
        index = list(np.sum(ref[:,c],axis=(1,2))>10)
        p = pre[:,c][index]
        r = ref[:,c][index]
        iou[c-1] = np.mean(np.sum(p*r,axis=(1,2))/(np.sum(p,axis=(1,2))+np.sum(r,axis=(1,2))-np.sum(p*r,axis=(1,2))))
    if return_all:
        return iou
    else:
        return np.mean(iou[iou>-1])