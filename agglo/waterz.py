import numpy as np
import waterz

def agglo_waterz(aff, waterz_param=None):
    if aff.dtype == np.uint8:
        aff = aff.astype(np.float32)/255.
    if waterz_param is None:
        waterz_param = {'wz_low': 0.1, 'wz_high' : 0.95, 'wz_mf' : 'aff75_his256', 'wz_thres' : [0.6]}            
        
    out = waterz.waterz(aff, waterz_param['wz_thres'], merge_function=waterz_param['wz_mf'],
                        aff_threshold=[waterz_param['wz_low'], waterz_param['wz_high']], return_seg=True)    
    return out