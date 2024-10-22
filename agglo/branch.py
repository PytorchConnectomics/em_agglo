import numpy as np
from scipy.ndimage import binary_erosion
import mahotas
import fastremap, cc3d
import waterz
from em_util.seg import segs_to_iou
from em_util.io import get_query_in, get_query_count_dict, get_query_count, compute_bbox_all

def agglo_branch_2d(aff, waterz_param):    
    seg_size = aff.shape[1:]    
    # step 1: 2D seg
    seg2d = np.zeros(seg_size, np.uint32)
    do_rebuild = True
    max_id = 0
    for z in range(seg_size[0]):
        seg = waterz.waterz(aff[:, z: z+1], waterz_param['wz_thres'], merge_function = waterz_param['wz_mf'], \
                                    aff_threshold = [waterz_param['wz_low'], waterz_param['wz_high']], \
                                    fragments_opt = waterz_param['opt_frag'], fragments_seed_nb = waterz_param['nb'],\
                                    bg_thres = waterz_param['bg_thres'], rebuild=do_rebuild)[0][0]        
        seg_seed = cc3d.connected_components(binary_erosion(seg) * seg, connectivity=6)
        seg_mask = seg==0
        seg = mahotas.cwatershed(seg_mask, seg_seed)
        seg[seg_mask] = 0   
        seg, _ = fastremap.renumber(seg, in_place=True) 
        
        seg_max = seg.max()
        seg[seg > 0] += max_id        
        max_id += seg_max        
        seg2d[z] = seg        
        do_rebuild = False
        # print(z, max_id)
    
    # step 2: compute 2D stats
    ## z-direction region graph
    rg_id, rg_score = waterz.getRegionGraph(aff, seg2d, 2, waterz_param['wz_mf'], rebuild=False)   
    ## 2D IoU
    get_seg = lambda x: seg2d[x]
    iouF = segs_to_iou(get_seg, range(seg2d.shape[0]))
    iouB = segs_to_iou(get_seg, range(seg2d.shape[0])[::-1])
    
    return seg2d, rg_id, rg_score, iouF, iouB


def agglo_branch_3d(iouF, rg_id, rg_score, branch_param):
    bh_iou = branch_param['bh_iou']
    bh_sz_ratio = branch_param['bh_sz_ratio']
    bh_rg = branch_param['bh_rg']
    ## pairs with big IoU
    iou = np.vstack(iouF)
    score = iou[:, 4].astype(float) / (iou[:, 2] + iou[:, 3] - iou[:, 4])
    gid = iou[score >= bh_iou, :2].astype(np.uint32)
    ## pairs with big size ratio and at least one of them is not captured above
    score2 = iou[:, 2].astype(float) / iou[:, 3]
    gid2 = iou[(score2 >= bh_sz_ratio) * \
               (score2 <= 1/bh_sz_ratio), :2].astype(np.uint32)
    singleton = np.in1d(gid2.ravel(), gid.ravel(), invert=True).reshape(gid2.shape).max(axis=1)
    gid = np.vstack([gid, gid2[singleton]])
    ## rg check
    arr2str = lambda x: f'{min(x)}-{max(x)}'            
    score = get_query_count_dict(rg_id, arr2str, rg_score, gid)
    gid = gid[score < bh_rg]
    ## merge into 3D supervoxel
    relabel = waterz.region_graph.merge_id(gid[:, 0], gid[:, 1])
    return relabel
   
def agglo_branch_3d_vertical(seg, relabel, iouF, iouB, rg_id, rg_score, branch_param):
    bh_rg = branch_param['bh_rg']
    bbox = compute_bbox_all(seg, True)    
    iouF2 = [None] * len(iouF)
    iouB2 = [None] * len(iouF)
    for z in range(len(iouF)):
        iouF2[z] = iouF[z][:,:3]
        iouF2[z][:, 2] = z
        iouB2[z] = iouB[z][:,:2]            
    iouF2 = np.vstack(iouF2)    
    iouB2 = np.vstack(iouB2)
    
    # unique pairs after relabel
    iou_new = iouF2[:, :2].copy()
    iou_new[iou_new < len(relabel)] = relabel[iou_new[iou_new < len(relabel)] ]
    is_unique = (iou_new[:,0]-iou_new[:,1]) != 0
    iou_unique = iou_new[is_unique]
    iou_z = iouF2[is_unique, 2]
    is_last = get_query_count(bbox[:,0], bbox[:,2], iou_unique[:,0]) == iou_z
    is_first = get_query_count(bbox[:,0], bbox[:,1], iou_unique[:,1]) == (iou_z + 1)
    mid = iouF2[is_unique,:2][is_last*is_first]
    
    ## rg check
    arr2str = lambda x: f'{min(x)}-{max(x)}'            
    score = get_query_count_dict(rg_id, arr2str, rg_score, mid)
    mid = mid[score < bh_rg]
    
    
    arr2str2 = lambda x: f'{x[0]}-{x[1]}'
    mid_bb = get_query_in(iouB2[:,:2], arr2str2, mid[:, ::-1])
    mid = mid[mid_bb].astype(np.uint32)
    mid[mid < len(relabel)] = relabel[mid[mid < len(relabel)]]
        
    relabel2 = waterz.region_graph.merge_id(mid[:, 0], mid[:, 1])
    return relabel2

def agglo_branch_3d_horizontal(seg, relabel, relabel2, iouF, iouB, rg_id, rg_score, branch_param):
    bh_1side_ratio = branch_param['bh_1side_ratio']
    bh_1side_sz = branch_param['bh_1side_sz']
    bh_rg = branch_param['bh_rg']
    # 1. get unique iou pairs after relabel
    iou = np.vstack(iouF+iouB)    
    iou_new = iou.copy()
    tmp = iou_new[:, :2]
    tmp[tmp < len(relabel)] = relabel[tmp[tmp < len(relabel)]]
    tmp[tmp < len(relabel2)] = relabel2[tmp[tmp < len(relabel2)]]
    is_unique = (iou_new[:,0]-iou_new[:,1]) != 0
    iou_unique = iou_new[is_unique]

    # 2. get candidate list: both are to_merge, 1sided IoU
    bbox = compute_bbox_all(seg, True)            
    num_b = (bbox[:,1:-1:2] == 0).sum(axis=1) + \
            (bbox[:,2] == seg.shape[0]-1) + (bbox[:,4] == seg.shape[1]-1) + \
            (bbox[:,6] == seg.shape[2]-1)
    to_merge = bbox[num_b<=1, 0]
    is_candidate = np.in1d(iou_unique[:,:2].ravel(), to_merge).reshape(-1,2).min(axis=1)    
    score_1side = iou_unique[:, 4] / iou_unique[:, 2]
    mid = iou[is_unique,:2][is_candidate * (score_1side >= bh_1side_ratio) * (iou_unique[:,4] >= bh_1side_sz)]

    # 3. check affinity            
    arr2str = lambda x: f'{min(x)}-{max(x)}'
    score = get_query_count_dict(rg_id, arr2str, rg_score, mid)
    mid = mid[score < bh_rg]
    mid[mid < len(relabel)] = relabel[mid[mid < len(relabel)]]
    mid[mid < len(relabel2)] = relabel2[mid[mid < len(relabel2)]]
    mid = np.hstack([mid.min(axis=1).reshape(-1,1), mid.max(axis=1).reshape(-1,1)])
    mid = np.unique(mid, axis=1)

    # 4. ordered agglomeration    
    ratio = ((bbox[:,2::2] - bbox[:,1:-1:2] + 1) / np.array(seg.shape)).max(axis=1)
    num_z = bbox[:,2]-bbox[:,1]+1
    ## remove singleton used more than once
    uid, uc = np.unique(mid.ravel(), return_counts=True)
    uid_z = get_query_count(bbox[:,0], num_z, uid)
    bid = uid[(uid_z==1) * (uc>1)]
    mid = mid[np.in1d(mid.ravel(), bid, invert=True).reshape(-1,2).min(axis=1)]
    uid = uid[np.in1d(uid, bid, invert=True)]
    ## greedy approach
    uid_r = get_query_count(bbox[:,0], ratio, uid)
    uid_ratio = dict(zip(uid, uid_r))
    uid_num_b = dict(zip(uid, get_query_count(bbox[:,0], num_b, uid)))
    uid_done = dict(zip(uid, np.zeros(len(uid))))
    mid_sel = []
    uid_sorted = uid[np.argsort(-uid_r)]
    for i in uid_sorted:
        if uid_done[i] == 0:
            # uid[i] involved and not done
            total_num_b = 0
            todo_id = i
            uid_done[todo_id] = 1
            total_num_b = uid_num_b[todo_id]
            todo_stat = np.zeros([0, 3])
            while total_num_b < 2:
                new_mid = mid[(mid == todo_id).max(axis=1)]
                new_id = new_mid[new_mid != todo_id]
                new_id_ratio = np.array([uid_ratio[x] if uid_done[x]==0 else 0 for x in new_id])
                new_stat = np.hstack([np.ones([len(new_id),1])*todo_id, new_id.reshape(-1,1), new_id_ratio.reshape(-1,1)])
                todo_stat = np.vstack([todo_stat, new_stat])
                if len(todo_stat)==0:
                        break
                else:
                    todo_sel = np.argmax(todo_stat[:,2])
                    if todo_stat[todo_sel, 2] == 0:
                        break
                todo_id = int(todo_stat[todo_sel, 1])
                uid_done[todo_id] = 1
                total_num_b += uid_num_b[todo_id]
                mid_sel += [list(todo_stat[todo_sel,:2])]
                todo_stat[todo_sel] = 0
    mid = np.vstack(mid_sel).astype(np.uint32)
    relabel3 = waterz.region_graph.merge_id(mid[:, 0], mid[:, 1])
    return relabel3
    
            
def agglo_branch(aff, waterz_param=None, branch_param=None):    
    if waterz_param is None:
        waterz_param = {'wz_low': 40, 'wz_high' : 250, 'wz_mf' : 'aff75_his256_ran255', 'wz_thres' : [5], \
            'opt_frag' : 1, 'nb' : 5, 'bg_thres' : 0.95}
    if branch_param is None:
        branch_param = {'bh_iou' : 0.8, 'bh_sz_ratio' : 0.6, 'bh_rg' : 32, \
            'bh_1side_ratio' : 0.6, 'bh_1side_sz' : 128}
    
    # step 1: 2D seg
    seg, rg_id, rg_score, iouF, iouB = agglo_branch_2d(aff, waterz_param)                
    
    # step 2: 2D -> 3D supervoxel    
    relabel = agglo_branch_3d(iouF, rg_id, rg_score, branch_param)
    seg[seg<len(relabel)] = relabel[seg[seg<len(relabel)]]
        
    # step 3: 3D supervoxel: vertical merge
    relabel2 = agglo_branch_3d_vertical(seg, relabel, iouF, iouB, rg_id, rg_score, branch_param)
    seg[seg<len(relabel2)] = relabel2[seg[seg<len(relabel2)]]
    
    # step 4: 3D supervoxel: horizontal merge 
    relabel3 = agglo_branch_3d_horizontal(seg, relabel, relabel2, iouF, iouB, rg_id, rg_score, branch_param)
    seg[seg<len(relabel3)] = relabel3[seg[seg<len(relabel3)]]
    return seg