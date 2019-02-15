#python E:\Bogdan\Dropbox\code_Seurat\ChromatinImaging\Scripts\BatchSequentialSmall2colV3.py 0
#External packages
import sys,glob,os
import numpy as np
import cPickle as pickle
import matplotlib.pylab as plt

#Internal packages
#add path
parent_dir =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir+os.sep+r'CommonTools')

#Internal imports
import IOTools as io
import FittingTools as ft
import AlignmentTools as at

#Extra functions

    

if __name__ == "__main__":
    ind = int(sys.argv[1])
    ### volatile ###########################
    master_folders = [r'N:\HCT116_6hAuxin_SI15noAB-Si16Ext',
                 r'\\169.254.192.20\Chromatin_NAS_1\Bogdan\HCT116_6hAuxin_SI15noAB-Si16Ext']
    #define analysis folder
    analysis_folder =r'Q:\Bogdan\NewAnalysis'+os.sep+os.path.basename(master_folders[0])
    #flat field correction
    im_repairs = np.load(analysis_folder+os.sep+'im_repairs_2col.npy')
    
    
    
    
    
    
    
    #get a list of hybe folders
    folders = ft.flatten([[folder for folder in glob.glob(master_folder+os.sep+'*') 
               if os.path.isdir(folder) and os.path.basename(folder)[0]=='H' and 
               np.any([let in os.path.basename(folder) for let in ['R']]) and
               np.all([let not in os.path.basename(folder)for let in ['H0','dapi']])
              ] for master_folder in master_folders])
    #keep complete sets
    num_daxs = np.array([len(glob.glob(folder+os.sep+'*.dax')) for folder in folders])
    folders = np.array(folders)[num_daxs==np.max(num_daxs)]
    #sort them by the order they were hybed
    folders = np.array(folders)[np.argsort(map(io.hybe_number,folders))]
    print "Found "+str(len(folders))+' folders.'
    fovs = np.sort(map(os.path.basename,glob.glob(folders[0]+os.sep+'*.dax')))
    ##Load the selected chromosme data
    dic = pickle.load(open(analysis_folder+os.sep+'Selected_Spot.pkl','r'))
    chr_pts = ft.partition_map(dic['coords'],dic['class_ids'])
    fov_ids = np.unique(dic['class_ids'])
    
    ###paramaters
    overwrite=False
    force_drift=False
    
    #drift paramaters
    ref=0 #drift reference hybe
    sz_ex=300 #the effective size of the cube to do drift correction
    hseed_beads=0 # The minimum bead height (prefitting)
    nbeads = 500 # The maximum number of beads
    


    #distance/numer of points cutoffs
    cutoff_window = 30 #fitted points within this distance will be reported
    cutoff_chr = 30 #if no prefered brightness points are detected within this distance, then include low brightness ones
    min_pts = 10 # number of fits/ window
    th_fluctuations = 2 # 2 std away for selecting a spot

    ##RUN

    from scipy.spatial.distance import cdist

    #iterate through good fields of view for which chromosomes have been selected
    folders_keep = list(folders[:])
    #itterate through fields of view
    #Get selected chromosomal positions
    fov_id = fov_ids[ind]
    file_=fovs[fov_id]
    chr_pts_ = chr_pts[ind] #positions of the selected centers of chromosomes in reference frame.
    chr_pts_ = np.array(chr_pts_)[:,::-1]#zxy coords of chromosomes already in the right position

    #Decide where to save the candidate positions of the hybe
    fl_cands = analysis_folder+os.sep+file_.replace('.dax','__current_cand.pkl')#file where to save candidates
    fl_cor = analysis_folder+os.sep+file_.replace('.dax','__drift.pkl')#file where to save drift correction
    fl_cor_fls = analysis_folder+os.sep+file_.replace('.dax','__driftfiles.npy')

    print fl_cands

    candid_spot={}

    #load data (pseudo-memory mapping)
    daxs_signal,names_signal,daxs_beads,names_beads = io.get_ims_fov_daxmap(folders_keep,file_,
                                                                         col_tags = None, pad = 10)
    #compute the drift for the field of view
    if len(daxs_beads)>1:
        txyz_both,_ = ft.get_STD_beaddrift_v2(daxs_beads,sz_ex=sz_ex,hseed=hseed_beads,nseed=nbeads,
                                           ref=None,force=force_drift,save_file=fl_cor)
        txyz = np.mean(txyz_both,1)
        txyz = np.array(txyz)-[txyz[ref]]
        np.save(fl_cor_fls,np.array(folders_keep))
    #repeat for colors
    #num_col = int(len(daxs_signal)/len(daxs_beads))
    #iterate through folders
    for iim in np.arange(len(daxs_signal))[:]:
        #iim specific vars
        im = daxs_signal[iim]
        
        ibead = im.index_beads
        icol = im.color
        tag = names_signal[iim]
        print tag
        txyz_ = txyz[ibead]

        #fit and update dictionary
        #check if recalculated
        candid_dic,candid_spot,drift_dic = {},{},{}
        bad_drift = False
        if os.path.exists(fl_cands):
            candid_spot,drift_dic = pickle.load(open(fl_cands,'rb'))
            txyz_old = drift_dic.get(tag,np.array([np.inf,np.inf,np.inf]))
            if np.linalg.norm(txyz_old-txyz_)>0.5:
                bad_drift=True
            key = np.max(candid_spot.keys())
            candid_dic,_ = candid_spot[key]
            
        if tag not in candid_dic or overwrite or bad_drift:
            ft.update_candid_spot(im,chr_pts_,txyz_,tag,drift_dic=drift_dic,th_std=th_fluctuations,num_pts=min_pts,
                            cutoff_window=cutoff_window,cutoff_chr=cutoff_chr,candid_spot=candid_spot,fl_cands=fl_cands,
                                  im_repair=im_repairs[icol],plt_val=False)