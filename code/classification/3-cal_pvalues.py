import os
import ast
import sys
import pandas as pd
import numpy as np

##### calculate p-values from scores derived on permuted data


############################################## project and result folders

# Project folder
wkdir = '/data/project/brainvar_topf_eval'

# adding core_functions folder to the system path and import the TOPF module (TOPF.py)
sys.path.insert(0, wkdir+'/code/core_functions')
import TOPF

############################################## the settings to be analysed
phenostr = 'Gender'
seedlist = list(np.arange(0,10,1))         # needs to be a list
feature_type = 'singlePC'                  # 'singlePC' or combinedPC
pcind = 1                                  # which PC or to the largest PC index when combinedPC
k_outer = 10                               # 10, number of folds for outer cv
k_inner = 5                                # 5, number of folds for inner cv
nroi = 436                                 # number of rois/features
dataset = 'hcp436'
cutntr = 132
clfname = 'svm'
probtype = 'binary_classification'
metric_list = ['accuracy', 'balanced_accuracy', 'roc_auc']


cmp_name = 'moviewise' # "moviewise" or "wholerun" or path to dataframe
# cmp_name = wkdir+'/data/HCP/Schaefer436/df_X_m_5_13_12_6_9_tr_120_120_120_120_120.csv'   
# movieid = 'comb_120tr_top5m'
#ntrlist = [30, 60, 90, 120, 150, 180, 210, 240]
#ntrlist = [120, 240, 360, 480, 600]

#ntrlist = [132]
thre = 0
clfdir = f'svm_rbf_gamma_thre{thre}'
nstart = 0

##################################################### no need to change below ####################################################################

if cmp_name =='moviewise':
    movienames = ['two_men','bridgeville','pockets','overcome','inception','social_net', 'oceans_11', 'flower', 'hotel', 'garden', 'dreary', 'home_alone', 'brokovich', 'star_wars']
    #movienames = ['two_men','bridgeville','pockets','inception','social_net', 'oceans_11', 'flower', 'hotel', 'garden', 'home_alone', 'brokovich', 'star_wars']
elif cmp_name == 'wholerun':
    movienames = ['run'+str(i) for i in np.arange(1,5,1)]
elif cmp_name is not None:
    movienames = [movieid]
    cmp_name = movieid
else:
    print('please specifiy whether moviewise or wholerun')

if cutntr <=0:
    ntr = None  # full length 
else:
    ntr = cutntr # cut to cutntr

if nstart ==0:
    rfolder = f'{dataset}/{phenostr}/{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'
    
else:
    rfolder = f'{dataset}/{phenostr}/{cmp_name}_cut_{str(ntr)}_from{nstart}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'
        
######################################################################################

def cal_p_value_perm(rpdir,permseedlist,met,fmricondition,df_original):

    # calculate the p-value based on null distribution of prediction performance scores derived from permuted data
    perm_all = []
    for permseed in permseedlist:
        score_fname = rpdir + f'/p{permseed}_scores_{fmricondition}.csv'
        df_temp = pd.read_csv(score_fname)
        score_p = df_temp[[met]].values.mean() # average over all CV repetitions
        perm_all = perm_all + [score_p]
    perm_all = np.array(perm_all)
    # original value
    df_otemp = df_original[df_original.movie ==fmricondition]
    score_o = df_otemp[[met]].values.mean()
    p_val = len(perm_all[perm_all>=score_o])+1 / (len(perm_all)+1)

    return p_val, score_o


# load original values
rdir = wkdir +f'/results_summary/classification/{clfdir}'
fname = rdir + f'/scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_thre{thre}.csv'
if os.path.isfile(fname):
    df_original = pd.read_csv(fname)
else:
    print('cannot find the real prediction scores')


# null distribution
r_rootdir = wkdir + '/results/classification/feature_before_norm/'

rpdir = r_rootdir +'perms/'+rfolder
permseedlist = np.arange(1,1001)
met = 'balanced_accuracy'

# loop over all movies
P_all = []
for fmricondition in movienames:
    p_val, score_o  = cal_p_value_perm(rpdir,permseedlist,met,fmricondition,df_original)
    print(fmricondition, 'original score:',score_o,'p-val:',p_val)
    P_all = P_all + [p_val]

P_all = np.reshape(P_all,[1,-1])
df_pval = pd.DataFrame(data=P_all,columns=movienames)

# where to save results
fname_p = rdir + f'/pvals_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_thre{thre}.csv'
df_pval.to_csv(fname_p)
print(df_pval)
print('file saved to:',fname_p)


