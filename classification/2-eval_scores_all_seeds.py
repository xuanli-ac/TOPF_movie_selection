import os
import ast
import sys
import pandas as pd
import numpy as np

##### calculate scores from predictions
 
############################################## project and result folders

# Project folder
wkdir = '/data/project/brainvar_topf_eval'
r_rootdir = '/data/project/brainvar_topf_eval/results/classification/feature_before_norm/'

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
#cutntr = 170
clfname = 'svm'
probtype = 'binary_classification'
metric_list = ['accuracy', 'balanced_accuracy', 'roc_auc']


cmp_name = 'moviewise' # "moviewise" or "wholerun" or path to dataframe
# cmp_name = wkdir+'/data/HCP/Schaefer436/df_X_m_5_13_12_6_9_tr_120_120_120_120_120.csv'   
# movieid = 'comb_120tr_top5m'
#ntrlist = [30, 60, 90, 120, 150, 180, 210, 240]
#ntrlist = [120, 240, 360, 480, 600]

ntrlist = [132]
thre = 0
#clfdir = f'svm_rbf_gamma_thre{thre}'
clfdir = f'ridge_clf_thre{thre}'
nstart = 0
########################################## where to save the results

filepath = '/data/project/brainvar_topf_eval/results_summary/classification/'+clfdir+'/'
if not os.path.exists(f"{filepath}"):
    os.makedirs(f"{filepath}")

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

############################ for changing cutntr
for cutntr in ntrlist:
    
    ################### initialise
    if cutntr <=0:
        ntr = None  # full length 
    else:
        ntr = cutntr # cut to cutntr
    
    if nstart ==0:
        rfolder = f'{dataset}/{phenostr}/{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'
        filename = filepath + f'full_scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_thre{thre}.csv'
        filename2 = filepath + f'scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_thre{thre}.csv'
    else:
        rfolder = f'{dataset}/{phenostr}/{cmp_name}_cut_{str(ntr)}_from{nstart}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'
        filename = filepath + f'full_scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_from{nstart}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_thre{thre}.csv'
        filename2 = filepath + f'scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_from{nstart}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_thre{thre}.csv'
    rdir = r_rootdir+rfolder
    
    # filename = filepath + f'full_scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}.csv'
    # filename2 = filepath + f'scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}.csv'

    # filename = filepath + f'full_scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_thre{thre}.csv'
    # filename2 = filepath + f'scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_thre{thre}.csv'

    # start calculate scores (defined in metric_list) for test data
    savepath = []
    nfold = k_outer
    foldwise = 1  # compute the score for each fold separately

    # for results of all folds, seeds and movies
    mlabel = []
    df_full_scores = pd.DataFrame()

    # for results of all seeds and movies
    m_vals = np.empty([0, len(metric_list)])
    seedlabel = []
    mlabel2 = []

    ################### initialise

    for fmricondition in movienames:
        print(fmricondition)

        Result_struct = TOPF.init_result_dir(rdir, clfdir)
        _, mean_seed, df_temp, m_labels = TOPF.main_compute_prediction_scores(fmricondition, clfname, probtype, Result_struct, metric_list, seedlist, nfold, savepath, foldwise, test=True)
        df_full_scores = pd.concat([df_full_scores, df_temp], ignore_index=True)
        mlabel = mlabel + [fmricondition] * (len(seedlist)*nfold)

        m_vals = np.append(m_vals, mean_seed, axis = 0)
        seedlabel = seedlabel + seedlist
        mlabel2 = mlabel2 + [fmricondition] * len(seedlist)

    # save scores of all folds, seeds and movies
    df_full_scores.insert(0,'movie', mlabel)
    df_full_scores.to_csv(filename, header=True)
    print('saved computed scores of all folds and seeds to', filename)   

    # save scores of all seeds and movies
    df_scores = pd.DataFrame(data=m_vals,columns=m_labels)
    df_scores.insert(0,'movie', mlabel2)
    df_scores.insert(0,'seed', seedlabel)
    df_scores.to_csv(filename2, header=True)
    print('saved computed scores to', filename2)

