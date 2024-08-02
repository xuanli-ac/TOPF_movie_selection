import os
import ast
import sys
import pandas as pd
import numpy as np

##### calculate scores from predictions
 
############################################## project and result folders

# Project folder
wkdir = '/data/project/brainvar_topf_eval'
r_rootdir = '/data/project/brainvar_topf_eval/results/neoregression/'

# adding core_functions folder to the system path and import the TOPF module (TOPF.py)
sys.path.insert(0, wkdir+'/code/core_functions')
import TOPF

############################################## the settings to be analysed
phenostr = 'PMAT24_A_CR'
seedlist = list(np.arange(0,10,1))         # needs to be a list
feature_type = 'combinedPC'                  # 'singlePC' or combinedPC
pcind = 2                                  # which PC or to the largest PC index when combinedPC
k_outer = 10                               # 10, number of folds for outer cv
k_inner = 5                                # 5, number of folds for inner cv
nroi = 436                                 # number of rois/features
dataset = 'hcp436'
cutntr = 132
clfname = 'ridge'
probtype = 'regression'
metric_list = ['pearson', 'spearman', 'r2']


cmp_name = 'wholerun' # "moviewise" or "wholerun" or path to dataframe
ntrlist = [120, 240, 360, 480, 600]
#ntrlist = [132]
thre = 0
clfdir = f'ridge_thre{thre}'
phenolist = ['PMAT24_A_CR']

########################################## where to save the results

filepath = '/data/project/brainvar_topf_eval/results_summary/neoregression/'+clfdir+'/'
if not os.path.exists(f"{filepath}"):
    os.makedirs(f"{filepath}")

##################################################### no need to change below ####################################################################

if cmp_name =='moviewise':
    movienames = ['two_men','bridgeville','pockets','overcome','inception','social_net', 'oceans_11', 'flower', 'hotel', 'garden', 'dreary', 'home_alone', 'brokovich', 'star_wars']
elif cmp_name == 'wholerun':
    movienames = ['run'+str(i) for i in np.arange(1,5,1)]
elif cmp_name is not None:
    movienames = [movieid]
    cmp_name = movieid
else:
    print('please specifiy whether moviewise or wholerun')

############################ for changing cutntr
for cutntr in ntrlist:

# ########################### for changing phenostr
# for phenostr in phenolist:    
    ################### initialise
    if cutntr <=0:
        ntr = None  # full length 
    else:
        ntr = cutntr # cut to cutntr
    
    rfolder = f'{dataset}/{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'
    rdir = r_rootdir+rfolder
    # filename = filepath + f'full_scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}.csv'
    # filename2 = filepath + f'scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}.csv'

    filename = filepath + f'full_scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_thre{thre}.csv'
    filename2 = filepath + f'scores_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_thre{thre}.csv'

    # start calculate scores (defined in metric_list) for test data
    savepath = []
    nfold = k_outer
    foldwise = 0  # compute the score for each fold separately

    # for results of all folds, seeds and movies
    mlabel = []
    df_full_scores = pd.DataFrame()

    # for results of all seeds and movies
    m_vals = np.empty([0, len(metric_list)])
    seedlabel = []
    mlabel2 = []

    ################### initialise

    # # if foldwise == 1
    # for fmricondition in movienames:
    #     print(fmricondition)

    #     Result_struct = TOPF.init_result_dir(rdir, clfdir+'/'+phenostr)
    #     _, mean_seed, df_temp, m_labels = TOPF.main_compute_prediction_scores(fmricondition, clfname, probtype, Result_struct, metric_list, seedlist, nfold, savepath, foldwise, test=True)
    #     df_full_scores = pd.concat([df_full_scores, df_temp], ignore_index=True)
    #     mlabel = mlabel + [fmricondition] * (len(seedlist)*nfold)

    #     m_vals = np.append(m_vals, mean_seed, axis = 0)
    #     seedlabel = seedlabel + seedlist
    #     mlabel2 = mlabel2 + [fmricondition] * len(seedlist)

    # # save scores of all folds, seeds and movies
    # df_full_scores.insert(0,'movie', mlabel)
    # df_full_scores.to_csv(filename, header=True)
    # print('saved computed scores of all folds and seeds to', filename)   

    # # save scores of all seeds and movies
    # df_scores = pd.DataFrame(data=m_vals,columns=m_labels)
    # df_scores.insert(0,'movie', mlabel2)
    # df_scores.insert(0,'seed', seedlabel)
    # df_scores.to_csv(filename2, header=True)
    # print('saved computed scores to', filename2)

    # foldwise = 0
    for fmricondition in movienames:
        print(fmricondition)

        Result_struct = TOPF.init_result_dir(rdir, clfdir+'/'+phenostr)
        _, mean_seed, df_temp, m_labels = TOPF.main_compute_prediction_scores(fmricondition, clfname, probtype, Result_struct, metric_list, seedlist, nfold, savepath, foldwise, test=True)
        
        m_vals = np.append(m_vals, mean_seed, axis = 0)
        seedlabel = seedlabel + seedlist
        mlabel2 = mlabel2 + [fmricondition] * len(seedlist)

    # save scores of all seeds and movies
    df_scores = pd.DataFrame(data=m_vals,columns=m_labels)
    df_scores.insert(0,'movie', mlabel2)
    df_scores.insert(0,'seed', seedlabel)
    df_scores.to_csv(filename2, header=True)
    print('saved computed scores over all samples within each seed to', filename2)