from eli5.sklearn import PermutationImportance
import os
import ast
import sys
import pandas as pd
import numpy as np
import pickle
from julearn.utils import configure_logging

## configure_logging_julearn
configure_logging(level='INFO')


############## read in arguments 

################################### read from bash when running on cluster ###########################################
wkdir = sys.argv[1]                     # Project folder: wkdir = '/Users/xli/Desktop/Github/TOPF_evaluation'
r_rootdir = sys.argv[2]                 # result root dir: '/Users/xli/Desktop/Github/TOPF_evaluation/results/'
dataset = sys.argv[3]                   # 'hcp268'
nroi = int(sys.argv[4])                 # number of rois/features
movieind = int(sys.argv[5])             # or run ind, ranging from 1-14 or 1-4
cmp_name = sys.argv[6]                  # "moviewise" or "wholerun"
phenostr = sys.argv[7]                  # 'Gender'
cutntr = int(sys.argv[8])              # cutntr<=0: using full length; otherwise: cut to the given length e.g., ntr=180
feature_type = sys.argv[9]             # 'singlePC' or combinedPC
pcind = int(sys.argv[10])               # which PC or to the largest PC index when combinedPC
clfname = sys.argv[11]                  # e.g., 'svm', which classifier (see all availabel classifiers here: https://juaml.github.io/julearn/main/steps.html)
k_outer = int(sys.argv[12])             # number of folds for outer cv
n_perm = int(sys.argv[13])              # number of permutations
scoring = sys.argv[14]                  # 'balanced_accuracy', https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
clfdir = sys.argv[15]

if cutntr <=0:
    ntr = None  # full length 
else:
    ntr = cutntr # cut to cutntr

# adding core_functions folder to the system path and import the TOPF module (TOPF.py)
sys.path.insert(0, wkdir+'/code/core_functions')
import TOPF

################################### read from bash when running on cluster ###########################################

# fixed values
seedlist = list(np.arange(0,10,1))  
k_inner = 5
norm = 1
############################ HCP dataset specific info #######################

movienames = ['two_men','bridgeville','pockets','overcome','inception','social_net', 'oceans_11', 'flower', 'hotel', 'garden', 'dreary', 'home_alone', 'brokovich', 'star_wars']
# compute for each movie clip (1-14)
if cmp_name == 'moviewise':
    if movieind >=1 and movieind <=4:
        run = 1
    elif movieind >=5 and movieind <=7:
        run = 2
    elif movieind >=8 and movieind <=11:
        run = 3
    elif movieind >=12 and movieind <=14:
        run = 4
    else:
        print('movie index should be an integer in range 1 to 14')  
    fmricondition = movienames[movieind-1]
 # or for the whole movie run (1-4)    
elif cmp_name == 'wholerun':
    fmricondition = 'run'+str(movieind)
else:
    print('please specify "moviewise" or "wholerun" ')

#^^^^^^^^^^^^^^^^^^^^^ HCP dataset specific info ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    

##################### where to find data

rfolder = f'{dataset}/{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'
rdir = r_rootdir+rfolder
Result_struct = TOPF.init_result_dir(rdir,clfdir+f'/{phenostr}')
featuredir = Result_struct.featuredir
modeldir = Result_struct.modeldir
preddir = Result_struct.preddir

# use features created for sex classification if available
fdir = f'/data/project/brainvar_topf_eval/results/classification/feature_before_norm/hcp{nroi}/Gender/'
ftem = f'{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'+'/features/'
fdir_exist_sex = fdir + ftem
print(fdir_exist_sex)
if os.path.exists(f"{fdir_exist_sex}"):
    Result_struct.featuredir = fdir_exist_sex
    featuredir = fdir_exist_sex

########################################## where to save the results
fimpdir = wkdir + '/results_summary'+'/feature_weights/'
if not os.path.exists(f"{fimpdir}"):
    os.makedirs(f"{fimpdir}")


############################################# Compute permutation feature importance over seeds and folds #######################################
print('current condition:', fmricondition)
print('current dataset:', dataset)
print('number of rois used:', nroi)

fw = []    # feature weights
score = [] # true_pred_score, double check by comparing with our computed scores
sind = []
find = []

for seed in seedlist:
    print('current seed:', seed)
    
    for foldind in range(0, k_outer, 1):
        print('current fold:', foldind+1)
         # load the fitted model
        filename = eval(Result_struct.modelfname)
        model = pickle.load(open(filename, 'rb'))

        # load test_data
        filename = eval(Result_struct.featurefname_test)
        test_data = pd.read_csv(filename)
        X_keys = [col for col in test_data.columns if 'PC' in col]
        y_keys = phenostr
        sub_test = test_data.Subject.values

        # normalise feature (fit on train and apply to test)
        filename_train = eval(Result_struct.featurefname_train)
        train_data = pd.read_csv(filename_train)
        F = train_data[X_keys].values
        Ft = test_data[X_keys].values
        F,Ft = TOPF.normalise_feature(F,Ft)
        test_data[X_keys] = Ft
        train_data[X_keys] = F

        # get scores of the phenotype
        df_true = pd.read_csv(eval(Result_struct.predfname_test))
        test_true = df_true[df_true.Subject.isin(sub_test)]['true'].values

        # permutation
        perm = PermutationImportance(model[clfname], scoring=scoring, random_state=42, n_iter=n_perm).fit(test_data[X_keys], test_true)
        fw_temp = perm.results_      # n_perm * n_roi
        fw = fw + fw_temp
        
        score_temp = perm.scores_.mean() # true_pred_score
        score = score + [score_temp]
    
    find = find + list(np.arange(0,k_outer,1))*n_perm
sind = sind + [seed]*n_perm*k_outer

# save results
fw = np.array(fw)
df_fw = pd.DataFrame(data=fw,columns=X_keys)
pind = list(np.arange(0,n_perm,1))* k_outer * len(seedlist)
df_fw.insert(0,'perm',pind)
df_fw.insert(0,'fold',find)
df_fw.insert(0,'seed',seed)

filename = fimpdir + f'fw_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}_{scoring}_{clfdir}_{fmricondition}_nperm{n_perm}.csv'
df_fw.to_csv(filename)
print('file saved as:', filename)

# print results
print('true pred scores:', score)
m = np.mean(fw,axis=0) # average over n_perm
m = m*-1
ranked = np.argsort(m)
s = np.sort(m)
print('top 10 important rois: ',ranked[0:10])
print('top 10 roi importance:',s[0:10]*-1)
