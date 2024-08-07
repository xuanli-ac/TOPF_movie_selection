import os
import ast
import sys
import pandas as pd
import numpy as np
from julearn.utils import configure_logging

## configure_logging_julearn
configure_logging(level='INFO')




############## read in arguments 

################################### read from bash ###########################################
# Project folder
wkdir = sys.argv[1] # wkdir = '/Users/xli/Desktop/Github/TOPF_evaluation'
file_subject_list = sys.argv[2] # wkdir+'/data/HCP/subject_list.txt'

# result root dir
r_rootdir = sys.argv[3] # '/Users/xli/Desktop/Github/TOPF_evaluation/results/classification/'

# adding core_functions folder to the system path and import the TOPF module (TOPF.py)
sys.path.insert(0, wkdir+'/code/core_functions')
import TOPF


dataset = sys.argv[4] # 'hcp268'
nroi = int(sys.argv[5])  # number of rois/features
seed = int(sys.argv[6])
movieind = sys.argv[7] # or run ind, ranging from 1-14 or 1-4, or a custom id
cmp_name = sys.argv[8] # "moviewise" or "wholerun" or a custom path to the dataframe
phenostr = sys.argv[9] # 'Gender'
cutntr = int(sys.argv[10]) # cutntr<=0: using full length; otherwise: cut to the given length e.g., ntr=180
feature_type = sys.argv[11] # 'singlePC' or combinedPC
pcind = int(sys.argv[12]) # which PC or to the largest PC index when combinedPC

clfname = sys.argv[13] # e.g., 'svm', which classifier (see all availabel classifiers here: https://juaml.github.io/julearn/main/steps.html)
k_outer = int(sys.argv[14]) # number of folds for outer cv
k_inner = int(sys.argv[15]) # number of folds for inner cv

# read in classfier settings
setting_file = sys.argv[16] # a path to the setting.txt file
clfdir = sys.argv[17]       # name of folder containing results regarding the given clfname
threshold = float(sys.argv[18]) # threshold for feature selection
nstart = int(sys.argv[19]) # start from the n-th TR (default = 0)


# probtype = sys.argv[16] # 'binary_classification'"
# print(sys.argv[17])
# param_keys = sys.argv[17] # "['svm__kernel', 'svm__C','scoring']"
# print(len(param_keys))
# print(sys.argv[18])
# param_values = eval(sys.argv[18]) # "[['linear','rbf'],[2**-1,2**0,2**1],'balanced_accuracy']"
# print(type(param_values))
# metric_list = eval(sys.argv[19]) # "['accuracy', 'balanced_accuracy', 'roc_auc']""
# print(metric_list)

if cutntr <=0:
    ntr = None  # full length 
    nstart = 0
else:
    ntr = cutntr # cut to cutntr
################################### read from bash ###########################################



############################ HCP dataset specific info #######################
# HCP data directory
ddir = wkdir +'/data/HCP'

# 7T subjects family structure (restricted info): downloaded from http://db.humanconnectome.org/
df_family_info = pd.read_csv(ddir+'/RESTRICTED_7T.csv')

# 7T subjects phenotype info (unrestricted info): downloaded from http://db.humanconnectome.org/
df_pheno = pd.read_csv(ddir+'/Subjects_Info.csv')

## cleaned: removed no-movie part and first 10 TRs of each movie clip, not used in computation just for info
df_movie_lengths = pd.read_csv(ddir+'/df_movie_lengths_cleaned.csv', index_col=None)
#print(df_movie_lengths)

# load the fmri dataframe of the specified movie clip or movie run as df_fmri (output of data_preparation)
if '268'in dataset:
    datadir = ddir+'/Shen268'
    if nroi>268:
        print('number of rois exceeds possible')
elif '436'in dataset:
    datadir = ddir+'/Schaefer436'
    if nroi>436:
        print('number of rois exceeds possible')
else:
    print('please specify the dataset name as "hcp268" or "hcp436')


movienames = ['two_men','bridgeville','pockets','overcome','inception','social_net', 'oceans_11', 'flower', 'hotel', 'garden', 'dreary', 'home_alone', 'brokovich', 'star_wars']
# compute for each movie clip (1-14)
if cmp_name == 'moviewise':
    movieind = int(movieind)
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
    df_fmri = pd.read_csv(datadir+'/session_'+str(run)+'/df_X_'+fmricondition+'.csv') 
 # or for the whole movie run (1-4)    
elif cmp_name == 'wholerun':
    movieind = int(movieind)
    fmricondition = 'run'+str(movieind)
    df_fmri = pd.read_csv(datadir+'/df_session_'+str(movieind)+'.csv')
elif os.path.isfile(cmp_name):
    print('use the dataframe specified:', cmp_name)
    fmricondition = movieind
    df_fmri = pd.read_csv(cmp_name)
    cmp_name = movieind
else:
    print('please specify "moviewise" or "wholerun" or give the correct path to the data')

#^^^^^^^^^^^^^^^^^^^^^ HCP dataset specific info ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    

##################### set up result directory structure and file names
if nstart ==0:
    rfolder = f'{dataset}/{phenostr}/{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'
else:
    rfolder = f'{dataset}/{phenostr}/{cmp_name}_cut_{str(ntr)}_from{nstart}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'

rdir = r_rootdir+rfolder

if not os.path.exists(f"{rdir}"):
    os.makedirs(f"{rdir}")

Result_struct = TOPF.init_result_dir(rdir, clfdir)


###################### load subject list to be analysed and check if missing values
with open(file_subject_list) as f:
    lines = [line.rstrip('\n') for line in f]
subject_list = np.array(list(map(int,lines))) # convert str 2 int and to an np array
nsub = len(subject_list) # number of total subjects
print("number of subjects: ", nsub)
missub,newsubs = TOPF.check_missing_data_phenotype(df_pheno, subject_list, phenostr)
if missub:
    print('Missing values! Please adjust the subject_list.txt file')
    quit()


############################################# start TOPF + score computation #######################################
print('current condition:', fmricondition)
print('current dataset:', dataset)
print('number of rois used:', nroi)

# set up machine learning model
J_model = TOPF.create_ML_julearn_model(clfname, setting_file)

# start prediction: 
df_test, df_train, df_bp = TOPF.main_TOPF(df_fmri, fmricondition, clfname, nroi, seed, subject_list, df_pheno, df_family_info, phenostr, J_model, Result_struct, feature_type, pcind, ntr, nstart, k_inner, k_outer, threshold)

# calculate scores (defined in metric_list) for test data
nfold = k_outer
foldwise = 1  # compute the score for each fold separately

# save scores to scoredir
scoredir = Result_struct.scoredir
savepath = eval(Result_struct.scorefname_test)


seedlist = [seed] # needs to be a list
probtype = J_model.probtype
metric_list = J_model.metric_list
mean_all, mean_seed, df_measure, _ = TOPF.main_compute_prediction_scores(fmricondition, clfname, probtype, Result_struct, metric_list, seedlist, nfold, savepath, foldwise, test=True)

# print
print('current condition:', fmricondition)
print('scores of each fold:')
print(df_measure)



