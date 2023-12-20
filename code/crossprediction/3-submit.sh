#!/bin/bash
# v3.2

###### HTCondor/Juseless variables ###################
init_dir='/data/project/brainvar_topf_eval'
code_dir='code/3-crossprediction'
name_py='3-crosspred_cv.py'
name_wrapper='3-wrap.sh'
######################################################


# print the .submit header
printf "# The environment

universe              = vanilla
getenv                = True
request_cpus          = 1
request_memory        = 2G

# Execution
initialdir            = ${init_dir}/${code_dir}
executable            = ${init_dir}/${code_dir}/${name_wrapper}
transfer_executable   = False
transfer_input_files = ${name_py}
\n"


############################################### set up arguments for TOPF here ##############################################
# project dir

wkdir=${init_dir}
fdir=${init_dir}/results/classification/feature_before_norm/
dataset='hcp436' # 'hcp268' or 'hcp436'
nroi=436 # 268 or 436
phenostr='Gender'
cmpname='moviewise' # "moviewise" or "wholerun" or a custom path to the dataframe
cutntr=0 # cutntr<=0: using full length; otherwise: cut to the given length e.g., ntr=170
featuretype='singlePC' # 'singlePC' or combinedPC
pcind=1 # which PC or to the largest PC index when combinedPC
flip='num' # 'abs' or 'num', flip pc and scores, needed for cross-pred
settingpath=${init_dir}/code/2-classification/svm_rbf_gamma.txt
clfname='svm' # e.g., 'svm', which classifier (see all availabel classifiers here: https://juaml.github.io/julearn/main/steps.html)
clfdir='svm_rbf_gamma_thre0'



# other settings


####################################################################################################################################


############################################### set up log folder ############################################
logs_dir=${init_dir}/${code_dir}/logs_${cmpname}_${featuretype}_${pcind}_flip${flip}_cut${cutntr}
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"
##############################################################################################################

# Create a Job for each parameter combination
for mtrainid in 1; do
    printf "arguments   = ${wkdir} ${fdir} ${dataset} ${nroi} ${phenostr} ${cmpname} ${cutntr} ${featuretype} ${pcind} ${flip} ${settingpath} ${clfname} ${clfdir} ${mtrainid}\n"
    #printf "requirements = Machine == \"cpu23.htc.inm7.de\"\n"
    #printf "requirements = Machine == \"cpu11.htc.inm7.de\" || Machine == \"cpu12.htc.inm7.de\"\n"
    printf "log         = ${logs_dir}/\$(Cluster).\$(Process)_scores_flip${flip}_m${mtrainind}.log\n"
    printf "output      = ${logs_dir}/\$(Cluster).\$(Process)_scores_flip${flip}_m${mtrainind}.out\n"
    printf "error       = ${logs_dir}/\$(Cluster).\$(Process)_scores_flip${flip}_m${mtrainind}.err\n"
    printf "Queue\n\n"    
done

