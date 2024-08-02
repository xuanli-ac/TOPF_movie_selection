#!/bin/bash
# v3.2

###### HTCondor/Juseless variables ###################
init_dir='/data/project/brainvar_topf_eval'
code_dir='code/neo-regression'
name_py='4-cal_feature_importance.py'
name_wrapper='4-wrap.sh'
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
rpath=${init_dir}/results/neoregression/ # where to find models and features

# other settings
dataset='hcp436' # 'hcp268' or 'hcp436'
nroi=436 # 268 or 436
movieid=1  # or run ind, ranging from 1-14 or 1-4
cmpname='moviewise'  # "moviewise" or "wholerun"
phenostr='PMAT24_A_CR'
cutntr=132 # cutntr<=0: using full length; otherwise: cut to the given length e.g., ntr=170
featuretype='singlePC' # 'singlePC' or combinedPC
pcind=1 # which PC or to the largest PC index when combinedPC
clfname='ridge' # e.g., 'svm', which classifier (see all availabel classifiers here: https://juaml.github.io/julearn/main/steps.html)
kouter=10 # 10, number of folds for outer cv
n_perm=1000 # 5, number of permutations
scoring='neg_mean_absolute_error'

threshold=0
clfdir='ridge_thre'${threshold}


#seedlist=(0 2)
#movieidlist=(1 2 3)
####################################################################################################################################


############################################### set up log folder ############################################
logs_dir=${init_dir}/${code_dir}/logs_fwperm1000_${clfdir}_${dataset}_${phenostr}_${cmpname}_${cutntr}_${featuretype}_${pcind}
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"
##############################################################################################################

# Create a Job for each parameter combination

for movieid in {1..14}; do
	printf "arguments   = ${wkdir} ${rpath} ${dataset} ${nroi}  ${movieid} ${cmpname} ${phenostr} ${cutntr} ${featuretype} ${pcind} ${clfname} ${kouter} ${n_perm} ${scoring} ${clfdir}\n"
	#printf "requirements = Machine == \"cpu23.htc.inm7.de\"\n"
	#printf "requirements = Machine == \"cpu11.htc.inm7.de\" || Machine == \"cpu12.htc.inm7.de\"\n"
	printf "log         = ${logs_dir}/\$(Cluster).\$(Process)_permutation_movie${movieid}_nroi${nroi}.log\n"
	printf "output      = ${logs_dir}/\$(Cluster).\$(Process)_permutation_movie${movieid}_nroi${nroi}.out\n"
	printf "error       = ${logs_dir}/\$(Cluster).\$(Process)_permutation_movie${movieid}_nroi${nroi}.err\n"
	printf "Queue\n\n"    
done
