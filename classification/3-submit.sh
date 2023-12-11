#!/bin/bash
# v3.2

###### HTCondor/Juseless variables ###################
init_dir='/data/project/brainvar_topf_eval'
code_dir='code/2-classification'
name_py='3-eval_score_permutation.py'
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
sublist=${init_dir}/data/HCP/subject_list.txt
rpath=${init_dir}/results/classification/feature_before_norm/
settingpath=${init_dir}/${code_dir}/svm_rbf_gamma.txt
threshold=0
clfdir='svm_rbf_gamma_thre'${threshold}
clfname='svm' # e.g., 'svm', which classifier (see all availabel classifiers here: https://juaml.github.io/julearn/main/steps.html)


# other settings
dataset='hcp436' # 'hcp268' or 'hcp436'
nroi=436 # 268 or 436
		 # seed=0
phenostr='Gender'
kouter=10 # 10, number of folds for outer cv
kinner=5 # 5, number of folds for inner cv

# often adjusted setttings
cmpname='moviewise' # "moviewise" or "wholerun" or a custom path to the dataframe
nstart=0
cutntr=132 # cutntr<=0: using full length; otherwise: cut to the given length e.g., ntr=170
featuretype='singlePC' # 'singlePC' or combinedPC
pcind=1 # which PC or to the largest PC index when combinedPC



################################################# fixed ntr ######################################################################

############################################### set up log folder ############################################
logs_dir=${init_dir}/${code_dir}/logs_scoresig_perm_${dataset}_${phenostr}_${cmpname}_start${nstart}_${cutntr}_${featuretype}_${pcind}_${clfdir}
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"
############################################################################################################## 

for pseed in {1..1000}; do
	for movieid in {11..14}; do
		printf "arguments   = ${wkdir} ${sublist} ${rpath} ${dataset} ${nroi} ${pseed} ${movieid} ${cmpname} ${phenostr} ${cutntr} ${featuretype} ${pcind} ${clfname} ${kouter} ${kinner} ${settingpath} ${clfdir} ${threshold} ${nstart}\n"
		#printf "requirements = Machine == \"cpu22.htc.inm7.de\"\n"
		#printf "requirements = Machine == \"cpu11.htc.inm7.de\" || Machine == \"cpu12.htc.inm7.de\"\n"
		printf "log         = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_seed${pseed}_thre${threshold}.log\n"
		printf "output      = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_seed${pseed}_thre${threshold}.out\n"
		printf "error       = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_seed${pseed}_thre${threshold}.err\n"
		printf "Queue\n\n"    
	done
done

################################################# fixed ntr #######################################################################