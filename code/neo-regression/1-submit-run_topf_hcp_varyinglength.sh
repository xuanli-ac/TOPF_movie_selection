#!/bin/bash
# v3.2

###### HTCondor/Juseless variables ###################
init_dir='/data/project/brainvar_topf_eval'
code_dir='code/neo-regression'
name_py='1-run_topf_hcp.py'
name_wrapper='1-wrap-run_topf_hcp.sh'
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
rpath=${init_dir}/results/neoregression/
settingpath=${init_dir}/${code_dir}/ridge_reg.txt
threshold=0
clfdir='ridge_thre'${threshold}
clfname='ridge' # e.g., 'svm', which classifier (see all availabel classifiers here: https://juaml.github.io/julearn/main/steps.html)


# other settings
dataset='hcp436' # 'hcp268' or 'hcp436'
nroi=436 # 268 or 436
seed=0
cmpname='wholerun' # "moviewise" or "wholerun" or a custom path to the dataframe

cutntr=132 # cutntr<=0: using full length; otherwise: cut to the given length e.g., ntr=170
featuretype='combinedPC' # 'singlePC' or combinedPC
pcind=2 # which PC or to the largest PC index when combinedPC
kouter=10 # 10, number of folds for outer cv
kinner=5 # 5, number of folds for inner cv
#phenolist=(PMAT24_A_CR)
nstart=0 # start from the nstart-th TR

# for test

phenostr=PMAT24_A_CR
#seedlist=(0 2)
#movieidlist=(1 2 3)
#threlist=(0.02 0.05)
####################################################################################################################################



### create a Job for each parameter combination

################################################# compute for a given dataframe with changing ntr #############################
# movieid='comb_120tr_top5m'  # movie ind or run ind, ranging from 1-14 or 1-4, or a customised id
# cmpname=${init_dir}/data/HCP/Schaefer436/df_X_m_5_13_12_6_9_tr_120_120_120_120_120.csv  

# for cutntr in {240..600..120}; do
# 	############################################### set up log folder ############################################
# 	logs_dir=${init_dir}/${code_dir}/logs_${dataset}_${phenostr}_${movieid}_${featuretype}_${pcind}_${clfdir}
# 	# create the logs dir if it doesn't exist
# 	[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"
# 	##############################################################################################################
# 	for seed in {1..9}; do	
# 		printf "arguments   = ${wkdir} ${sublist} ${rpath} ${dataset} ${nroi} ${seed} ${movieid} ${cmpname} ${phenostr} ${cutntr} ${featuretype} ${pcind} ${clfname} ${kouter} ${kinner} ${settingpath} ${clfdir}\n"
# 		#printf "requirements = Machine == \"cpu23.htc.inm7.de\"\n"
# 		#printf "requirements = Machine == \"cpu11.htc.inm7.de\" || Machine == \"cpu12.htc.inm7.de\"\n"
# 		printf "log         = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_ntr${cutntr}_seed${seed}.log\n"
# 		printf "output      = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_ntr${cutntr}_seed${seed}.out\n"
# 		printf "error       = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_ntr${cutntr}_seed${seed}.err\n"
# 		printf "Queue\n\n"    
# 	done
# done
################################################# compute for a given dataframe with changing ntr #############################



################################################# Changing ntr for moviewise or runwise #########################################

for cutntr in {120..600..120}; do
	############################################### set up log folder ############################################
	logs_dir=${init_dir}/${code_dir}/new_logs_start${nstart}_${cutntr}_${dataset}_${phenostr}_${cmpname}_${featuretype}_${pcind}_${clfdir}
	# create the logs dir if it doesn't exist
	[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"
	##############################################################################################################
	for seed in {0..9}; do
		for movieid in {1..4}; do
			printf "arguments   = ${wkdir} ${sublist} ${rpath} ${dataset} ${nroi} ${seed} ${movieid} ${cmpname} ${phenostr} ${cutntr} ${featuretype} ${pcind} ${clfname} ${kouter} ${kinner} ${settingpath} ${clfdir} ${threshold} ${nstart}\n"
			#printf "requirements = Machine == \"cpu23.htc.inm7.de\"\n"
			#printf "requirements = Machine == \"cpu11.htc.inm7.de\" || Machine == \"cpu12.htc.inm7.de\"\n"
			printf "log         = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_seed${seed}.log\n"
			printf "output      = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_seed${seed}.out\n"
			printf "error       = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_seed${seed}.err\n"
			printf "Queue\n\n"    
		done
	done
done
################################################# Changing ntr for moviewise or runwise #########################################




################################################# fixed ntr ######################################################################

# ############################################### set up log folder ############################################
# logs_dir=${init_dir}/${code_dir}/logs_nonorm_${dataset}_${phenostr}_${cmpname}_start${nstart}_${cutntr}_${featuretype}_${pcind}_${clfdir}
# # create the logs dir if it doesn't exist
# [ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"
# ############################################################################################################## 

# for seed in {1..9}; do
# 	for movieid in {1..14}; do
# 		printf "arguments   = ${wkdir} ${sublist} ${rpath} ${dataset} ${nroi} ${seed} ${movieid} ${cmpname} ${phenostr} ${cutntr} ${featuretype} ${pcind} ${clfname} ${kouter} ${kinner} ${settingpath} ${clfdir} ${threshold} ${nstart}\n"
# 		#printf "requirements = Machine == \"cpu22.htc.inm7.de\"\n"
# 		#printf "requirements = Machine == \"cpu11.htc.inm7.de\" || Machine == \"cpu12.htc.inm7.de\"\n"
# 		printf "log         = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_seed${seed}_thre${threshold}.log\n"
# 		printf "output      = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_seed${seed}_thre${threshold}.out\n"
# 		printf "error       = ${logs_dir}/\$(Cluster).\$(Process)_movie${movieid}_seed${seed}_thre${threshold}.err\n"
# 		printf "Queue\n\n"    
# 	done
# done

################################################# fixed ntr #######################################################################