#!/bin/bash
# v3.2

###### HTCondor/Juseless variables ###################
init_dir='/data/project/brainvar_topf_eval'
code_dir='code/5-movie_feature_extraction'
name_py='1-pliers_feature_extraction.py'
name_wrapper='1-wrap.sh'
######################################################


# print the .submit header
printf "# The environment

universe              = vanilla
getenv                = True
request_cpus          = 1
request_memory        = 10G

# Execution
initialdir            = ${init_dir}/${code_dir}
executable            = ${init_dir}/${code_dir}/${name_wrapper}
transfer_executable   = False
transfer_input_files = ${name_py}
\n"


############################################### set up arguments for TOPF here ##############################################
dpath=~/data_useful/HCP/movie_stimulus
rpath=${init_dir}/results_summary/moviefeatures
mflist=(Tempo)
#mflist=(RMS Brightness FaceLocations Saliency)

############################################### set up log folder ############################################
logs_dir=${init_dir}/${code_dir}/logs_movie_feature_extraction
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"
##############################################################################################################

# Create a Job for each parameter combination
for mfeature in "${mflist[@]}"; do
    for runid in {1..4}; do
	    printf "arguments   = ${dpath} ${rpath} ${runid} ${mfeature}\n"
	    #printf "requirements = Machine == \"cpu23.htc.inm7.de\"\n"
	    #printf "requirements = Machine == \"cpu11.htc.inm7.de\" || Machine == \"cpu12.htc.inm7.de\"\n"
	    printf "log         = ${logs_dir}/\$(Cluster).\$(Process)_feature${mfeature}_run${runid}.log\n"
	    printf "output      = ${logs_dir}/\$(Cluster).\$(Process)_feature${mfeature}_run${runid}.out\n"
	    printf "error       = ${logs_dir}/\$(Cluster).\$(Process)_feature${mfeature}_run${runid}.err\n"
	    printf "Queue\n\n"    
    done
done
