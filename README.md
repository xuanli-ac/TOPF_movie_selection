# Stimulus Selection for Predictions of Phenotypes
Stimulus selection influences prediction of individual phenotypes in naturalistic conditions
https://www.biorxiv.org/content/10.1101/2023.12.07.570273v1

## virtual environment for prediction/classification analyses
```
conda create -n topf_eval python=3.9
conda activate topf_eval
conda install ipython ipykernel autopep8
conda install matplotlib 
conda install -c conda-forge nilearn eli5 seaborn statsmodels pingouin
pip3 install -U julearn
conda install scikit-learn pandas numpy h5py
pip install pliers
pip install cmake (required when installing pliers)
pip install -r pliers-optional-dependencies.txt
pip install --upgrade protobuf==3.20.3 (downgrade to this version to be able to install FaceRecognitionExtractor)
conda install -c conda-forge wordcloud
```

## another virtual environment for movie feature extraction with whisper
```
conda create -n whisper_env python=3.9
pip install git+https://github.com/openai/whisper.git
conda install ipykernel
#brew install ffmpeg
```

## Description:

### prepare data
1. FMRI data: download preprocessed fMRI data and save them into '/data/HCP/Schaefer436' folder within the project folder. Data are available at: https://figshare.com/s/1f700c1c2aa0493cd9e2

2. Family structure data: Download family structure (restricted) data of all 7T subjects from HCP (https://db.humanconnectome.org/) after approval and save it as "RESTRICTED_7T.csv" in '/data/HCP/'

3. List of subjects ('subject_list.txt') and phenotypic data ('Subjects_Info.csv') are under '/data/HCP/'

### code structure
1. classification - code for performing sex classification within each movie clip, evaluating performance, significance and feature importance, as well as detail settings of each classification algorithm ('*.txt').

2. crossprediction - code for performing classification across movie clips

3. movie_feature_extraction - code for extract movie features using the pliers toolbox (McNamara et al., 2017). Movie stimuli can be downloaded at https://db.humanconnectome.org/. 

### results
Examples of results can be found under '/results_summary'.

### scripts
All .sh files are needed only when running code on server with HTcondor; otherwise only the python scripts are needed. Depending on systems and settings, the computation might take 1 - 2 hours for each computation job.

For example:

In Mac/PC terminal:
```
conda activate topf_eval
python3.9 1-run_topf_hcp.py /myproject /myproject/data/HCP/subject_list.txt /myresultdir/ hcp436 436 0 2 moviewise Gender 132 singlePC 1 rf 10 5 myproject/code/classification/rf.txt rf_thre0 0 0
```
On cluster + HTCondor:

```
./1-submit-run_hcp_topf.sh | condor_submit

```