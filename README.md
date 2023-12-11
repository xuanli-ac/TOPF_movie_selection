# Stimulus Selection for Predictions of Phenotypes
Stimulus selection influences prediction of individual phenotypes in naturalistic conditions

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
