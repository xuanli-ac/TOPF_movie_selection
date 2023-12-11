import pandas as pd
import numpy as np
import sys
import os
import pliers
from pliers.filters import FrameSamplingFilter
from pliers.extractors import merge_results


# # to suppress error in finding gpus for tensorflow
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


def choose_pliers_extractor(movie_feature):
    '''
    Names of movie_features, strings.
    Currently, we use ['RMS','Brightness', 'Saliency', 'Vibrance','FaceRecognitionFaceLocations']  
    More could be added. 
    Possible options:https://pliers.readthedocs.io/en/latest/transformers.html#list-of-extractor-classes

    '''
    
    resample = 0 # set default as no resampling of stimulus is needed

    # loudness
    if 'RMS' in movie_feature:
        from pliers.extractors import RMSExtractor
        # Create an instance of this extractor
        ext = RMSExtractor()
    # speech/music rate (beats per minute)
    elif 'Tempo' in movie_feature:
        from pliers.extractors import TempoExtractor
        # Create an instance of this extractor
        ext = TempoExtractor(aggregate=None)
    elif 'Brightness' in movie_feature:
        from pliers.extractors import BrightnessExtractor
        ext = BrightnessExtractor()
    elif 'Saliency' in movie_feature:
        from pliers.extractors import SaliencyExtractor
        ext = SaliencyExtractor()
    elif 'Vibrance' in movie_feature:
        from pliers.extractors import VibranceExtractor
        ext = VibranceExtractor()
    elif 'FaceLocations' in movie_feature:
        from pliers.extractors import FaceRecognitionFaceLocationsExtractor
        ext = FaceRecognitionFaceLocationsExtractor()
        #ext = FaceRecognitionFaceLocationsExtractor(model='cnn')  # for more accurate locations
        resample = 1 # preproc is needed to transform from video to images to speed up computation
    else:
        print('please specificy the extractor to use')

    return ext, resample 


def extract_movie_features_to_dataframe(ext, runid, dpath, rpath, resample, version='Post'):
    if version == 'Post':
        runlist = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4','7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']
        subdir = 'Post_20140821_version'
    else:
        runlist = ['7T_MOVIE1_CC1.mp4', '7T_MOVIE2_HO1.mp4','7T_MOVIE3_CC2.mp4', '7T_MOVIE4_HO2.mp4']
        subdir = 'Pre_20140821_version'

    movie_stimulus = dpath + '/' + subdir + '/' + runlist[runid-1]

    # if video resampled to images specified
    if resample !=0:
        movie_stimulus = movie_sampling(movie_stimulus, hz=hz)

    # Extract features from the stimulus
    result = ext.transform(movie_stimulus)

    # save result of dataframe
    if type(result)=='list':
        print('automatically merge all results')
        r_df = merge_results(result, metadata=False)
    else:
        r_df = result.to_df()

    fname = rpath + '/' + movie_feature + '_'+ subdir + '_' + runlist[runid-1] + '.csv'
    r_df.to_csv(fname)
    
    return r_df, fname

def movie_sampling(movie, hz):
    # sample video to images
    filt = FrameSamplingFilter(hertz=hz)
    frames = filt.transform(movie)

    # Number of sampled frames/images
    nimage = frames.n_frames
    print('number of frames used: ', nimage)
    print('hertz: ', hz)
    return frames



# system input parameters
dpath = sys.argv[1]       # ~/data_useful/HCP/movie_stimulus
rpath = sys.argv[2]       # result dir 
runid = int(sys.argv[3])  # run id: 1-4
movie_feature = sys.argv[4] # 'brightness','rms','sailency','face'

# global variables
hz = 2 # sampling rate: number of frames/per second



###################################### do movie feature extraction and save results as dataframes in rpath
ext, resample = choose_pliers_extractor(movie_feature)
df_result, fname = extract_movie_features_to_dataframe(ext, runid, dpath, rpath, resample, version='Post')

print('done for movie run', runid)
print('movie feature:', movie_feature)
print('results saved to ', fname)