import os
import ast
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve, roc_auc_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold,KFold,train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.io as sio
from julearn import run_cross_validation
from julearn.utils import configure_logging
import pickle
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from joblib import dump, load
from sklearn.metrics import explained_variance_score



###################################### Useful sub-functions ##########################################

##################################### some check-ups #############################
def check_missing_data_phenotype(df_pheno, subjects_id, phenostr):
    df_temp = df_pheno.loc[df_pheno['Subject'].isin(subjects_id)]
    S_temp = df_temp[['Subject',phenostr]]
    if S_temp[phenostr].isnull().values.any():
        print('please check you subjects, containing missing values for', phenostr)
        ind = S_temp[phenostr].isnull().values
        missing_subs = subjects_id[ind]
        new_subs = subjects_id[~ind]
        print("Values for these subjects are missing:", missing_subs)
    else:
        print('data are complete for Machine Learning part')
        missing_subs = []
        new_subs = []
    return missing_subs, new_subs
#####################################################################################

##################################### feature extraction and preprocessing ###############################

# could later move the discard trs here
def form_data_across_subjects(df_m, subjects_id, roi, ntr=None, nstart=0):
    """
    Aggregate z-score normalized fMRI time series across subjects within a given ROI.
    The returned Rsig is the input of PCA (feature extraction)
    #df_m: dataframe of fmri data from a given movie clip/run/session
    #ntr: desired number of trs to preserve
    #roi: an integer, roi index from 1!!
    #subjects_id: subjects to be analyzed
    """
    
    # get movie length from the first subject
    movie_len = df_m[df_m.subject==int(subjects_id[0])].shape[0]

    # init the data matrix for the roi
    X = np.empty((movie_len,0))

    # concatenate subjects within the given roi
    for sub in subjects_id:
        # check if data dimension is correct for each subject
        if df_m[df_m.subject == int(sub)].shape[0]:
            df_temp = df_m[df_m.subject==int(sub)]
            x_temp = df_temp[str(roi)].values
            x_temp = np.reshape(x_temp,(movie_len,1))
            X = np.append(X, x_temp, axis = 1)
        else:
            print('fmri data of subject ', sub, ' mismatch with others!')
            break

    # truncate the data to the desired length if needed
    # from the first to the ntr-th volume  (before 8 Feb. 2023)
    # X = X[0:ntr,:]

    # from the nstart to the nstart+ntr-th volume (added 8 Feb 2023)
    X = X[nstart:ntr+nstart,:]  # if ntr+nstart> ntotal, use all available automatically
        
    # z-score normalise each time series, ddof=1 consistent with matlab
    X = scipy.stats.zscore(X,ddof=1)  # nTR * nSub
    return X


def normalise_feature(F,Ft):
    """
    normalise feature for ML for training and test separately to avoid data leakage
    F: training; ntrainsub*nfeatures
    Ft: test; ntestsub*nfeatures
    """
    scaler = StandardScaler().fit(F)
    F = scaler.transform(F)
    Ft = scaler.transform(Ft)
    return F, Ft   


def flip_pc_and_clean_loadings(pc_score,pc_loading,flip=None,clean=None):
    """
    flip pc score and loadings when needed, eg. for cross-condition comparison
    """
    if flip =='abs':
        # make sure the maximum absolute value comes from postive
        if np.abs(pc_loading).max()>np.max(pc_loading):
            pc_loading = pc_loading*-1
            pc_score = pc_score*-1
    elif flip == 'num':
        # make sure number of pos larger than number of neg
        if np.sum(pc_loading >= 0) < np.sum(pc_loading < 0):
            pc_loading = pc_loading*-1
            pc_score = pc_score*-1
    else:
        # do nothing
        pc_score = pc_score
        pc_loading = pc_loading
    
    # set all negative to zero
    if clean:
        pc_loading[np.argwhere(pc_loading<0)] = 0 
    
    return pc_score,pc_loading


def feature_selection(train_data, test_data, var_data, threshold):
    X_keys = [col for col in train_data.columns if 'PC' in col]
    Var = var_data[X_keys].values
    Var = np.reshape(Var,[-1,1])
    f_preserved = np.argwhere(Var>=threshold)[:,0]
    
    X_new = np.array(X_keys)
    X_new = X_new[f_preserved]
    X_new = list(X_new)

    F = train_data[X_keys].values
    F_new = F[:,f_preserved]

    Ft = test_data[X_keys].values
    Ft_new = Ft[:,f_preserved]

    return F_new, Ft_new, X_new


def perform_feature_extraction(df_m, sub_train, sub_test, nroi, pcind=1, ftype='singlePC', ntr=None, nstart=0, flip=None,clean=None):
    """
    # Extract and normalise features for training and testing subjects of each outer CV fold.
    ### Features refer to the individual-specific topographies.
    ### For training subjects, each feature is the PC loading of a ROI.
    ### For testing subjects, each feature is the correlation between PC learned on training and fMRI time series of a ROI.
    ### sub_train/sub_test: subjects' ids for training/test sets; e.g. 100610
    ### nnode: number of rois
    ### ntr: desired number of TRs to preserve; default preserve all TRs
    ### pcind: index of PC whose loadings will be used as features, index from 1 
    ### F, Ft, Var: Features for training, test and variance explained by PC
    ### flip = 'abs','num' if flip pc loadings and scores
    ### clean = 1, if set all neg loadings to pos
    """

    ntrain = sub_train.shape[0]
    ntest = sub_test.shape[0]
    
    # get movie_len and check ntr
    movie_len = df_m[df_m.subject==int(sub_train[0])].shape[0] # full length

    if ntr is not None:
        if ntr>movie_len:
            print('desired TRs larger than possible, using the full length!')
            ntr = movie_len
    else:
        ntr = movie_len #tranform none to a integer
    
    if nstart > movie_len:
        print('start from ', nstart, 'not possible')
        

    print('Total number of TRs:', movie_len)
    print('Number of TRs used: ', ntr) 
    print('start from ', nstart)   
    
    F = np.empty([ntrain,0],float) # training features
    Ft = np.empty([ntest,0],float) # testing features
    Var = np.empty([1,0],float) # variance explained by PC
    PCs = np.empty([ntr,0],float) # PC scores; shared responses
    X_keys = []

    pca = PCA(n_components=pcind+1) # nc: keep the first pcind+1 components (PC)

    if ftype == 'singlePC':
        start_pc = pcind-1
        npc = 1
    elif ftype == 'combinedPC':
        start_pc = 0
        npc = pcind
    else:
        print('please specify feature type: "singlePC" or "combinedPC"!')

    # Start PCA: roi index from 1 !!
    for roi in range(1,nroi+1,1):

        # Training: perform PCA for each ROI and extract PC loadings as features

        X = form_data_across_subjects(df_m, sub_train, roi, ntr, nstart)
        SCORE = pca.fit_transform(X) #column
        COEFF = pca.components_
        EXPLAINED = pca.explained_variance_ratio_
        LATENT = pca.explained_variance_
       
        for pc in range(start_pc,pcind,1):

            # set variable names for features
            X_keys = X_keys+ [f'{roi}_PC{pc+1}']

            # get pc loadings and scores 
            temp = COEFF[pc,:]*np.sqrt(LATENT[pc]) # PC_pcind loading as row
            temp = np.reshape(temp, (-1, 1)) # change to column
            stemp = SCORE[:,pc]
            stemp = np.reshape(stemp,(-1,1))
    
            # flip and clean if needed
            stemp,temp = flip_pc_and_clean_loadings(stemp,temp,flip,clean)

            # output
            F = np.append(F,temp,axis=1)   # pc_score: ntrain*npc*nroi
            Var = np.append(Var,EXPLAINED[pc]) # variance
            PCs = np.append(PCs, stemp, axis=1) #loading: movie_len or ntr*npc*nroi
        
        # Testing if sub_test is not empty:
            if ntest>0:
                Xt = form_data_across_subjects(df_m, sub_test, roi, ntr, nstart)
                ft_temp = np.empty([1,0],float) # test loadings
                for testind in range(0,ntest,1):
                    temp = scipy.stats.pearsonr(Xt[:,testind],stemp) # PC_pcind score
                    tr = temp[0] # correlation = loading
                    # set to zero if negative if clean != 0
                    if clean:
                        if tr<0:
                            tr = tr*0
                    ft_temp = np.append(ft_temp,tr)

                # output
                ft_temp = np.reshape(ft_temp,(-1,1))  # a column
                Ft = np.append(Ft,ft_temp, axis=1) #ntest * npc* nroi
            else:
                Ft = []
        # show progress
        if roi % 50 == 0:
            print('Node',roi,' PCA finished')

    # Outputs

    Var = np.reshape(Var,[1,-1]) # nroi * npc *1; a row
    
    return F,Ft,Var,PCs,X_keys


######################## won't be used after 24.01.2023 ########################################
def perform_feature_extraction_singlePC(df_m, sub_train, sub_test, nroi, pcind=1, ntr=None, flip=None,clean=None, norm=1):
    """
    # Extract and normalise features for training and testing subjects of each outer CV fold.
    ### Features refer to the individual-specific topographies.
    ### For training subjects, each feature is the PC loading of a ROI.
    ### For testing subjects, each feature is the correlation between PC learned on training and fMRI time series of a ROI.
    ### sub_train/sub_test: subjects' ids for training/test sets; e.g. 100610
    ### nnode: number of rois
    ### ntr: desired number of TRs to preserve; default preserve all TRs
    ### pcind: index of PC whose loadings will be used as features, index from 1 
    ### F, Ft, Var: Features for training, test and variance explained by PC
    ### flip = 'abs','num' if flip pc loadings and scores
    ### clean = 1, if set all neg loadings to pos
    ### norm = 1, z-score normalise each feature
    """
    
    nc = pcind+1 # nc: keep the first nc components (PC)
    ntrain = sub_train.shape[0]
    ntest = sub_test.shape[0]
    
    # get movie_len and check ntr
    movie_len = df_m[df_m.subject==int(sub_train[0])].shape[0] # full length

    if ntr is not None:
        if ntr>movie_len:
            print('desired TRs larger than possible, using the full length!')
            ntr = movie_len
    else:
        ntr = movie_len #tranform none to a integer
    
    print('Total number of TRs:', movie_len)
    print('Number of TRs used: ', ntr)    
    
    F = np.empty([ntrain,0],float) # training features
    Ft = np.empty([1,0],float) # testing features
    Var = np.empty([1,0],float) # variance explained by PC
    PCs = np.empty([ntr,0],float) # PC scores; shared responses
    
    pca = PCA(n_components=nc)
    
    # TRAINING: perform PCA for each ROI and extract PC loadings as features
    # roi index from 1 !!
    for roi in range(1,nroi+1,1):
        X = form_data_across_subjects(df_m, sub_train, roi, ntr)
        SCORE = pca.fit_transform(X) #column
        COEFF = pca.components_
        EXPLAINED = pca.explained_variance_ratio_
        LATENT = pca.explained_variance_
        
        # get pc loadings and scores
        temp = COEFF[pcind-1,:]*np.sqrt(LATENT[pcind-1]) # PC_pcind loading as row: pcind=1 refers to PC1
        temp = np.reshape(temp, (-1, 1)) # change to column
        stemp = SCORE[:,pcind-1]
        stemp = np.reshape(stemp,(-1,1))
    
        # flip and clean if needed
        stemp,temp = flip_pc_and_clean_loadings(stemp,temp,flip,clean)

        # save 
        F = np.append(F,temp,axis=1)   # pc_score: ntrain*nroi
        Var = np.append(Var,EXPLAINED[pcind-1]) # variance
        PCs = np.append(PCs, stemp, axis=1) #loading: movie_len or ntr*nroi
        
    # Testing:
        Xt = form_data_across_subjects(df_m, sub_test, roi, ntr)
        for testind in range(0,ntest,1):
            temp = scipy.stats.pearsonr(Xt[:,testind],SCORE[:,pcind-1]) # PC_pcind score
            tr = temp[0] # correlation = loading
            # set to zero if negative if clean != 0
            if clean:
                if tr<0:
                    tr = tr*-1
            Ft = np.append(Ft,tr)
        if roi % 50 == 0:
            print('Node',roi,' PCA finished')

    Ft = np.reshape(Ft, (-1, ntest)) 
    Ft = np.transpose(Ft) # (ntest,nroi)
    Var = np.reshape(Var,[-1,1])

    # normalizing each feature for training and test
    if norm:
        F,Ft = normalise_feature(F,Ft)

    # set variable names for features
    X_keys = [f'%d_PC{pcind}' % i for i in range(1, nroi+1, 1)]
    
    return F,Ft,Var,PCs,X_keys


def perform_feature_extraction_multiplePC(df_m, sub_train, sub_test, nroi, pcind=1, ntr=None, flip=None, clean=None, norm=1):
    """
    Extract features for training and testing subjects of each outer CV fold.
    Features refer to the individual-specific topographies.
    For training subjects, each feature is the PC loading of a ROI.
    For testing subjects, each feature is the correlation between PC learned on training and fMRI time series of a ROI.
    # sub_train/sub_test: subjects' ids for training/test sets; e.g. 100610
    # nnode: number of rois
    # ntr: desired number of TRs to preserve; default preserve all TRs
    # npc: number of PCs whose loadings will be used as features
    # F, Ft, Var: Features for training, test and variance explained by PC
    """

    ntrain = sub_train.shape[0]
    ntest = sub_test.shape[0]

    F = np.empty([ntrain,0],float) # training features
    Ft = np.empty([ntest,0],float) # testing features
    Var = np.empty([nroi,0],float) # variance explained by PC
    PCs = np.empty([0,nroi],float) # PC scores; shared responses
    X_keys = []

    for pc in range(1,pcind+1,1):
        F_temp,Ft_temp,Var_temp,PCs_temp,Xk_temp = perform_feature_extraction_singlePC(df_m, sub_train, sub_test, nroi, pc, ntr, flip, clean, norm)
        F = np.append(F,F_temp,axis=1)
        Ft = np.append(Ft,Ft_temp,axis=1)
        Var = np.append(Var,Var_temp,axis=1)
        if pc == 1:
            PCs = np.append(PCs,PCs_temp,axis=0)
        else:
            PCs = np.append(PCs,PCs_temp,axis=1)
        X_keys = X_keys+Xk_temp

    return F,Ft,Var,PCs,X_keys

######################## above won't be used after 24.01.2023 ########################################

#####################################################################################################

def get_dataframe(F,sub_list,df_pheno,phenostr,X_keys): 
    """
    ## get the targets of train/test subjects and create dataframe (subid+target+features) for ML (julearn)
    Integrate features and target (phenotype) as a single dataframe ready for prediction
    Subjects with missing values should be excluded beforehand by function "check_missing_data_phenotype"
    """  
    # df_pheno: phenotype information provided by HCP ('unrestricted')
    # sub_train/sub_test: sub id
    # phenostr: the HCP label for the phenotype to be predicted
    
    # get target scores for the given subjects and phenotype
    df_temp = df_pheno.loc[df_pheno['Subject'].isin(sub_list)] # data for training subjects
    df_score = df_temp[['Subject', phenostr]].sort_values(by='Subject',kind='stable').reset_index(drop=True)   
    
    # convert features to dataframe
    df_F = pd.DataFrame(data=F,columns=X_keys)
  
    # Insert 'Subject' column
    df_F.insert(0, 'Subject', sub_list)
   
    # combine feature and target according to subject id in target (sorted already)
    df_data = pd.merge(df_score, df_F, on="Subject", how="left")

    return df_data

def perm_traindata(F,sub_list,sub_full,df_pheno,phenostr,X_keys,permseed): 
    """
    ## permute train data for permutation test"
    """  
    # df_pheno: phenotype information provided by HCP ('unrestricted')
    # sub_train/sub_test: sub id
    # sub_full: full subject list (train+test)
    # phenostr: the HCP label for the phenotype to be predicted
    
    # get target scores for the all subjects 
    df_temp = df_pheno.loc[df_pheno['Subject'].isin(sub_full)] # data for all subjects
    df_score = df_temp[['Subject', phenostr]].sort_values(by='Subject',kind='stable').reset_index(drop=True)   
    target = df_score[[phenostr]].values

    # permutate over all subjects
    ns = len(sub_full)
    indp = np.random.RandomState(seed=permseed).permutation(ns)
    target = target[indp]
    df_score[phenostr] = target

    # only train data
    df_score = df_score.loc[df_score['Subject'].isin(sub_list)] # data for train subjects

    # convert features to dataframe
    df_F = pd.DataFrame(data=F,columns=X_keys)
  
    # Insert 'Subject' column
    df_F.insert(0, 'Subject', sub_list)
   
    # combine feature and target according to subject id in target (sorted already)
    df_data = pd.merge(df_score, df_F, on="Subject", how="left")

    return df_data
####################################################################################################################



############################################ machine learning set up ###############################################
def cv_control_for_family(subject_list,df_pheno, phenostr, seed, k=10, df_family_info=None):
    """
    split train/test for cv folds and control for family structure if needed (for HCP!)
    subject_list: full subject list
    df_pheno: phenotype.csv - HCP-like file
    phenostr: phenotype label of the given dataset
    df_family_info: gives the HCP family info
    """
    # target of all subjects
    df_target = df_pheno[['Subject', phenostr]]
    df_target = df_target[df_target.Subject.isin(subject_list)].sort_values(by='Subject',kind='stable').reset_index(drop=True)
    target = df_target[phenostr].values

    # family info; control for family structure
    if df_family_info is not None:
        df_fam = df_family_info[['Subject', 'Family_ID']]
        df_fam = df_fam[df_fam.Subject.isin(subject_list)].sort_values(by='Subject',kind='stable').reset_index(drop=True)
        groups = df_fam['Family_ID'].values
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        cv_fit = cv.split(subject_list, target, groups)

    # if no family info available   
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        cv_fit = cv.split(subject_list, target)
        groups = None
    
    return cv_fit, groups

####### set up set up model paramters from setting files
class create_ML_julearn_model:
    def __init__(self, clfname, setting_file) -> None:
        
        # currently available models of Julearn: https://juaml.github.io/julearn/main/steps.html
        probtype_available = ['regression','binary_classification','multiclass_classification']
        clfnamelist = ['svm', 'rf', 'et', 'adaboost','baggging','gradientboost','gauss','logit','logitcv','linreg','ridge','ridgecv','sgd']

        # read in model settings from a setting.txt
        if os.path.exists(setting_file):
            with open(setting_file) as f:
                ms = f.read()
            model_params = ast.literal_eval(ms) # a dict

            # probtype
            key = 'problem_type'
            if key in model_params.keys():
                probtype = model_params[key]
                model_params.pop(key, None) # delete the key
                if any(probtype in s for s in probtype_available):
                    self.probtype = probtype
                else:
                    print('Specified problem type not found! Set to binary_classification automatically')
                    self.probtype = 'binary_classification'
            else:
                print('problem_type not defined! Set to binary_classification automatically')
                self.probtype = 'binary_classification'

            # metric_list
            key = 'metric_list'
            if key in model_params.keys():
                metric_list = model_params[key]
                model_params.pop(key, None) # delete the key
                self.metric_list = metric_list
            else:
                self.metric_list = []
                print('metric_list not defined! will use the default metrics according to problem_type later')
            
            # clfname
            if any(clfname in s for s in clfnamelist):    
                self.name = clfname
            else:
                print('Specified model not found! Set to svm automatically')
                self.name = 'svm'
                
            # the rest keys will be used in model_params
            if len(model_params.keys()) == len(model_params.values()):
                self.model_params = model_params
            else:
                print('number of model hyperparameters and number of given values mismatch!')

        else:
            print('please specify the path to model settings')
            

######################################################################################################################


########################################  set up results formation ###################################################

####### results folder structure set up
class init_result_dir:
    def __init__(self, rdir, clfdir) -> None:
        # rdir: the resultdir specified beforehand, e.g.,

        ######################################## feature dir (independent of classifiers)
        featuredir = rdir+'/features/'
        if not os.path.exists(f"{featuredir}"):
            os.makedirs(f"{featuredir}")
        self.featuredir = featuredir
        
        #save feature dataframe of each fold separately

        # older version - before 24.01.2023 (normalised features)
        # self.featurefname_test = "featuredir + f'features_test_{fmricondition}_seed{seed}_fold{foldind+1}.csv'"
        # self.featurefname_train = "featuredir + f'features_train_{fmricondition}_seed{seed}_fold{foldind+1}.csv'"

        # on and after 25.01.2023 (pc loadings before normalisation)
        self.featurefname_test = "featuredir + f'loadings_test_{fmricondition}_seed{seed}_fold{foldind+1}.csv'"
        self.featurefname_train = "featuredir + f'loadings_train_{fmricondition}_seed{seed}_fold{foldind+1}.csv'"
        self.varfname = "featuredir + f'var_{fmricondition}_seed{seed}_fold{foldind+1}.csv'"

        ######################################## other results (dependent of classifiers)

        ######################################## prediction dir 
        preddir = rdir+'/'+clfdir+'/prediction/'
        if not os.path.exists(f"{preddir}"):
            os.makedirs(f"{preddir}")
        self.preddir = preddir

        # file names for predictions on test data/train data over all folds
        self.predfname_test = "preddir + f'pred_test_{fmricondition}_seed{seed}.csv'"
        self.predfname_train = "preddir + f'pred_train_{fmricondition}_seed{seed}.csv'"
        
        
        ######################################## model dir 
        modeldir = rdir+'/'+clfdir+'/fitted_models/'
        if not os.path.exists(f"{modeldir}"):
            os.makedirs(f"{modeldir}")
        self.modeldir = modeldir

        # save model of each fold separately
        self.modelfname = "modeldir + f'model_{fmricondition}_seed{seed}_fold{foldind+1}.sav'"
        self.modelfname_bp = "modeldir + f'bestparams_{fmricondition}_seed{seed}.csv'"
        

        ######################################## score dir 
        scoredir = rdir+'/'+clfdir+'/scores/'
        if not os.path.exists(f"{scoredir}"):
            os.makedirs(f"{scoredir}")
        self.scoredir = scoredir
        self.scorefname_test = "scoredir + f'/scores_test_{fmricondition}_seed{seed}.csv'"
        self.scorefname_train = "scoredir + f'/scores_train_{fmricondition}_seed{seed}.csv'" 


####### concatenate_best_params 
def concatenate_best_params(best_param, bp, foldind):
    # best_param: current dict to be updated
    # bp: from the given (foldind) fold 
    # foldind: 0 
    # merge two dictionaries
    def mergeDictionary(dict_1, dict_2):
        dict_3 = {**dict_1, **dict_2}
        for key, value in dict_3.items():
            if key in dict_1 and key in dict_2:
                dict_3[key] = [value , dict_1[key]]
        return dict_3
    
    if foldind == 0:
        best_param = bp
    elif foldind ==1:
        best_param = mergeDictionary(best_param, bp) #values -->list
    else:
        for key, value in bp.items():
            best_param[key]+=[value]
    return best_param
######################################################################################################################



############################################# calculate prediction scores #############################################
def get_predproba_for_roc_auc(estimator, J_model, test_data, X_keys):
    """
    get predict_proba on test for calculating roc_auc for binary classification if available
    """ 
    # estimator/model: fitted model (Julearn output)
    # J_model: class defined by create_ML_julearn_model, containing model parameters
    # test_data: feature dataframe
    # X_keys: feature labels
    # ytest_proba: output of predict_proba or decision_function of given samples
    
    #### get the sklearn model from julearn model
    clfname = J_model.name # e.g., 'rf'
    model_sk = estimator[clfname]

    if J_model.probtype == 'binary_classification':
        # svm or similar clfs, which set clf__probability = True 
        key_proba = [key for key, value in J_model.model_params.items() if 'probability' in key]
        if key_proba and J_model.model_params[key_proba[0]]:
            ytest_proba = estimator.predict_proba(test_data[X_keys])[:,1] #binary case: get probability of the class with "greater label"
        
        # ridge and similar clfs
        elif hasattr(model_sk, 'decision_function') and callable(model_sk.decision_function):
            ytest_proba = estimator.decision_function(test_data[X_keys])

        # random forest/or other similar clfs
        elif J_model.name!='svm' and hasattr(model_sk, 'predict_proba') and callable(model_sk.predict_proba):
            ytest_proba = estimator.predict_proba(test_data[X_keys])[:,1]

        else:
            ytest_proba = []
    else:
        ytest_proba = []
    return ytest_proba



def cal_measures(df_temp, probtype, metric_list):
    """
    calculate measures of a given prediction with true values using metrics in metric_list or use default if empty
    """
    # final metric_list and measures: just in case some predefined metrics can't be calculated
    mlist = []
    measures = []
    acc = None
    bacc = None
    auc = None
    pr = None
    sr = None
    r2 = None
    mae = None

    if df_temp.empty:
        print('Input dataframe is empty!')
            
    if probtype == 'binary_classification':
        # if metric_list is empty, set to default 
        if not metric_list:
            metric_list = ['accuracy','balanced_accuracy','roc_auc']
        for met in metric_list:
            if met == 'accuracy':
                acc = accuracy_score(df_temp.true, df_temp.pred)
            elif met =='balanced_accuracy':
                bacc = balanced_accuracy_score(df_temp.true, df_temp.pred)
            elif met == 'roc_auc':
                auc = roc_auc_score(df_temp.true, df_temp.proba)
            else:
                print('this measure needs to be manually defined in func "cal_measures" !')
        if acc is not None:
            measures = measures + [acc]
            mlist = mlist + ['accuracy']
        if bacc is not None:
            measures = measures + [bacc]
            mlist = mlist + ['balanced_accuracy']
        if auc is not None:
            measures = measures + [auc]
            mlist = mlist + ['roc_auc']

    #### for regression not tested yet ###########
    elif probtype == 'regression':
        # if metric_list is empty, set to default 
        if not metric_list:
            metric_list = ['pearson','spearman','r2','mean_absolute_error']
        for met in metric_list:
            if "pearson" in met:
                pr = scipy.stats.pearsonr(df_temp.true, df_temp.pred)
            if "spearman" in met:
                sr = scipy.stats.spearmanr(df_temp.true, df_temp.pred)
            if "r2" in met:
                r2 = r2_score(df_temp.true, df_temp.pred)
            if met == "mean_absolute_error":
                mae = mean_absolute_error(df_temp.true, df_temp.pred)
                
        if pr is not None:
            measures = measures + [pr[0]]
            mlist = mlist + ['pearson']
        if sr is not None:
            measures = measures + [sr[0]]
            mlist = mlist + ['spearman']
        if r2 is not None:
            measures = measures + [r2]
            mlist = mlist + ['r2']
        if mae is not None:
            measures = measures + [mae]
            mlist = mlist + ['mean_absolute_error']    
    
    return measures, mlist

######################################################################################################################



######################################################## MAIN FUNCTIONS ##############################################

def main_TOPF(df_m, fmricondition, clfname, nroi, seed, subject_list, df_pheno, df_family_info, phenostr, J_model, Result_struct, feature_type='singlePC', pcind=1, ntr=None, nstart=0, k_inner=5, k_outer=10, threshold=0, flip=None, clean=None, norm=1, permseed=0):
    
    """
    J_model: self-defined ML Julearn model class. Parameters set outside beforehand. Could be replaced by sklearn models when needed
    feature_type: "singlePC" or "combinedPC". 
    pcind: when "singlePC", use the pcind-th PC loadings as features; when combinedPC, use the 1st to pcind-th PC loadings together as features
    ntr: use data length of ntr TRs (default = None, using the full length without truncation)
    nstart: use data start from the nstart-th TR volume
    """
    ######################################## create empty list to save results
    
    # for saving predictions
    pred_train = []  
    true_train = []
    pred_test = []
    true_test = []
    proba_test = [] # predict_proba on test
    
    fold_label_train = []
    fold_label_test = []
    seed_label_train = []
    seed_label_test = []
    subid_train = []
    subid_test = []
    best_param = {}

    # initialize result directory structures + filename patterns
    # Result_struct = init_result_dir(rdir,clfdir) # done outside in the run.py
    featuredir = Result_struct.featuredir
    modeldir = Result_struct.modeldir
    preddir = Result_struct.preddir
    

    
    ######################################## start outer cv 

    # # get outer_cv splits (stratifiedgroupkfold) while controlling for family structure (HCP!)
    # outer_cv, groups = cv_control_for_family(subject_list, df_pheno, phenostr, seed, k_outer, df_family_info)

    ## create sex-balanced, family-structure-controlled splits when regression (31 Jan. 2023 added):
    outer_cv, groups = cv_control_for_family(subject_list, df_pheno, 'Gender', seed, k_outer, df_family_info)


    # start outer_cv
    for foldind, (train_index, test_index) in enumerate(outer_cv):
        # print
        print('current fold: ', foldind+1)

        # train and test subject_id lists
        sub_train = subject_list[train_index]
        sub_test = subject_list[test_index]


        ######################################## get features and save dataframes
        # feature file names
        ftestfile = eval(Result_struct.featurefname_test)
        ftrainfile = eval(Result_struct.featurefname_train)
        varfname = eval(Result_struct.varfname)

        # if exist, read in
        if os.path.isfile(ftestfile) and os.path.isfile(ftrainfile) and os.path.isfile(varfname):
            print('Features already extracted and saved.')
            print('Test data path: ', ftestfile)
            print('Train data path: ', ftrainfile)
            test_data = pd.read_csv(ftestfile)
            train_data = pd.read_csv(ftrainfile)
            X_keys = [col for col in test_data.columns if 'PC' in col]
            var_data = pd.read_csv(varfname)

        # otherwise, get features - PC loadings and save for later use    
        else:
            F,Ft,Var,PCs,X_keys = perform_feature_extraction(df_m, sub_train, sub_test, nroi, pcind, feature_type, ntr, nstart, flip, clean)
            train_data =get_dataframe(F,sub_train,df_pheno,phenostr,X_keys)
            test_data = get_dataframe(Ft,sub_test,df_pheno,phenostr,X_keys)
            # train_data = train_data.reindex(sorted(train_data.columns), axis=1) # reorder features as first PC1 for all rois, then PC2 ...
            # test_data = test_data.reindex(sorted(test_data.columns), axis=1)
            test_data.to_csv(ftestfile, header=True,index=True)
            train_data.to_csv(ftrainfile, header=True,index=True)
            var_data = pd.DataFrame(data=Var,columns=X_keys)
            var_data.to_csv(varfname,header=True,index=True)   
        print('features:', X_keys)
        # conduct feature selection if threshold is not 0
        if threshold:
            F, Ft, X_keys = feature_selection(train_data, test_data, var_data, threshold)
        else:
            F = train_data[X_keys].values
            Ft = test_data[X_keys].values

        print('Number of features used:', len(X_keys))

        # normalizing each feature for training and test
        if norm:
            F,Ft = normalise_feature(F,Ft)
        
        # transform the preprocessed features as dataframes
        train_data = get_dataframe(F,sub_train,df_pheno,phenostr,X_keys)
        test_data = get_dataframe(Ft,sub_test,df_pheno,phenostr,X_keys)
        
        # permute train data for pemutation test
        if permseed>0:
            train_data = perm_traindata(F,sub_train,subject_list,df_pheno,phenostr,X_keys,permseed)

        ######################################## settings for inner cv 
        y_keys = phenostr
        

        # Default: stratifiedkfold for inner loops, repeated 5 times; or simply run_cross_validation(cv=5)
        # inner_cv = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=seed)

        # if stratifiedgroupkfold for inner loops; run_cross_validation(groups=groups)
        # inner_cv, groups = cv_control_for_family(sub_train, df_pheno, phenostr, seed, k_inner, df_family_info)
       
        
        
        ######################################## train model on training data using inner-cv settings

        # Note: this part could be replaced by sklearn functions if needed
        ##################################### Julearn specific ########################################################
        scores, estimator = run_cross_validation(
        X=X_keys, y=y_keys, data=train_data, preprocess_X=None,
        problem_type=J_model.probtype, model= J_model.name, model_params=J_model.model_params, return_estimator='final',
        cv=k_inner, seed=seed)
         ###############################################################################################################

    
        ######################################## save the best parameters and models of each fold
        
        # when tuning hyperparameters: should has attribute "best_estimator_"
        if hasattr(estimator, 'best_estimator_'):
            model = estimator.best_estimator_
            bp = estimator.best_params_
            print('best parameter: ', bp)
            best_param = concatenate_best_params(best_param, bp, foldind)
        else:
            print('Has no attribute best_estimator_')
            model = estimator
    
        ######################################## save fitted model
        # save to fitted_models folder (if non-perm data)
        if permseed<=0:
            filename = eval(Result_struct.modelfname)
            pickle.dump(model, open(filename, 'wb'))

        ######################################## save predictions of each fold

        # predict on test data
        ytest_pred = model.predict(test_data[X_keys])
        ytest = test_data[y_keys]

        # final training predictions
        ytrain_pred = model.predict(train_data[X_keys])
        ytrain = train_data[y_keys]

        # get predict_proba on test for calculating roc_auc for binary classification if available
        ytest_proba = get_predproba_for_roc_auc(model, J_model, test_data, X_keys)

        # Print out some scores just out of curiosity
        if J_model.probtype == 'binary_classification':
            bacc_test = balanced_accuracy_score(ytest, ytest_pred)
            bacc_train = balanced_accuracy_score(ytrain, ytrain_pred)
            print('Test balanced accuracy for fold', foldind+1, 'is', bacc_test)
            print('Training balanced accuracy for fold', foldind+1, 'is', bacc_train)
        elif J_model.probtype == 'regression':
            pr_test = scipy.stats.pearsonr(ytest, ytest_pred)
            pr_train = scipy.stats.pearsonr(ytrain, ytrain_pred)
            print('Test pearson correlation for fold', foldind+1, 'is', pr_test[0])
            print('Training pearson correlation for fold', foldind+1, 'is', pr_train[0])

        ######################################## concatenate results across all folds
        pred_train = pred_train + list(ytrain_pred)
        true_train = true_train + list(ytrain)

        pred_test = pred_test + list(ytest_pred)
        true_test = true_test + list(ytest)
        proba_test = proba_test + list(ytest_proba)

        #fold_label_train = fold_label_train + [f'fold{foldind+1}']*len(sub_train)
        fold_label_train = fold_label_train + [foldind+1]*len(sub_train)
        fold_label_test = fold_label_test + [foldind+1]*len(sub_test)
        
        seed_label_train = seed_label_train + [seed]*len(sub_train)
        seed_label_test = seed_label_test + [seed]*len(sub_test)

        subid_train = subid_train + list(sub_train)
        subid_test = subid_test + list(sub_test)
        

    ######################################## save prediction results as dataframes for training and test separately of all folds   
    Ptest = []
    Ptest.append(pred_test)
    Ptest.append(true_test)
    df_pred_test = pd.DataFrame(Ptest).T
    df_pred_test.columns = ['pred','true']
    if proba_test: 
        df_pred_test['proba'] = proba_test
    df_pred_test.insert(0,'Subject',subid_test)
    df_pred_test.insert(0,'fold',fold_label_test)
    df_pred_test.insert(0,'seed',seed_label_test)

    Ptrain = []
    Ptrain.append(pred_train)
    Ptrain.append(true_train)
    df_pred_train = pd.DataFrame(Ptrain).T
    df_pred_train.columns = ['pred','true']
    df_pred_train.insert(0,'Subject',subid_train)
    df_pred_train.insert(0,'fold',fold_label_train)
    df_pred_train.insert(0,'seed',seed_label_train)
    
    # save to prediction folder only if non-perm
    if permseed<=0:
        fileptest = eval(Result_struct.predfname_test)
        df_pred_test.to_csv(fileptest, header=True)
        fileptrain = eval(Result_struct.predfname_train)
        df_pred_train.to_csv(fileptrain, header=True)

    ######################################## save best parameters of all folds
    BP = []
    parlist = list(best_param.keys())
    for p in parlist:
        BP.append(list(best_param[p]))
    df_best_params = pd.DataFrame(BP).T
    df_best_params.columns = parlist
    print('best paramters:')
    print(df_best_params)
    df_best_params.insert(0,'fold', np.arange(1,k_outer+1,1))
    df_best_params.insert(0,'seed',[seed] * k_outer)
    #print(df_best_params)

    # save to model folder
    if permseed<=0:
        filebp = eval(Result_struct.modelfname_bp)
        df_best_params.to_csv(filebp, header=True, index=True)

    #return df_pred_test, df_pred_train, df_best_params
    return df_pred_test, df_pred_train, df_best_params


def main_compute_prediction_scores(fmricondition, clfname, probtype, Result_struct, metric_list, seed_list, nfold, savepath, foldwise=1, test=True):
    """ compute prediction scores using specified measures for a given fmricondition+clfname """
    # fmricondition, clfname will be needed when loading and saving file
    # Result_struct: output of main_TOPF, gives the path to folders of prediction results or just Result_struct = init_result_dir(rdir,clfdir)
    # metric_list = ['accuracy', 'balanced_accuracy', 'roc_auc'] for binary classification
    # metric_list = ['r2_score','mean_absolute_error','spearman','pearson'] for regression
    # metric_list should be given by user, but needs to be added to the func "cal_measures" beforehand
    # seed_list used
    # nfold: k_outer
    # foldwise = 1 : compute within each fold; 0/None: compute over all samples within each seed/rep
    # test=True: compute for predictions derived on test; otherwise: compute for predictions derived on training
    
    m_values = []
    m_labels = []
    fold_label = []
    seed_label = []
    mean_seed = []

    preddir = Result_struct.preddir
    if test:
        file = Result_struct.predfname_test
    else:
        file = Result_struct.predfname_train
    

    for seed in seed_list:
        
        df_test = pd.read_csv(eval(file))
        m_seed = []

        if foldwise: 
            print('Measures are computed within each fold')
            
            for fold in range(1,nfold+1,1):
                
                #df_temp = df_test[(df_test.seed==str(seed)) & (df_test.fold==str(fold))].reset_index(drop=True)
                df_temp = df_test[(df_test.seed==seed) & (df_test.fold==fold)].reset_index(drop=True)
                m_temp, mlist = cal_measures(df_temp, probtype, metric_list)  # calculate measures
                m_values = m_values + m_temp
                fold_label = fold_label + [fold]
                seed_label = seed_label + [seed]
                m_seed = m_seed + m_temp # within each seed

            # mean over all folds for a seed 
            m_seed = np.reshape(m_seed, [-1, len(mlist)]) 
            mean_seed = mean_seed + list(np.mean(m_seed, axis = 0))

        else:
            print('Measures are computed within each seed over all samples')
            df_temp = df_test[df_test.seed==seed].reset_index(drop=True)
            m_temp, mlist = cal_measures(df_temp, probtype, metric_list)  # calculate measures
            m_values = m_values + m_temp
            seed_label = seed_label + [seed]
            fold_label = []
            mean_seed = m_values 

    m_labels = mlist

    # save as dataframe
    m_values = np.array(m_values)
    m_values = np.reshape(m_values, [-1, len(m_labels)])
    df_measure = pd.DataFrame(data=m_values,columns=m_labels)
    if fold_label:
        df_measure.insert(0,'fold', fold_label)
    df_measure.insert(0,'seed', seed_label)

    # print results
    mean_all = np.mean(m_values, axis = 0)
    print('The measures computed are: ', m_labels)
    print('The mean over all seeds are: ', mean_all)

    mean_seed = np.reshape(mean_seed, [-1, len(m_labels)])
    # save
    if savepath:
        df_measure.to_csv(savepath, header=True)

    return mean_all, mean_seed, df_measure, m_labels



############################################# END #############################################
