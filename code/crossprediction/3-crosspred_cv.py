import os
import ast
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve, roc_auc_score, precision_score
import scipy.stats
from julearn import run_cross_validation
from julearn.utils import configure_logging

## configure_logging_julearn
configure_logging(level='INFO')



############################################# functions ##############################################################
def flip_pc_loadings(F, flip=None):
    """
    flip pc loadings when needed, eg. for cross-condition comparison
    F: features (pc loadings) - nsub * nfeature
    """
    nsub = F.shape[0]
    nfeature = F.shape[1]
    fsign = np.empty([1,0],float) #  -1 (flip) or 1 (not)


    for i in range(0,nfeature):
        pc_loading = F[:,i]
        t = 1

        # make sure the maximum absolute value comes from postive
        if flip =='abs':
            if np.abs(pc_loading).max()>np.max(pc_loading):
                t = -1
        
        # make sure number of pos larger than number of neg
        elif flip == 'num':
            if np.sum(pc_loading >= 0) < np.sum(pc_loading < 0):
                t = -1
        else:
            # do nothing
            t = 1
        fsign = np.append(fsign,t)

    return fsign


def main_crosspred(ddir, clfdir, mtrain, mtest, J_model, seed, phenostr, k_outer=10, flip=None, cleanup=None):
    '''
    ddir: data dir (to get features before normalisation and fitted models) - output of 2-classification
    clfdir: which clf
    mtrain: movie (fmricondition) for training
    mtest: movie to be tested
    flip: two ways - abs or num; none - no flip;
    '''
    ######################################## create empty list to save results
    
    # target
    y_keys = phenostr

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


    # dirs of features (before normalisation) and fitted models
    featuredir = ddir + '/features/'
    modeldir = ddir + '/'+ clfdir + '/fitted_models/'
   
    # start cv
    for foldind in range(0,k_outer):
        
        # print
        print('current fold: ', foldind+1)

        # load the fitted model
        filename = modeldir+f"model_{mtrain}_seed{seed}_fold{foldind+1}.sav"
        if os.path.isfile(filename):
            model = pickle.load(open(filename, 'rb'))
        else:
            print('Cannot find the fitted model!')
            print(filename)

        # load training and test features
        ftrainfile = featuredir+f"loadings_train_{mtrain}_seed{seed}_fold{foldind+1}.csv" # training of movie_train
        ftestfile = featuredir+f"loadings_test_{mtest}_seed{seed}_fold{foldind+1}.csv" # test of movie_test
        ftestfile_train = featuredir+f"loadings_train_{mtest}_seed{seed}_fold{foldind+1}.csv" # train of movie_test

        if os.path.isfile(ftrainfile) and os.path.isfile(ftestfile) and os.path.isfile(ftestfile_train):
            train_data = pd.read_csv(ftrainfile)
            X_keys = [col for col in train_data.columns if 'PC' in col]
            #print('training features:', X_keys)
            test_data = pd.read_csv(ftestfile)
            
            sub_test = test_data['Subject'].values
            sub_train = train_data['Subject'].values

            #########################################
            # clean up the mess in feature order to match feature order used in model training
            if cleanup:
            
                if feature_type == 'singlePC':
                    X_keys_ordered = [f'PC1_%d' % i for i in range(1, 436+1, 1)]
                else:
                    X_keys_ordered = []
                    for j in range(1,437):
                        for i in range(1,3):
                            X_keys_ordered = X_keys_ordered + [f'PC{i}_{j}']
            
                #print('X_key_ordered:', X_keys_ordered)

                # reorder train features
                if X_keys != X_keys_ordered:
                    print('reorder features to match with model and save')
                    new_train = train_data[X_keys_ordered]
                    new_train.insert(0,'Gender',train_data['Gender'].values)
                    new_train.insert(0,'Subject', sub_train)
                    train_data = new_train
                    new_train.to_csv(ftrainfile, header=True,index=True)
                    print('reordered train:', mtrain)

                # reorder test features    
                X_keys = [col for col in test_data.columns if 'PC' in col]
                if X_keys != X_keys_ordered:
                    new_test = test_data[X_keys_ordered]
                    new_test.insert(0,'Gender',test_data['Gender'].values)
                    new_test.insert(0,'Subject', sub_test)
                    test_data = new_test
                    new_test.to_csv(ftestfile, header=True,index=True)
                    print('reordered test', mtest)
           
                # test movie train
                test_data_train = pd.read_csv(ftestfile_train) # train of movie_test where PC scores are derived to check if flip is needed
                X_keys = [col for col in test_data_train.columns if 'PC' in col]
                if X_keys != X_keys_ordered:
                    new_test_train = test_data_train[X_keys_ordered]
                    new_test_train.insert(0,'Gender',test_data_train['Gender'].values)
                    print(len(sub_train))
                    new_test_train.insert(0,'Subject', sub_train)
                    test_data_train = new_test_train
                    new_test_train.to_csv(ftestfile_train, header=True,index=True)
                    print('reordered test train', mtest)

                X_keys = X_keys_ordered
        ################################# clean up above (27 Jan)

            F = train_data[X_keys].values
            Ft = test_data[X_keys].values

        # flip test features if necessary
            fsign_train = flip_pc_loadings(F, flip)
            test_data_train = pd.read_csv(ftestfile_train) # train of movie_test where PC scores are derived to check if flip is needed
            Ft_train = test_data_train[X_keys].values
            fsign_test = flip_pc_loadings(Ft_train, flip)
            fsign = fsign_train * fsign_test
            Ft = Ft * fsign
            nflip = list(np.argwhere(fsign==-1))
            if nflip:
                print('flipped some features')

        # normalise feature (fit on train and apply to test)
            F,Ft = TOPF.normalise_feature(F,Ft)
            test_data[X_keys] = Ft
            train_data[X_keys] = F
        else:
            print('features missing!')

            
        ######################################## save predictions of each fold
        # predict on test data
        ytest_pred = model.predict(test_data[X_keys])
        ytest = test_data[y_keys]

        # get predict_proba on test for calculating roc_auc for binary classification if available
        ytest_proba = TOPF.get_predproba_for_roc_auc(model, J_model, test_data, X_keys)

        # final training predictions
        ytrain_pred = model.predict(train_data[X_keys])
        ytrain = train_data[y_keys]

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

    return df_pred_test, df_pred_train


def compute_prediction_scores(predfolder, J_model, mtrain, mtest, metric_list, seed_list, nfold, foldwise=1):
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
    mtest_label = []
    mtest_label_seed = []
    seed_label_seed = []
    probtype = J_model.probtype
 
    
    



    for seed in seed_list:
        #movielist = list(set(movienames) - {mtrain})
        predfile = "predfolder + f'/pred_train_{mtrain}_test_{mtest}_seed{seed}.csv'"

        df_test = pd.read_csv(eval(predfile))
        m_seed = []

        if foldwise: 
            print('Measures are computed within each fold')
            
            for fold in range(1,nfold+1,1):
                
                #df_temp = df_test[(df_test.seed==str(seed)) & (df_test.fold==str(fold))].reset_index(drop=True)
                df_temp = df_test[(df_test.seed==seed) & (df_test.fold==fold)].reset_index(drop=True)
                m_temp, mlist = TOPF.cal_measures(df_temp, probtype, metric_list)  # calculate measures
                m_values = m_values + m_temp
                fold_label = fold_label + [fold]
                seed_label = seed_label + [seed]
                m_seed = m_seed + m_temp # within each seed
                mtest_label = mtest_label + [mtest]

            # mean over all folds for a seed 
            m_seed = np.reshape(m_seed, [-1, len(mlist)]) 
            mean_seed = mean_seed + list(np.mean(m_seed, axis = 0))
            seed_label_seed = seed_label_seed + [seed]
            mtest_label_seed = mtest_label_seed+ [mtest]

        else:
            print('Measures are computed within each seed over all samples')
            df_temp = df_test[df_test.seed==seed].reset_index(drop=True)
            m_temp, mlist = TOPF.cal_measures(df_temp, probtype, metric_list)  # calculate measures
            m_values = m_values + m_temp
            seed_label = seed_label + [seed]
            fold_label = []
            mean_seed = m_values 
            mtest_label_seed = mtest_label_seed+ [mtest]

    m_labels = mlist

    # save as dataframe
    m_values = np.array(m_values)
    m_values = np.reshape(m_values, [-1, len(m_labels)])
    df_measure = pd.DataFrame(data=m_values,columns=m_labels)

    # foldwise
    if fold_label:
        df_measure.insert(0,'fold', fold_label)
    df_measure.insert(0,'seed', seed_label)
    df_measure.insert(0,'mtest',mtest_label)

    # print results
    mean_all = np.mean(m_values, axis = 0)
    print('The measures computed are: ', m_labels)
    print('The mean over all seeds are: ', mean_all)

    # mean of seed
    mean_seed = np.reshape(mean_seed, [-1, len(m_labels)])
    df_seedwise = pd.DataFrame(data=mean_seed,columns=m_labels)
    df_seedwise.insert(0,'seed', seed_label_seed)
    df_seedwise.insert(0,'mtest',mtest_label_seed)

    return df_measure, df_seedwise, m_labels

######################################################### input ################################
wkdir = sys.argv[1]
fdir = sys.argv[2]
dataset = sys.argv[3]
nroi = int(sys.argv[4])
phenostr = sys.argv[5]
cmp_name = sys.argv[6]
ntr = int(sys.argv[7])
feature_type = sys.argv[8] # 'singlePC' or combinedPC
pcind = int(sys.argv[9]) # which PC or to the largest PC index when combinedPC
flip = sys.argv[10] # 'abs' or 'num', flip pc and scores, needed for cross-pred
setting_file = sys.argv[11]
clfname = sys.argv[12]
clfdir = sys.argv[13]
mtrainind = int(sys.argv[14])

#mtestind = int(sys.argv[5])


k_outer = 10
k_inner = 5
seedlist = list(np.arange(0,10))
cleanup = None

# dfolder = f'{dataset}/{phenostr}/{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_flip{flip}_clean{str(clean)}_n{nroi}'
# rfolder = f'{dataset}/{phenostr}/{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_flip{flip}_clean{str(clean)}_n{nroi}'

if ntr <=0:
    ntr = None  # full length 
    nstart = 0



# adding core_functions folder to the system path and import the TOPF module (TOPF.py)
sys.path.insert(0, wkdir+'/code/core_functions')
import TOPF

# subfolder naming patterns
dfolder = f'{dataset}/{phenostr}/{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_ko{k_outer}_ki{k_inner}_n{nroi}'
rfolder = f'{dataset}/{phenostr}/{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_flip{flip}'

# result
r_rootdir = '/data/project/brainvar_topf_eval/results/crossprediction/'
rdir = r_rootdir + rfolder
predfolder = rdir + '/prediction'
if not os.path.exists(f"{predfolder}"):
    os.makedirs(f"{predfolder}")

# datadir
#fdir = wkdir + '/results/classification/feature_before_norm/'
ddir = fdir + dfolder

# set up machine learning model
J_model = TOPF.create_ML_julearn_model(clfname, setting_file)

# movies
if cmp_name == 'moviewise':
    movienames = ['two_men','bridgeville','pockets','overcome','inception','social_net', 'oceans_11', 'flower', 'hotel', 'garden', 'dreary', 'home_alone', 'brokovich', 'star_wars']
elif cmp_name == 'wholerun':
    movienames = ['run1', 'run2', 'run3', 'run4']
else:
    movienames = []

#movienames = ['two_men','bridgeville']

# metrics
metric_list = ['balanced_accuracy', 'roc_auc']

# score dir
sdir = wkdir + '/results_summary/crossprediction'

######################################################### output #################################

mtrain = movienames[mtrainind-1]
print('train on movie:', mtrain)

df_full_scores = pd.DataFrame()
df_scores = pd.DataFrame()

for mtest in movienames:
    print('test on movie:', mtest)

    for seed in range(0,10):
        print('seed:', seed)
        # get predictions
        df_pred_test, df_pred_train = main_crosspred(ddir, clfdir, mtrain, mtest, J_model, seed, phenostr, k_outer, flip)
            
        # save to prediction folder
        fileptest = predfolder + f'/pred_train_{mtrain}_test_{mtest}_seed{seed}.csv'
        df_pred_test.to_csv(fileptest, header=True)

    # calculate scores - summarise
    df_measure, df_seedwise, _ = compute_prediction_scores(predfolder, J_model, mtrain, mtest, metric_list, seedlist, k_outer, foldwise=1)
    df_full_scores = pd.concat([df_full_scores, df_measure], ignore_index=True)
    df_scores = pd.concat([df_scores, df_seedwise], ignore_index=True)
        
    filename = sdir + f'/full_scores_train_{mtrain}_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_flip{flip}.csv'
    filename2 = sdir + f'/scores_train_{mtrain}_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_flip{flip}.csv'

    df_full_scores.to_csv(filename, header=True)                                                  
    df_scores.to_csv(filename2, header=True)
    print('train on movie:', mtrain)
    print('test on movie:', mtest)
    print('saved computed scores to', filename2)



################################ loop over all movies
# for i, mtrain in enumerate(movienames):
#     print('train on movie:', mtrain)
#     df_full_scores = pd.DataFrame()
#     df_scores = pd.DataFrame()

#     for j, mtest in enumerate(movienames):
#         print('train on movie:', mtrain)
#         print('test on movie:', mtest)

#         for seed in range(0,10):
#             print('seed:', seed)
#             # get predictions
#             df_pred_test, df_pred_train = main_crosspred(ddir, clfdir, mtrain, mtest, J_model, seed, phenostr, k_outer, flip)
            
#             # save to prediction folder
#             fileptest = predfolder + f'/pred_train_{mtrain}_test_{mtest}_seed{seed}.csv'
#             df_pred_test.to_csv(fileptest, header=True)

#         # calculate scores - summarise
#         seedlist = list(np.arange(0,10))
#         df_measure, df_seedwise, _ = compute_prediction_scores(predfolder, J_model, mtrain, mtest, metric_list, seedlist, k_outer, foldwise=1)
#         df_full_scores = pd.concat([df_full_scores, df_measure], ignore_index=True)
#         df_scores = pd.concat([df_scores, df_seedwise], ignore_index=True)
        
#         filename = sdir + f'/full_scores_train_{mtrain}_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_flip{flip}.csv'
#         filename2 = sdir + f'/scores_train_{mtrain}_{dataset}_{phenostr}_{cmp_name}_cut_{str(ntr)}_{feature_type}_{pcind}_flip{flip}.csv'

#         df_full_scores.to_csv(filename, header=True)                                                  
#         df_scores.to_csv(filename2, header=True)
#         print('train on movie:', mtrain)
#         print('test on movie:', mtest)
#         print('saved computed scores to', filename2)
        

