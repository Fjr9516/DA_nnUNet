import numpy as np
import nibabel as nib
import cc3d
import scipy
import os
import pandas as pd
import surface_distance
import sys
import math
import shutil
import multiprocessing
import SimpleITK as sitk

def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 3
    new_seg[seg == 2] = 1
    return new_seg

def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(os.path.join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, os.path.join(output_folder, filename))

def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def dice(im1, im2):
    """
    Computes Dice score for two images

    Parameters
    ==========
    im1: Numpy Array/Matrix; Predicted segmentation in matrix form 
    im2: Numpy Array/Matrix; Ground truth segmentation in matrix form

    Output
    ======
    dice_score: Dice score between two images
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * (intersection.sum()) / (im1.sum() + im2.sum())

def get_TissueWiseSeg(prediction_matrix, gt_matrix, tissue_type):
    """
    Converts the segmentatations to isolate tissue types

    Parameters
    ==========
    prediction_matrix: Numpy Array/Matrix; Predicted segmentation in matrix form 
    gt_matrix: Numpy Array/Matrix; Ground truth segmentation in matrix form
    tissue_type: str; Can be WT, ET or TC

    Output
    ======
    prediction_matrix: Numpy Array/Matrix; Predicted segmentation in matrix form with 
                       just tissue type mentioned
    gt_matrix: Numpy Array/Matrix; Ground truth segmentation in matrix form with just 
                       tissue type mentioned
    """

    if tissue_type == 'WT':
        np.place(prediction_matrix, (prediction_matrix != 1) & (prediction_matrix != 2) & (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)
        
        np.place(gt_matrix, (gt_matrix != 1) & (gt_matrix != 2) & (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)
    
    elif tissue_type == 'TC':
        np.place(prediction_matrix, (prediction_matrix != 1)  & (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)
        
        np.place(gt_matrix, (gt_matrix != 1) & (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)
        
    elif tissue_type == 'ET':
        np.place(prediction_matrix, (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)
        
        np.place(gt_matrix, (gt_matrix != 3), 0)
        np.place(gt_matrix, (gt_matrix > 0), 1)
    
    return prediction_matrix, gt_matrix


def get_GTseg_combinedByDilation(gt_dilated_cc_mat, gt_label_cc):
    """
    Computes the Corrected Connected Components after combing lesions
    together with respect to their dilation extent

    Parameters
    ==========
    gt_dilated_cc_mat: Numpy Array/Matrix; Ground Truth Dilated Segmentation 
                       after CC Analysis
    gt_label_cc: Numpy Array/Matrix; Ground Truth Segmentation after 
                       CC Analysis

    Output
    ======
    gt_seg_combinedByDilation_mat: Numpy Array/Matrix; Ground Truth 
                                   Segmentation after CC Analysis and 
                                   combining lesions
    """    
    
    
    gt_seg_combinedByDilation_mat = np.zeros_like(gt_dilated_cc_mat)

    for comp in range(np.max(gt_dilated_cc_mat)):  
        comp += 1

        gt_d_tmp = np.zeros_like(gt_dilated_cc_mat)
        gt_d_tmp[gt_dilated_cc_mat == comp] = 1
        gt_d_tmp = (gt_label_cc*gt_d_tmp)

        np.place(gt_d_tmp, gt_d_tmp > 0, comp)
        gt_seg_combinedByDilation_mat += gt_d_tmp
        
    return gt_seg_combinedByDilation_mat


def get_LesionWiseScores(prediction_seg, gt_seg, label_value, dil_factor):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth
    segmentations

    Parameters
    ==========
    prediction_seg: str; location of the prediction segmentation    
    gt_label_cc: str; location of the gt segmentation
    label_value: str; Can be WT, ET or TC
    dil_factor: int; Used to perform dilation

    Output
    ======
    tp: Number of TP lesions WRT prediction segmentation
    fn: Number of FN lesions WRT prediction segmentation
    fp: Number of FP lesions WRT prediction segmentation 
    gt_tp: Number of Ground Truth TP lesions WRT prediction segmentation 
    metric_pairs: list; All the lesion-wise metrics  
    full_dice: Dice Score of the pair of segmentations
    full_gt_vol: Total Ground Truth Segmenatation Volume
    full_pred_vol: Total Prediction Segmentation Volume
    """

    ## Get Prediction and GT segs matrix files
    pred_nii = nib.load(prediction_seg)
    gt_nii = nib.load(gt_seg)
    pred_mat = pred_nii.get_fdata()
    gt_mat = gt_nii.get_fdata()

    ## Get Spacing to computes volumes
    ## Brats Assumes all spacing is 1x1x1mm3
    sx, sy, sz = pred_nii.header.get_zooms()

    ## Get the prediction and GT matrix based on 
    ## WT, TC, ET

    pred_mat, gt_mat = get_TissueWiseSeg(
                                prediction_matrix = pred_mat,
                                gt_matrix = gt_mat,
                                tissue_type = label_value
                            )
    
    ## Get Dice score for the full image
    if np.all(gt_mat==0) and np.all(pred_mat==0):
        full_dice = 1.0
    else:
        full_dice = dice(
                    pred_mat, 
                    gt_mat
                )
    
    ## Get HD95 sccre for the full image
    
    if np.all(gt_mat==0) and np.all(pred_mat==0):
        full_hd95 = 0.0
    else:
        full_sd = surface_distance.compute_surface_distances(gt_mat.astype(int), 
                                                             pred_mat.astype(int), 
                                                             (sx,sy,sz))
        full_hd95 = surface_distance.compute_robust_hausdorff(full_sd, 95)

    ## Get Sensitivity and Specificity
    full_sens, full_specs = get_sensitivity_and_specificity(result_array = pred_mat, 
                                                            target_array = gt_mat)
    
    ## Get GT Volume and Pred Volume for the full image
    full_gt_vol = np.sum(gt_mat)*sx*sy*sz
    full_pred_vol = np.sum(pred_mat)*sx*sy*sz

    ## Performing Dilation and CC analysis

    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)

    gt_mat_cc = cc3d.connected_components(gt_mat, connectivity=26)
    pred_mat_cc = cc3d.connected_components(pred_mat, connectivity=26)

    gt_mat_dilation = scipy.ndimage.binary_dilation(gt_mat, structure = dilation_struct, iterations = dil_factor)
    gt_mat_dilation_cc = cc3d.connected_components(gt_mat_dilation, connectivity=26)

    gt_mat_combinedByDilation = get_GTseg_combinedByDilation(
                                                            gt_dilated_cc_mat = gt_mat_dilation_cc, 
                                                            gt_label_cc = gt_mat_cc
                                                        )
    
    ## Performing the Lesion-By-Lesion Comparison

    gt_label_cc = gt_mat_combinedByDilation
    pred_label_cc = pred_mat_cc

    gt_tp = []
    tp = []
    fn = []
    fp = []
    metric_pairs = []

    for gtcomp in range(np.max(gt_label_cc)):
        gtcomp += 1

        ## Extracting current lesion
        gt_tmp = np.zeros_like(gt_label_cc)
        gt_tmp[gt_label_cc == gtcomp] = 1

        ## Extracting ROI GT lesion component
        gt_tmp_dilation = scipy.ndimage.binary_dilation(gt_tmp, structure = dilation_struct, iterations = dil_factor)

        # Volume of lesion
        gt_vol = np.sum(gt_tmp)*sx*sy*sz 
        
        ## Extracting Predicted true positive lesions
        pred_tmp = np.copy(pred_label_cc)
        #pred_tmp = pred_tmp*gt_tmp
        pred_tmp = pred_tmp*gt_tmp_dilation
        intersecting_cc = np.unique(pred_tmp) 
        intersecting_cc = intersecting_cc[intersecting_cc != 0] 
        for cc in intersecting_cc:
            tp.append(cc)

        ## Isolating Predited Lesions to calulcate Metrics
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp[np.isin(pred_tmp,intersecting_cc,invert=True)] = 0
        pred_tmp[np.isin(pred_tmp,intersecting_cc)] = 1

        ## Calculating Lesion-wise Dice and HD95
        dice_score = dice(pred_tmp, gt_tmp)
        surface_distances = surface_distance.compute_surface_distances(gt_tmp, pred_tmp, (sx,sy,sz))
        hd = surface_distance.compute_robust_hausdorff(surface_distances, 95)

        metric_pairs.append((intersecting_cc, 
                            gtcomp, gt_vol, dice_score, hd))
        
        ## Extracting Number of TP/FP/FN and other data
        if len(intersecting_cc) > 0:
            gt_tp.append(gtcomp)
        else:
            fn.append(gtcomp)

    fp = np.unique(
            pred_label_cc[np.isin(
                pred_label_cc,tp+[0],invert=True)])
    
    return tp, fn, fp, gt_tp, metric_pairs, full_dice, full_hd95, full_gt_vol, full_pred_vol, full_sens, full_specs


def get_sensitivity_and_specificity(result_array, target_array):
    """
    This function is extracted from GaNDLF from mlcommons

    You can find the documentation here - 

    https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/metrics/segmentation.py#L196

    """
    iC = np.sum(result_array)
    rC = np.sum(target_array)

    overlap = np.where((result_array == target_array), 1, 0)

    # Where they agree are both equal to that value
    TP = overlap[result_array == 1].sum()
    FP = iC - TP
    FN = rC - TP
    TN = np.count_nonzero((result_array != 1) & (target_array != 1))

    Sens = 1.0 * TP / (TP + FN + sys.float_info.min)
    Spec = 1.0 * TN / (TN + FP + sys.float_info.min)

    # Make Changes if both input and reference are 0 for the tissue type
    if (iC == 0) and (rC == 0):
        Sens = 1.0

    return Sens, Spec



def get_LesionWiseResults(pred_file, gt_file, challenge_name, output=None):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth
    segmentations

    Parameters
    ==========
    pred_file: str; location of the prediction segmentation    
    gt_file: str; location of the gt segmentation
    challenge_name: str; name of the challenge for parameters


    Output
    ======
    Saves the performance metrics as CSVs
    results_df: pd.DataFrame; lesion-wise results with other metrics
    """
    
    ## Dilation and Threshold Parameters
    if challenge_name == 'BraTS-GLI':
        dilation_factor = 3
        lesion_volume_thresh = 50
    elif challenge_name == 'BraTS-SSA':
        dilation_factor = 3
        lesion_volume_thresh = 50
    elif challenge_name == 'BraTS-MEN':
        dilation_factor = 1
        lesion_volume_thresh = 50
    elif challenge_name == 'BraTS-PED':
        dilation_factor = 3
        lesion_volume_thresh = 50
    elif challenge_name == 'BraTS-MET':
        dilation_factor = 1
        lesion_volume_thresh = 2       
        

    final_lesionwise_metrics_df = pd.DataFrame()
    final_metrics_dict = dict()
    label_values = ['WT', 'TC', 'ET']

    for l in range(len(label_values)):
        tp, fn, fp, gt_tp, metric_pairs, full_dice, full_hd95, full_gt_vol, full_pred_vol, full_sens, full_specs = get_LesionWiseScores(
                                                            prediction_seg = pred_file,
                                                            gt_seg = gt_file,
                                                            label_value = label_values[l],
                                                            dil_factor = dilation_factor
                                                        )
        
        metric_df = pd.DataFrame(
            metric_pairs, columns=['predicted_lesion_numbers', 'gt_lesion_numbers', 
                                   'gt_lesion_vol', 'dice_lesionwise', 'hd95_lesionwise']
                ).sort_values(by = ['gt_lesion_numbers'], ascending=True).reset_index(drop = True)
        
        metric_df['_len'] = metric_df['predicted_lesion_numbers'].map(len)

        ## Removing <= 50 lesions from analysis
        fn_sub = (metric_df[(metric_df['_len'] == 0) &
                  (metric_df['gt_lesion_vol'] <= lesion_volume_thresh)
                  ]).shape[0]
        
        
        gt_tp_sub = (metric_df[(metric_df['_len'] != 0) & 
            (metric_df['gt_lesion_vol'] <= lesion_volume_thresh)
            ]).shape[0]
        
        
        metric_df['Label'] = [label_values[l]]*len(metric_df)
        metric_df = metric_df.replace(np.inf, 374)

        final_lesionwise_metrics_df = final_lesionwise_metrics_df._append(metric_df)
        metric_df_thresh = metric_df[metric_df['gt_lesion_vol'] > lesion_volume_thresh]
        
        try:
            lesion_wise_dice = np.sum(metric_df_thresh['dice_lesionwise'])/(len(metric_df_thresh) + len(fp))
        except:
            lesion_wise_dice = np.nan
            
        try:
            lesion_wise_hd95 = (np.sum(metric_df_thresh['hd95_lesionwise']) + len(fp)*374)/(len(metric_df_thresh) + len(fp))
        except:
            lesion_wise_hd95 = np.nan

        if math.isnan(lesion_wise_dice):
            lesion_wise_dice = 1

        if math.isnan(lesion_wise_hd95):
            lesion_wise_hd95 = 0
        
        metrics_dict = {
            'Num_TP' : len(gt_tp) - gt_tp_sub, # GT_TP
            #'Num_TP' : len(tp),
            'Num_FP' : len(fp),
            'Num_FN' : len(fn) - fn_sub,
            'Sensitivity': full_sens,
            'Specificity': full_specs,
            'Legacy_Dice' : full_dice,
            'Legacy_HD95' : full_hd95,
            'GT_Complete_Volume' : full_gt_vol,
            'LesionWise_Score_Dice' : lesion_wise_dice,
            'LesionWise_Score_HD95' : lesion_wise_hd95
        }

        final_metrics_dict[label_values[l]] = metrics_dict


    #final_lesionwise_metrics_df.to_csv(os.path.split(pred_file)[0] + '/' +
    #                                   os.path.split(pred_file)[1].split('.')[0] + 
    #                                   '_lesionwise_metrics.csv',
    #                                   index=False)
    
    
    results_df = pd.DataFrame(final_metrics_dict).T
    results_df['Labels'] = results_df.index
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, 'Labels', results_df.pop('Labels'))
    results_df.replace(np.inf, 374, inplace=True)
    
    if output:
        results_df.to_csv(output, index=False)
    
    return results_df

if __name__ == '__main__':
    dataset_main_dir = '/patg/to/your/Datasets/'
    brats_data_dir = f'{dataset_main_dir}BraTS2023-PED/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/'
    nnUNet_raw = f'{dataset_main_dir}nnUNet/nnUNet_raw'
    Dataset = 'Dataset140_BraTS2023_PED'

    # 0/ prepare BraTS convention outputs
    CV_folders  = [
                    f'{dataset_main_dir}nnUNet/nnUNet_results/Dataset140_BraTS2023_PED/nnUNetTrainer_TL_300epochs__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessed/',
    ]
    folders_pred = [
                    f'{dataset_main_dir}nnUNet/nnUNet_results/Dataset140_BraTS2023_PED/nnUNetTrainer_TL_300epochs__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/BraST_convension/',
    ]
    for input_folder, output_folder in zip(CV_folders, folders_pred):
        print(f'From {input_folder} -> {output_folder}')
        convert_folder_with_preds_back_to_BraTS_labeling_convention(
            input_folder, output_folder, num_processes=12)

    # # 1/ prepare PED training labelsTr
    # case_ids = subdirs(f'{brats_data_dir}', prefix='BraTS', join=False)
    # labelstr = os.path.join( f'{nnUNet_raw}/{Dataset}', "labelsTr_BraTS_convention")
    #
    # maybe_mkdir_p(labelstr)
    #
    # for c in case_ids:
    #     shutil.copy(os.path.join(brats_data_dir, c, c + "-seg.nii.gz"), os.path.join(labelstr, c + '.nii.gz'))

    # 2/ calculate metrics
    challenge_name = 'BraTS-PED'

    folder_ref  = f'{nnUNet_raw}/{Dataset}/labelsTr_BraTS_convention/' # labelsTr_BraTS_convention

    file_ending = '.nii.gz'
    for folder_pred in folders_pred:
        files_pred0 = subfiles(folder_pred, suffix=file_ending, join=False)
        files_ref0 = subfiles(folder_ref, suffix=file_ending, join=False)

        # present = [os.path.isfile(os.path.join(folder_pred, i)) for i in files_ref0]
        # assert all(present), "Not all files in folder_pred exist in folder_ref" # JR: for fold 0

        present = [os.path.isfile(os.path.join(folder_ref, i)) for i in files_pred0]
        assert all(present), "Not all files in folder_pred exist in folder_ref"

        files_pred = [os.path.join(folder_pred, i) for i in files_pred0]
        files_ref = [os.path.join(folder_ref, i) for i in files_pred0]

        print(f'Len of files_ref = {len(files_ref)}\n')
        print(f'Len of files_pred = {len(files_pred)}\n')

        # Create an empty DataFrame to store the results
        final_results_df = pd.DataFrame(columns=[
        'Dice-ET', 'Dice-TC', 'Dice-WT',
        'HD95-ET', 'HD95-TC', 'HD95-WT',
        'Sensitivity-ET', 'Sensitivity-TC', 'Sensitivity-WT',
        'Specificity-ET', 'Specificity-TC', 'Specificity-WT',
        'LesionWise_Dice-ET', 'LesionWise_Dice-TC', 'LesionWise_Dice-WT',
        'LesionWise_HD95-ET', 'LesionWise_HD95-TC', 'LesionWise_HD95-WT'
        ])

        # Mapping dic
        Mapping_dic = {'Dice':'Legacy_Dice',
                       'HD95':'Legacy_HD95',
                       'Specificity':'Specificity',
                       'Sensitivity':'Sensitivity',
                       'LesionWise_Dice':'LesionWise_Score_Dice',
                       'LesionWise_HD95':'LesionWise_Score_HD95',
                       'GT_Complete_Volume':'GT_Complete_Volume'}
        regions = ['ET','TC','WT']
        case_id = 0
        order_id = []
        for gt_file, pred_file in zip(files_ref, files_pred):
            print(f'Calculate LesionWiseResults for this pair:')
            print(f'gt_file: {gt_file}')
            print(f'files_pred: {pred_file}')
            order_id.append(pred_file.split('/')[-1])

            results_df = get_LesionWiseResults(pred_file, gt_file, challenge_name, output=None)
            print(results_df)

            for title in final_results_df.columns:
                metric, region = title.split('-')
                final_results_df.at[case_id, title] = results_df.set_index('Labels').at[region, Mapping_dic[metric]]

            case_id+=1
            print(final_results_df)

        final_results_df['ID'] = order_id

        # Save the final results to a CSV file
        final_results_df.to_csv(f'{folder_pred}BraTS2023_metrics.csv')