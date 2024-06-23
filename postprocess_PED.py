import numpy as np
import nibabel as nib
import cc3d
import SimpleITK as sitk
import glob
import os
from tqdm import tqdm

'''
BraTS 2023 PED winner group: CNMC_PMI2023
Postpro: clean small disconnected regions (115 mm3, 0.045 LW_DICE improvement) 
+ ET label redefinition (ET/WT optimal threshold applied)  (0.04, ~0.11 TC LW_DICE improvement) 
+ ED label redefinition (ED/WT optimal threshold applied)  (1, ~0.25 ET LW_DICE improvement) 
Ref: https://github.com/Precision-Medical-Imaging-Group/BraTS2023-inferCode/blob/main/postproc/postprocess.py
'''

def maybe_make_dir(path):
    if (not (os.path.isdir(path))):
        os.makedirs(path)


def read_image(path):
    img_sitk = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img_sitk)
    return img_sitk, img


def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 3
    new_seg[seg == 2] = 1
    return new_seg


def get_ratio_ncr_wt(seg):
    ncr_voxels = np.sum((seg == 1))
    wt_voxels = np.sum(seg != 0)
    if (wt_voxels == 0):
        return 1

    return ncr_voxels / wt_voxels


def get_ratio_ed_wt(seg):
    ed_voxels = np.sum((seg == 2))
    wt_voxels = np.sum(seg != 0)
    if (wt_voxels == 0):
        return 1

    return ed_voxels / wt_voxels


def get_ratio_et_wt(seg: np.ndarray):
    et_voxels = np.sum(seg == 3)
    wt_voxels = np.sum(seg != 0)
    if (wt_voxels == 0):
        return 1

    return et_voxels / wt_voxels


def get_ratio_tc_wt(seg):
    tc_voxels = np.sum((seg == 1) & (seg == 3))
    wt_voxels = np.sum(seg != 0)
    if (wt_voxels == 0):
        return 1

    return tc_voxels / wt_voxels


def convert_et_to_ncr(seg: np.ndarray):
    seg[seg == 3] = 1
    return seg


def convert_ed_to_ncr(seg: np.ndarray):
    seg[seg == 2] = 1
    return seg


def get_greatest_label(seg: np.ndarray) -> str:
    ratios = {
        "ncr": get_ratio_ncr_wt(seg),
        "ed": get_ratio_ed_wt(seg),
        "et": get_ratio_et_wt(seg),
        # "tc": get_ratio_tc_wt(seg),
    }
    greatest_label = max(ratios, key=ratios.get)
    return greatest_label, ratios[greatest_label]


def postprocess_image(seg, label, ratio=0.04):
    if (label == "et"):
        ratio_et_wt = get_ratio_et_wt(seg)
        if (ratio_et_wt < ratio):
            convert_et_to_ncr(seg)
    elif (label == "ed"):
        ratio_ed_wt = get_ratio_ed_wt(seg)
        if (ratio_ed_wt < ratio):
            convert_ed_to_ncr(seg)

    return seg


def save_image(img, img_sitk, out_path):
    new_img_sitk = sitk.GetImageFromArray(img)
    new_img_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(new_img_sitk, out_path)


def postprocess_batch(input_folder, output_folder, label_to_optimize, ratio=0.04, convert_to_brats_labels=False):
    seg_list = sorted(glob.glob(os.path.join(input_folder, "*.nii.gz")))
    for seg_path in tqdm(seg_list):
        seg_sitk, seg = read_image(seg_path)
        if (convert_to_brats_labels):
            seg = convert_labels_back_to_BraTS(seg)
        seg_pp = postprocess_image(seg, label_to_optimize, ratio)
        out_path = os.path.join(output_folder, os.path.basename(seg_path))
        save_image(seg_pp, seg_sitk, out_path)


def get_connected_labels(seg_file):
    seg_obj = nib.load(seg_file)
    seg = seg_obj.get_fdata()
    seg_ncr = np.where(seg == 1, 1, 0)
    seg_ed = np.where(seg == 2, 2, 0)
    seg_et = np.where(seg == 3, 3, 0)
    labels_ncr, n_ncr = cc3d.connected_components(seg_ncr, connectivity=26, return_N=True)
    labels_ed, n_ed = cc3d.connected_components(seg_ed, connectivity=26, return_N=True)
    labels_et, n_et = cc3d.connected_components(seg_et, connectivity=26, return_N=True)
    return labels_ncr, labels_ed, labels_et, n_ncr, n_ed, n_et


def remove_disconnected(seg_file, out_file, thresh=50):
    seg_obj = nib.load(seg_file)
    labels_ncr, labels_ed, labels_et, n_ncr, n_ed, n_et = get_connected_labels(seg_file)

    # process ncr
    vols = []
    for i in range(n_ncr):
        tmp = np.where(labels_ncr == i + 1, 1, 0)
        vol = np.count_nonzero(tmp)
        if vol < thresh:
            labels_ncr = np.where(labels_ncr == i + 1, 0, labels_ncr)
            vols.append(vol)
    # print(f"NCR: removed regions {len(vols)} volumes {vols}")
    removed_ncr = len(vols)
    # process ed
    vols = []
    for i in range(n_ed):
        tmp = np.where(labels_ed == i + 1, 1, 0)
        vol = np.count_nonzero(tmp)
        if vol < thresh:
            labels_ed = np.where(labels_ed == i + 1, 0, labels_ed)
            vols.append(vol)
    # print(f"ED: removed regions {len(vols)} volumes {vols}")
    removed_ed = len(vols)
    # process et
    vols = []
    for i in range(n_et):
        tmp = np.where(labels_et == i + 1, 1, 0)
        vol = np.count_nonzero(tmp)
        if vol < thresh:
            labels_et = np.where(labels_et == i + 1, 0, labels_et)
            vols.append(vol)
    # print(f"ET: removed regions {len(vols)} volumes {vols}")
    removed_et = len(vols)

    new_ncr = np.where(labels_ncr != 0, 1, 0)
    new_ed = np.where(labels_ed != 0, 2, 0)
    new_et = np.where(labels_et != 0, 3, 0)
    new_seg = new_ncr + new_ed + new_et
    new_obj = nib.Nifti1Image(new_seg.astype(np.int8), seg_obj.affine)
    nib.save(new_obj, out_file)
    return removed_ncr, n_ncr, removed_ed, n_ed, removed_et, n_et


def remove_dir(input_dir, output_dir, thresh):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for seg1 in input_dir.iterdir():
        if seg1.name.endswith('.nii.gz'):
            casename = seg1.name
            seg2 = output_dir / casename
            ncr, n_ncr, ed, n_ed, et, n_et = remove_disconnected(seg1, seg2, thresh)
            print(f"{casename} removed regions NCR {ncr:03d}/{n_ncr:03d} "
                  f"ED {ed:03d}/{n_ed:03d} ET {et:03d}/{n_et:03d}")
        else:
            print("Wrong input file!")

    return output_dir

if __name__ == '__main__':
    input_folder  = "/your/nnUNet_results/Dataset140_BraTS2023_PED/nnUNetTrainer_TL_FT_1en4_300epochs__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/BraST_convension/"
    output_folder_base = "/your/nnUNet_results/Dataset140_BraTS2023_PED/nnUNetTrainer_TL_FT_1en4_300epochs__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/Postpro"
    labels_to_optimize = ['ed', 'et']
    ratios = [1.0, 0.04]
    for ratio, label_to_optimize in zip(ratios, labels_to_optimize):
        output_folder = f'{output_folder_base}_{label_to_optimize}'
        print(f'Postpro to {output_folder}')
        maybe_make_dir(output_folder)
        postprocess_batch(input_folder, output_folder, label_to_optimize, ratio=ratio, convert_to_brats_labels=False)

    maybe_make_dir(f'{output_folder_base}_ed_et')
    postprocess_batch(f'{output_folder_base}_ed',  f'{output_folder_base}_ed_et', labels_to_optimize[1], ratio=ratios[1], convert_to_brats_labels=False)