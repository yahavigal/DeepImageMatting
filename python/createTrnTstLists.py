import os
import ipdb
from sklearn.utils import shuffle
import itertools
from sklearn.utils import resample
import re

def get_files_in_dirs(root_dirs, ext_to_save, toDilateVideo = False, videos_ext = 'Video'):
    file_names = []   
    for root_dir in root_dirs:
        for root, dirs, fileNames in os.walk(root_dir):
            if re.search(videos_ext, root, re.IGNORECASE):
                isVideo = True
	    else:
		isVideo = False
            count = 0
	    for fileName in fileNames:
                if fileName.find(ext_to_save) > 0:
		    count += 1
                    path_to_file = os.path.join(root, fileName)
                    if os.path.exists(path_to_file):
			if isVideo and toDilateVideo:
			    if count%5 == 0:
                                file_names.append(path_to_file)
                        else:
                            file_names.append(path_to_file)
		    else:
                        print path_to_file
    return file_names

def write_list_to_file(file_list, path_to_file):
    with open(path_to_file, 'w') as f:
        f.writelines("%s\n" % l for l in file_list)

def shuffle_list_keep_img_num_in_sets( case_names, num_imgs_in_case, trn_cases_ratio, toKeepImgNum = True):
    seed=1001
    case_names, num_imgs_in_case = shuffle(case_names, num_imgs_in_case, random_state=seed)
    num_trn_cases = int( trn_cases_ratio* len(case_names))

    if 	toKeepImgNum:
        num_trials = 3
        count = 0
        num_trn_imgs = sum(num_imgs_in_case[0:num_trn_cases])
        num_tst_imgs = sum(num_imgs_in_case[num_trn_cases:])
        ratio = float(num_trn_imgs) / (num_tst_imgs + num_trn_imgs)
	while count < num_trials and abs(ratio - trn_cases_ratio) > 0.05:
	    case_names, num_imgs_in_case = shuffle(case_names, num_imgs_in_case, random_state=(seed+count*100))
	    num_trn_imgs = sum(num_imgs_in_case[0:num_trn_cases])
            num_tst_imgs = sum(num_imgs_in_case[num_trn_cases:])
            ratio = float(num_trn_imgs) / (num_tst_imgs + num_trn_imgs)
	    count += 1
    trn_cases = case_names[0:num_trn_cases]
    tst_cases = case_names[num_trn_cases:]
    trn_cases_numImgs = num_imgs_in_case[0:num_trn_cases]
    tst_cases_numImgs = num_imgs_in_case[num_trn_cases:]

    return trn_cases, tst_cases, trn_cases_numImgs, tst_cases_numImgs

def add_only_trn_images(case_dirs, num_images_to_select, is4temporal):
    num_imgs_per_case = int( round( num_images_to_select/ len(case_dirs)))
    file_names = []
    seed = 6458
    count = 1
    strtInd = 1
    if is4temporal:
        strtInd = 0 # 4
    for case in case_dirs:
	file_list = get_files_in_dirs([case], "_color.png")
        samples = min(num_imgs_per_case, len(file_list) )
	if samples < num_imgs_per_case:
	    print "number of images in case {} less then required {} {}".format(case, samples, num_imgs_per_case)
	    
	inds2take = resample(range(strtInd, len(file_list)), n_samples=samples, random_state=seed*count)
        count += 1
	for ind in inds2take:
	    file_names.append(file_list[ind])
    
    return file_names

def change_cases_to_dilated(trn_only_main_dirs, case_dirs, trn_only_part_final, trn_only_dil_parts, root_dirs_dil):
    if len(trn_only_main_dirs) >1:
	error("recoding is needed, only one dir is supported now")

    seed=1111
    case_dirs_new = shuffle(case_dirs, random_state=seed)

    parts_to_change = [x / trn_only_part_final for x in trn_only_dil_parts]

    num_cases = len(case_dirs_new)
    num_cases_to_change = [int(x * num_cases) for x in parts_to_change]

    start_ind = 0
    main_dir = trn_only_main_dirs[0]
    for ind in range(0, len(num_cases_to_change)):
	nc = num_cases_to_change[ind]
	for case in range (start_ind, start_ind + nc):
	    case_dirs_new[case]  = case_dirs_new[case].replace(main_dir, root_dirs_dil[ind])

	start_ind = start_ind + nc

    return case_dirs_new

if __name__ == "__main__":

    toDilateVideo = False
    videos_ext = 'video'

    is4temporal = True
    path_to_trn = "/media/or/1TB-data/Sets4multipleDataSets/temporal_lists/train_images_real60_synt20_syntDil3_20_syntDil5_20_syntDil7_20.txt"
    path_to_tst = "/media/or/1TB-data/Sets4multipleDataSets/temporal_lists/test_real_only_v3.txt"
    
    trn_only_flags = [ False, True]
    trn_only_part = 0.2 # part of training only images to agg	
    root_dirs = [ "/media/or/1TB-data/DataSet_3_new_wo_b8/images", "/media/or/1TB-data/cc_067_no_shifts/DataSet_2_composed/videos"]

    trn_only_dil_parts = [0.2, 0.2, 0.2]
    root_dirs_dil = [ "/media/or/1TB-data/cc_067_no_shifts_temporal_dil_3/videos", "/media/or/1TB-data/cc_067_no_shifts_temporal_dil_5/videos", "/media/or/1TB-data/cc_067_no_shifts_temporal_dil_7/videos"]

    case_dir_names = []
    num_frames_in_case = []
    case_dir_names_train_only = []

    trn_cases_ratio = 0.8
    toKeepImgNum = False

    trn_only_main_dirs = []
    for root_dir, trn_only_flag in itertools.izip_longest( root_dirs, trn_only_flags):
        print root_dir, trn_only_flag
	if trn_only_flag:
	    trn_only_main_dirs.append(root_dir)
        for root, dirs, fileNames in os.walk(root_dir):
            if len(fileNames) > 0 and re.search(videos_ext, root, re.IGNORECASE):
                if os.path.exists(root):
		    num_files_in_dir = 0
    		    for fileName in fileNames:
                        if fileName.find("_color.png") > 0:
		            num_files_in_dir += 1
		    if trn_only_flag is None or trn_only_flag == False :
                        case_dir_names.append(root)     
                        num_frames_in_case.append(num_files_in_dir)
		    else:
		        case_dir_names_train_only.append(root)
		else:
                    print root

    trn_dirs, tst_dirs, trn_dirs_numImgs, tst_dirs_numImgs = shuffle_list_keep_img_num_in_sets( case_dir_names, num_frames_in_case, trn_cases_ratio, toKeepImgNum)
    trn_only_list = []
    if len(case_dir_names_train_only) > 0:
	num_trn_imgs = sum(trn_dirs_numImgs)
	if len(trn_only_dil_parts) > 0:
	    trn_only_part_final = trn_only_part + sum(trn_only_dil_parts)
	    case_dir_names_train_only = \
		change_cases_to_dilated(trn_only_main_dirs, case_dir_names_train_only, trn_only_part_final, trn_only_dil_parts, root_dirs_dil)
	else:
	    trn_only_part_final = trn_only_part
	num_imgs_to_select = trn_only_part_final*(num_trn_imgs/(1 - trn_only_part_final))
        trn_only_list = add_only_trn_images(case_dir_names_train_only, num_imgs_to_select, is4temporal)

	

    print "number of cases all real {} ".format(len( case_dir_names))
    print "number of cases train {} ".format(len( trn_dirs))
    print "number of cases test {} ".format(len( tst_dirs))


    print "number of cases trn only {} ".format(len(case_dir_names_train_only))
    print "number of images trn only {} ".format(len( trn_only_list))
    
    trn_list = get_files_in_dirs(trn_dirs, "_color.png", toDilateVideo, videos_ext)
    trn_list += trn_only_list
    write_list_to_file(trn_list, path_to_trn)

    print "number of images train {} ".format(len( trn_list))
    
    if is4temporal:
        write_list_to_file(tst_dirs, path_to_tst)
	print "number of cases test {} ".format(len( tst_dirs))
    else:
        tst_list = get_files_in_dirs(tst_dirs, "_color.png", toDilateVideo, videos_ext)
        write_list_to_file(tst_list, path_to_tst)
        print "number of images test {} ".format(len( tst_list))        

    
    






