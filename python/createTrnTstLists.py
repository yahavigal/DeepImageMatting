import os
import ipdb
from sklearn.utils import shuffle
import itertools
from sklearn.utils import resample
import re

def get_files_in_dirs(root_dirs, ext_to_save, toDilateVideo = False, videos_ext = 'Video', dillStep = 5):
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
			    if count%dillStep == 0:
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

def add_images_by_number(case_dirs, num_images_to_select):
    num_imgs_per_case = int( round( num_images_to_select/ len(case_dirs)))
    file_names = []
    seed = 6458
    count = 1
    strtInd = 0
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

    toDilateVideo = True
    dillStep = 5
    videos_ext = 'video'

    stillsSetDirMain = "/media/or/1TB-data/Test/"
    videosSetDirMain = "/media/or/1TB-data/Test/temporal_lists/"

    path_to_trn_stills = os.path.join(stillsSetDirMain, "train_all_sets.txt")
    path_to_tst_stills = os.path.join(stillsSetDirMain, "test_all_sets.txt")

    path_to_trn_videos = os.path.join(videosSetDirMain, "train_images_real60_synt10_syntDil3_10_syntDil5_10_syntDil7_10.txt")
    path_to_tst_videos_dirs = os.path.join(videosSetDirMain, "test_real_only_dirs.txt")
    path_to_tst_videos_files = os.path.join(videosSetDirMain, "test_real_only_files.txt")
    
    trn_only_flags = [ False, False, True]
    trn_only_part_stills = 0.2 # part of training only images to add to video set
    trn_only_part_videos = 0.1 # part of training only images to add to video set	
    # !!! Do not forget to add data set 1 renamed for temporal manually
    root_dirs = [ "/media/or/1TB-data/DataSet_1_new/images","/media/or/1TB-data/DataSet_3_new/images", "/media/or/1TB-data/cc_067_no_shifts/DataSet_2_composed/videos"] 

    trn_only_dil_parts = [0.1, 0.1, 0.1]
    root_dirs_dil = [ "/media/or/1TB-data/cc_067_no_shifts_temporal_dil_3/videos", "/media/or/1TB-data/cc_067_no_shifts_temporal_dil_5/videos", "/media/or/1TB-data/cc_067_no_shifts_temporal_dil_7/videos"]

    case_dir_names_stills = []
    num_frames_in_case_stills = []
    case_dir_names_train_only_stills = []
    
    case_dir_names_videos = []
    num_frames_in_case_videos = []
    case_dir_names_train_only_videos = []

    trn_cases_ratio = 0.8
    toKeepImgNum_stills = True
    toKeepImgNum_videos = False

    trn_only_main_dirs_video = []
    for root_dir, trn_only_flag in itertools.izip_longest( root_dirs, trn_only_flags):
        print root_dir, trn_only_flag
	if trn_only_flag and re.search(videos_ext, root_dir, re.IGNORECASE):
	    trn_only_main_dirs_video.append(root_dir)
        for root, dirs, fileNames in os.walk(root_dir):
            if re.search(videos_ext, root, re.IGNORECASE):
                isVideo = True
	    else:
		isVideo = False
            if len(fileNames) > 0:
                if os.path.exists(root):
		    num_files_in_dir = 0
    		    for fileName in fileNames:
                        if fileName.find("_color.png") > 0:
		            num_files_in_dir += 1
		    if trn_only_flag is None or trn_only_flag == False :
                        if isVideo == True:
                            case_dir_names_videos.append(root)
                            num_frames_in_case_videos.append(num_files_in_dir)
                        else:
                            case_dir_names_stills.append(root)     
                            num_frames_in_case_stills.append(num_files_in_dir)
		    else:
                        if isVideo == True:
                            case_dir_names_train_only_videos.append(root)
                        else:
		            case_dir_names_train_only_stills.append(root)
		else:
                    print root

    trn_dirs_stills, tst_dirs_stills, trn_dirs_numImgs_stills, tst_dirs_numImgs_stills = shuffle_list_keep_img_num_in_sets( case_dir_names_stills, num_frames_in_case_stills, trn_cases_ratio, toKeepImgNum_stills)
    trn_dirs_videos, tst_dirs_videos, trn_dirs_numImgs_videos, tst_dirs_numImgs_videos = shuffle_list_keep_img_num_in_sets( case_dir_names_videos, num_frames_in_case_videos, trn_cases_ratio, toKeepImgNum_videos)

    trn_only_list_stills = []
    if len(case_dir_names_train_only_stills) > 0:
	num_trn_imgs_stills = sum(trn_dirs_numImgs_stills)
	num_imgs_to_select_stills = trn_only_part_stills*(num_trn_imgs_stills/(1 - trn_only_part_stills))
        trn_only_list_stills = add_images_by_number(case_dir_names_train_only_stills, num_imgs_to_select_stills)

    trn_only_list_videos = []
    if len(case_dir_names_train_only_videos) > 0:
	num_trn_imgs_videos = sum(trn_dirs_numImgs_videos)
	if len(trn_only_dil_parts) > 0:
	    trn_only_part_final = trn_only_part_videos + sum(trn_only_dil_parts)
	    case_dir_names_train_only_videos = \
		change_cases_to_dilated(trn_only_main_dirs_video, case_dir_names_train_only_videos, trn_only_part_final, trn_only_dil_parts, root_dirs_dil)
	else:
	    trn_only_part_final = trn_only_part
	num_imgs_to_select = trn_only_part_final*(num_trn_imgs_videos/(1 - trn_only_part_final))
        trn_only_list_videos = add_images_by_number(case_dir_names_train_only_videos, num_imgs_to_select)  
	
    # stills
    ipdb.set_trace()    
    trn_dirs_stills = trn_dirs_stills + trn_dirs_videos
    tst_dirs_stills = tst_dirs_stills + tst_dirs_videos
    trn_list = get_files_in_dirs(trn_dirs_stills, "_color.png", toDilateVideo, videos_ext, dillStep)
    trn_list += trn_only_list_stills
    write_list_to_file(trn_list, path_to_trn_stills)
    tst_list = get_files_in_dirs(tst_dirs_stills, "_color.png", toDilateVideo, videos_ext, dillStep)
    write_list_to_file(tst_list, path_to_tst_stills)

    print "number of cases stills {} ".format(len( case_dir_names_stills) + len( case_dir_names_videos))    
    print "number of cases stills train {} ".format(len( trn_dirs_stills))
    print "number of cases stills test {} ".format(len( tst_dirs_stills))
    print "number of cases stills train only {} ".format(len( case_dir_names_train_only_stills))
    print "number of images trn stills {} ".format(len( trn_list))
    print "number of images tst stills {} ".format(len( tst_list))
    print "number of images trn only stills {} ".format(len( trn_only_list_stills))

    # temporal
    trn_list = get_files_in_dirs(trn_dirs_videos, "_color.png", False, videos_ext)
    trn_list += trn_only_list_videos
    write_list_to_file(trn_list, path_to_trn_videos)

    write_list_to_file(tst_dirs_videos, path_to_tst_videos_dirs)
    tst_list = get_files_in_dirs(tst_dirs_videos, "_color.png", False, videos_ext)
    write_list_to_file(tst_list, path_to_tst_videos_files)

    print "number of cases videos {} ".format( len( case_dir_names_videos))    
    print "number of cases videos train {} ".format(len( trn_dirs_videos))
    print "number of cases videos test {} ".format(len( tst_dirs_videos))
    print "number of cases videos train only {} ".format(len( case_dir_names_train_only_videos))
    print "number of images trn videos {} ".format(len( trn_list))
    print "number of images tst videos {} ".format(len( tst_list))
    print "number of images trn only videos {} ".format(len( trn_only_list_videos))
    

  

    
    






