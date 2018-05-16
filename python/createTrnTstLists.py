import os
import ipdb
from sklearn.utils import shuffle


def get_files_in_dirs(root_dirs, ext_to_save):
    file_names = []
    for root_dir in root_dirs:
        for root, dirs, fileNames in os.walk(root_dir):
	    for fileName in fileNames:
                if fileName.find(ext_to_save) > 0:
                    path_to_file = os.path.join(root, fileName)
                    if os.path.exists(path_to_file):
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
	    case_names, num_imgs_in_case = shuffle(case_names, num_imgs_in_case)
	    num_trn_imgs = sum(num_imgs_in_case[0:num_trn_cases])
            num_tst_imgs = sum(num_imgs_in_case[num_trn_cases:])
            ratio = float(num_trn_imgs) / (num_tst_imgs + num_trn_imgs)
	    count += 1

    trn_cases = case_names[0:num_trn_cases]
    tst_cases = case_names[num_trn_cases:]

    return trn_cases, tst_cases

if __name__ == "__main__":
    path_to_trn = "/media/or/1TB-data/train_images_sets_1_3.txt"
    path_to_tst = "/media/or/1TB-data/test_images_sets_1_3.txt"
    case_dir_names = []
    num_frames_in_case = []	
    root_dirs = [ "/media/or/1TB-data/DataSet_3/images", "/media/or/1TB-data/Data_Set_1_new/images/"]
    for root_dir in root_dirs:
        for root, dirs, fileNames in os.walk(root_dir):
            if len(fileNames) > 0:
                if os.path.exists(root):
                   case_dir_names.append(root)
		   num_files_in_dir = 0
                   for fileName in fileNames:
                       if fileName.find("_color.png") > 0:
		          num_files_in_dir += 1
                   num_frames_in_case.append(num_files_in_dir)
		else:
                    print root

    trn_cases_ratio = 0.8
    toKeepImgNum = False
    trn_dirs, tst_dirs = shuffle_list_keep_img_num_in_sets( case_dir_names, num_frames_in_case, trn_cases_ratio, toKeepImgNum)

    print "number of cases all {} ".format(len( case_dir_names))
    print "number of cases train {} ".format(len( trn_dirs))
    print "number of cases test {} ".format(len( tst_dirs))

    trn_list = get_files_in_dirs(trn_dirs, "_color.png")
    write_list_to_file(trn_list, path_to_trn)
    
    tst_list = get_files_in_dirs(tst_dirs, "_color.png")
    write_list_to_file(tst_list, path_to_tst)

    print "number of images train {} ".format(len( trn_list))
    print "number of images test {} ".format(len( tst_list))






