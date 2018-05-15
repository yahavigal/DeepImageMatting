import os
import sys
import json
import random

if __name__ == "__main__":
    path_to_trn = "/media/or/1TB-data/train_images_sets_1_3.txt"
    path_to_tst = "/media/or/1TB-data/test_images_sets_1_3.txt"
    file_names = []
    root_dirs = [ "/media/or/1TB-data/DataSet_3/images", "/media/or/1TB-data/Data_Set_1_new/images/"]
    for root_dir in root_dirs:
        for root, dirs, fileNames in os.walk(root_dir):
            for fileName in fileNames:
                if fileName.find("_color.png") > 0:
                    path_to_file = os.path.join(root, fileName)
                    if os.path.exists(path_to_file):
                        file_names.append(path_to_file)
		    else:
                        print path_to_file

    # random with constant results
    seed_num = 1001
    random.seed(seed_num)
    random.shuffle(file_names)
    # split to train and test
    train_part = 0.85

    num_els_trn = int(len(file_names)*train_part)
    num_els_trn = int(len(file_names)*train_part)

    trn_list = file_names[0:num_els_trn]
    tst_list = file_names[num_els_trn:len(file_names)]

    print len(file_names)
    print len( trn_list)
    print len( tst_list)

    trn_file = open(path_to_trn, 'w')
    for item in trn_list:
        trn_file.write("%s\n" % item)
    tst_file = open(path_to_tst, 'w')
    for item in tst_list:
        tst_file.write("%s\n" % item)





