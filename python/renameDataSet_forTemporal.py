import os
import ipdb
from shutil import copyfile
import re

def get_files_in_dir( root, fileNames, ext_to_save, toDilateVideo = False, videos_ext = 'Video'):
    file_names = []   
    if re.search(videos_ext, root, re.IGNORECASE):
        isVideo = True
    else:
	isVideo = False
    count = 0
    for fileName in fileNames:
        if fileName.find(ext_to_save) > 0:
	    count += 1
	    if isVideo and toDilateVideo:
		if count%5 == 0:
                    file_names.append(fileName)
            else:
                file_names.append(fileName)

    file_names_sorted = sorted(file_names, key=lambda item: (int(item.partition('_')[0]) ) )
    return file_names_sorted
   
def replace_frameNum_in_image_path(image_path, frame_num_old, frame_num_new):        
    spl = os.path.split(image_path)
    new_name = spl[1].replace(frame_num_old, frame_num_new)
    new_path = os.path.join(spl[0], new_name)
    return new_path

def copy_and_rename_files( count, root, clr_name, clr_ext, gt_ext, depth_ext, src_main_dir, dst_main_dir, main_ext, depth_filled_ext):    
    path_2_clr = os.path.join(root, clr_name);
    path_2_gt =  path_2_clr.replace(clr_ext, gt_ext)
    path_2_depth =  path_2_clr.replace(clr_ext, depth_ext)
    path_2_depth_filled =  path_2_clr.replace( main_ext, main_ext + depth_filled_ext)
    path_2_depth_filled =  path_2_depth_filled.replace( clr_ext, depth_ext)

    frame_num = clr_name.partition('_')[0]
    path_2_clr_new = path_2_clr.replace(src_main_dir, dst_main_dir);
    path_2_gt_new =  path_2_gt.replace(src_main_dir, dst_main_dir)
    path_2_depth_new =  path_2_depth.replace(src_main_dir, dst_main_dir)
    path_2_depth_filled_new =  path_2_depth_filled.replace( src_main_dir, dst_main_dir)

    count_str = str(count)
    path_2_clr_new = replace_frameNum_in_image_path(path_2_clr_new, frame_num, count_str)
    path_2_gt_new =  replace_frameNum_in_image_path(path_2_gt_new, frame_num, count_str)
    path_2_depth_new = replace_frameNum_in_image_path(path_2_depth_new, frame_num, count_str)
    path_2_depth_filled_new = replace_frameNum_in_image_path(path_2_depth_filled_new, frame_num, count_str);

    dir_images = os.path.dirname(path_2_clr_new)
    if not os.path.exists(dir_images):
        os.makedirs(dir_images)

    dir_depth_filled = os.path.dirname(path_2_depth_filled_new)
    if not os.path.exists(dir_depth_filled):
        os.makedirs(dir_depth_filled)

    copyfile(path_2_clr, path_2_clr_new)
    copyfile(path_2_gt, path_2_gt_new)
    copyfile(path_2_depth, path_2_depth_new)
    copyfile(path_2_depth_filled, path_2_depth_filled_new)    


if __name__ == "__main__":
    toDilateVideo = False
    videos_ext = 'Video'

    path_to_src = "/media/or/1TB-data/DataSet_1_new/images"
    main_ext = "images"
    
    depth_filled_ext = "_depthF"    

    src_main_dir = "DataSet_1_new"
    dst_main_dir = "DataSet_1_new_renamed_temporal"

    clr_ext = "_color.png"
    gt_ext = "_color_silhuette.png"
    depth_ext = "_depth.png"
    
    for root, dirs, fileNames in os.walk(path_to_src):
        if len(fileNames) > 0:
	    clr_names_sorted = get_files_in_dir( root, fileNames, clr_ext, toDilateVideo, videos_ext)
	    count = 0;
	    for clr_name in clr_names_sorted:
		count = count + 1
		copy_and_rename_files( count, root, clr_name, clr_ext, gt_ext, depth_ext, src_main_dir, dst_main_dir, main_ext, depth_filled_ext)
            



    
    


