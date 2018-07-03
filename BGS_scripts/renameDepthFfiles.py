import os
import sys

import ipdb

if __name__ == "__main__":

    root_dirs = [ "/home/or/BGS_data2/set3/DataSet_3/images/batch4", "/home/or/BGS_data2/set3/DataSet_3/images/batch5", "/home/or/BGS_data2/set3/DataSet_3/videos/batch4", "/home/or/BGS_data2/set3/DataSet_3/videos/batch5"]
    ipdb.set_trace()
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


