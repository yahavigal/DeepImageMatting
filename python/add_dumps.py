import os
import ipdb
from sklearn.utils import shuffle

if __name__ == "__main__":
    ipdb.set_trace()
    dirs_to_add = ["Dumps"]
    root_dirs = [ "/media/or/Data/cc_067_no_shifts/DataSet_2_composed/videos_depthF"]
    for root_dir, dir_to_add in zip(root_dirs, dirs_to_add):
        for root, dirs, fileNames in os.walk(root_dir):
            if len(fileNames) > 0:
                if os.path.exists(root):
		    new_root = os.path.join(root, dir_to_add)
		    if not os.path.exists(new_root):
                        os.makedirs(new_root)
                    for fileName in fileNames:
                       os.rename( os.path.join( root,fileName) , os.path.join( new_root,fileName))

    
