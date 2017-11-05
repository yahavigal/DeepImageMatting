
import argparse
import os
import ipdb
import re
import shutil

gt_ext = "_silhuette"
adverserial_ext = "_adv"
def save_depth_tree(root,images_input,output,target_ext):
    if not os.path.exists(output):
        os.mkdir(output)
    if os.path.isdir(images_input):
        images_list = [os.path.join(images_input, x)
                                  for x in os.listdir(images_input)
                                  if x.endswith(".png") and x.find(gt_ext) == -1]
    elif os.path.isfile(images_input):
        images = open(images_input).readlines()
        images = [x[0:-1] for x in images if x.endswith('\n')]
        images_list = [x for x in images
                                  if x.endswith(".png") and x.find(gt_ext) == -1 and x.find(adverserial_ext) == -1]

    #main loop
    for image_path in images_list:

        depth_path = image_path.replace("color","depth")
        if not os.path.exists(depth_path):
            continue

        c1 = depth_path.split(os.path.sep)
        c2 = root.split(os.path.sep)
        c3 = [x for x in c1 if x in c2]
        c4 = [x for x in c1 if x not in c2][1:]

        final_path =  os.path.join(os.path.sep.join(c3), root, os.path.sep.join(c4))
        if not os.path.exists(os.path.split(final_path)[0]):
            os.makedirs(os.path.split(final_path)[0])

        frame_num = re.findall(r'\d+',final_path.split(os.path.sep)[-1])[0]
        file_name_dst =  os.path.join(os.path.split(final_path)[0], target_ext + frame_num + ".png")
        shutil.copyfile(depth_path,file_name_dst)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--images_list', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--target_ext', type=str, required=False, default="depth_")
    args = parser.parse_args()

    save_depth_tree(args.root, args.images_list, args.output,args.target_ext)