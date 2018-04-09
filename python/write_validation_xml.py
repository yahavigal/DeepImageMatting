import xml.etree.ElementTree as ET
import os
from xml.dom import minidom
import argparse
import ipdb
gt_ext = "_silhuette"

validation_root = '\\\\ger\\ec\\proj\\ha\\RSG\\3D_ValidationVol2\\Nightmare\\Data\\BGS\\DataSet_1_new\\'
def prettify(elem):
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def write_image(root,frame_num,image_path,data_root):

    split = image_path.split(os.sep)
    image_path = validation_root + '\\'.join(split[data_root:])
    frame = ET.SubElement(root,'frame')
    frame.set('number',str(frame_num))

    action = ET.SubElement(frame,'action')
    action.set('id','0')
    action.set('name','bgs')
    input_ = ET.SubElement(action,'input')
    color = ET.SubElement(input_,'image')
    color.set('type','color')
    color.set('src',image_path)
    color.set('id','0')

    depth = ET.SubElement(input_,'image')
    depth.set('type','depth')
    depth.set('src',image_path.replace('color','depth'))
    depth.set('id','1')

    path = os.path.splitext(image_path)
    gt_path = path[0] + gt_ext + path[1]
    gt = ET.SubElement(action,'groundtruth')
    bgs = ET.SubElement(gt,'image')
    bgs.set('src',gt_path)









if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--data_root', type=int, required=False,default=5)
    args = parser.parse_args()

    images = open(args.images_dir).readlines()
    images = [x[0:-1] for x in images if x.endswith('\n')]
    list_images = [x for x in images
                   if x.endswith(".png") and x.find(gt_ext) == -1 or os.path.isdir(x)]

    root = ET.Element('frames')
    root.set('total', str(len(list_images)))
    for i,image_path in enumerate(list_images):
        write_image(root, str(i), image_path, args.data_root)

    with open('bgs_test.xml','w') as f:
        f.write(prettify(root))

