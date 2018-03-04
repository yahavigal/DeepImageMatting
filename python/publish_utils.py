# Import smtplib for the actual sending function
import smtplib
import platform
import shutil
import os
from email.mime.text import MIMEText
import commands
import ipdb
import csv

def replace_names_in_log_file(publish,dst_path,results_path):

    orig_path = find_orig_path(publish)
    dst_path = dst_path.replace(orig_path[1], orig_path[0])
    dst_path_orig = dst_path.replace(os.sep,'\\')

    with open(os.path.join(results_path,'test_log_file.txt'),'rb') as csvfile:
        with open(os.path.join(results_path,'test_log_file_links.csv'),'w') as csvtarget:
            reader = csv.DictReader(csvfile, delimiter=' ')
            writer = csv.DictWriter(csvtarget,delimiter=' ',fieldnames= reader.fieldnames)
            images_in_dir = [x for x in os.listdir(results_path) if 'mask.png' in x]
            for row in reader:
                path = row['image_path'].replace(os.sep,'_')
                match = [x for x in images_in_dir if x[0:x.index('iou') - 1] in path]
                if len(match) == 0:
                    continue
                match = match[0]
                current_path =   os.path.join(dst_path_orig,match)
                current_path = current_path.replace(os.sep,'\\')
                row['image_path'] = current_path
                writer.writerow(row)



def find_orig_path(publish):
    mount_points = commands.getoutput('mount -v')
    mount_points = mount_points.split('\n')
    orig_path = [x.split(' ')[0:4:2] for x in  mount_points
                 if x.split(' ')[2] in publish and x.split(' ')[4] == 'cifs'][0]
    return orig_path

def create_dlc_file(deploy_path,weights_path,out_file):
    deploy_path = os.path.abspath(deploy_path)
    weights_path = os.path.abspath(weights_path)
    cur_dir = os.getcwd()
    os.chdir('../BGS_scripts')
    dlc_command = 'bash convert_to_dlc.sh {} {} {}'.format(deploy_path,
                                                           weights_path,
                                                           os.path.abspath(out_file))
    sudo_pswd= '123'
    os.system('echo %s|sudo -S %s' % (sudo_pswd,dlc_command))
    os.chdir(cur_dir)

def publish_emails(publish,dst_path):

    orig_path = find_orig_path(publish)
    dst_path = dst_path.replace(orig_path[1], orig_path[0])
    dst_path = dst_path.replace(os.sep,'\\')
    msg = MIMEText('please take a look at {}'.format(dst_path))

    msg['Subject'] = 'the results from train env. has been published'

    me = 'BGS.deep_learning@intel.com'
    omer = 'omer.achrack@intel.com'
    david = 'david.stanhill@intel.com'
    eyal = 'eyal.rond@intel.com'
    alexandra = 'alexandra.manevitch@intel.com'
    tamir = 'tamir.einy@intel.com'
    yahav = 'yahav.avigal@intel.com'
    ofir = 'ofir.levy@intel.com'

    list_to_send = [alexandra,david,omer,eyal,tamir,yahav,ofir]
    msg['From'] = me
    msg['To'] = ', '.join(list_to_send)

    s = smtplib.SMTP()
    s.connect('smtp.intel.com', 25)
    s.sendmail(me,list_to_send, msg.as_string())
    s.quit()

def publish_results(publish,trainer):
    dst_path = trainer.exp_name.replace(' ','_')
    dst_path_candidate = dst_path + "_0"
    dst_path_candidate = os.path.join(publish,dst_path_candidate)
    while os.path.exists(dst_path_candidate):
        exp_num = int(dst_path_candidate.split('_')[-1])
        dst_path_candidate ='{}_{}'.format(dst_path,exp_num+1)
        dst_path_candidate = os.path.join(publish,dst_path_candidate)
    replace_names_in_log_file(publish,dst_path_candidate,trainer.results_path)
    shutil.copytree(trainer.results_path,dst_path_candidate,
                    ignore = shutil.ignore_patterns('*.caffemodel'))
    trainer.solver.net.save(os.path.join(dst_path_candidate,"final.caffemodel"), False)
    create_dlc_file(os.path.join(trainer.results_path,trainer.deploy_file),
                    os.path.join(dst_path_candidate,"final.caffemodel"),
                    os.path.join(dst_path_candidate,trainer.deploy_file.replace('prototxt','dlc')))
    publish_emails(publish,dst_path_candidate)

