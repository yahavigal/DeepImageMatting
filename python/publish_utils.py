# Import smtplib for the actual sending function
import smtplib
import platform
import shutil
import os
from email.mime.text import MIMEText
import commands
import ipdb

def find_orig_path(publish):
    mount_points = commands.getoutput('mount -v')
    mount_points = mount_points.split('\n')
    orig_path = [x.split(' ')[0:4:2] for x in  mount_points
                 if x.split(' ')[2] in publish and x.split(' ')[4] == 'cifs'][0]
    return orig_path

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
    shutil.copytree(trainer.results_path,dst_path_candidate,
                    ignore = shutil.ignore_patterns('*.caffemodel'))
    trainer.solver.net.save(os.path.join(dst_path_candidate,"final.caffemodel"), False)
    publish_emails(publish,dst_path_candidate)

