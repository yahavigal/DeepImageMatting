
import time
from watchdog.observers import Observer
from watchdog.observers import polling
from watchdog.events import FileSystemEventHandler
import shutil
import json
import codecs
import os
from collections import OrderedDict
import argparse
import ipdb

class Watcher:

    def __init__(self,path,results_path):
        self.path = path
        self.results_path = results_path
        self.observer = polling.PollingObserver()
        self.should_run = True

    def run(self):
        event_handler = Handler(self)
        self.observer.schedule(event_handler, self.path, recursive=True)
        self.observer.start()
        try:
            while self.should_run == True:
                print 'while loop', self.should_run
                time.sleep(5)
        except Exception, e:
            print 'on expection'
            self.observer.stop()
            self.should_run = False
        self.observer.join()

    def on_stop(self,src_path):
        os.system('cp -r {} {}{}'.format(src_path,self.results_path,os.sep))
        os.system('rm -r {}'.format(src_path))
        print 'cp -r {} {}{}'.format(src_path,self.results_path,os.sep)
        print 'rm -r {}'.format(src_path)
        self.observer.stop()
        self.should_run = False


class Handler(FileSystemEventHandler):


    def __init__(self,observer):
        self.observer = observer

    def on_any_event(self,event):
        if event.is_directory:
            if 'bgs_model_request_0_output' ==  os.path.split(event.src_path)[1]:
                print event.src_path, event.event_type
                self.observer.on_stop(event.src_path)


def change_bgs_json(json_path,weights_path,model_path):
    with  codecs.open(json_path,'r+',encoding='utf-8') as f:
        data = json.load(f,object_pairs_hook=OrderedDict)
        print os.path.split(model_path)[1]
        print os.path.split(weights_path)[1]
        data['rgbProcessors']['RGBProcessorCNN']['modelFiles']['windowsModelFile'] = os.path.split(model_path)[1]
        data['rgbProcessors']['RGBProcessorCNN']['modelFiles']['trainingFile'] = os.path.split(weights_path)[1]
    return data

def get_latest_json():
    cur_dir = os.getcwd()
    os.chdir('/home/or/ir-face-authentication')
    os.system('git pull origin master')
    print 'cp Configuration/bgs/bgsConfig.json {}'.format(cur_dir)
    os.system('cp Configuration/bgs/bgsConfig.json {}'.format(cur_dir))
    os.chdir(cur_dir)


def change_json(json_path,weights_path,model_path):
    dirname = get_benchmark_dir()
    with  codecs.open(json_path,'r+',encoding='utf-8') as f:
        data = json.load(f)
        data['tag'] = 'output'
        #data['files_to_replace'] = []
        print os.path.split(model_path)[1]
        print os.path.split(weights_path)[1]
        data['files_to_replace'] = ['bgsConfig.json']
        data['files_to_replace'].append(os.path.split(weights_path)[1])
        data['files_to_replace'].append(os.path.split(model_path)[1])
        data['params_general']['results_path'] = '\\\\percdsk510\\shared_input\\'
    return data

def get_benchmark_dir():
    ind = 0
    dirname = '/home/or/benchmark_request/bgs_model_request_{}'.format(ind)
    while os.path.exists(dirname):
        ind+=1
        dirname = '/home/or/benchmark_request/bgs_model_request_{}'.format(ind)
    return dirname

def make_benchmark_dir(weights_path,model_path,json_data,bgs_json_data):
    dirname = get_benchmark_dir()
    dirname_tmp = os.path.join(os.path.split(dirname)[0] ,'tmp_' + os.path.split(dirname)[1])
    os.mkdir(dirname_tmp)
    shutil.copyfile(weights_path,os.path.join(dirname_tmp,os.path.split(weights_path)[1]))
    shutil.copyfile(model_path,os.path.join(dirname_tmp,os.path.split(model_path)[1]))
    with open(os.path.join(dirname_tmp,'request.json'),'w') as f:
        json.dump(json_data,f,indent=4)
    with open(os.path.join(dirname_tmp,'bgsConfig.json'),'w') as f:
        json.dump(bgs_json_data,f,indent=4)
    shutil.move(dirname_tmp,dirname)
    return dirname

def trigger_benchmark(weights_path,model_path):
    get_latest_json()
    json_data = change_json('request.json',weights_path,model_path)
    bgs_json_data = change_bgs_json('bgsConfig.json',weights_path,model_path)
    dirname = make_benchmark_dir(weights_path,model_path,json_data,bgs_json_data)
    results_path = os.path.split(weights_path)[0]
    watch = Watcher('/home/or/benchmark_request',results_path)
    watch.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True, help='model definition in prototxt file')
    parser.add_argument('--model', type=str, required=True, help='path to weights file')
    args = parser.parse_args()
    trigger_benchmark(args.model,args.proto)


