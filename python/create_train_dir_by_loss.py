from test_net import *

def find_num_repetitions(indices,val):
    for i in xrange(len(indices)):
        if val >= list(reversed(indices))[i]:
            return i +1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--trimap_dir', type=str, required=False, default=None)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--slice', type=int, required=False, default = 3)

    args = parser.parse_args()

    dict_per_loss =  test_net(args.train_dir, args.net, args.model, args.trimap_dir, save_loss_per_image = True)
    sorted_loss = sorted(dict_per_loss,key=dict_per_loss.get,reverse=True)

    target_file_path = os.path.splitext(args.train_dir)[0]
    target_file_path += "_by_loss"+os.path.splitext(args.train_dir)[1]
    fraction = 1.0/args.slice
    indices = (len(sorted_loss)*np.arange(0.0,1.0,fraction)).astype(np.uint32)
    #ipdb.set_trace()

    with open(target_file_path,'w') as new_list:

        for i,loss in  enumerate(sorted_loss):
            image_path = dict_per_loss[loss]
            num_repetitions =  find_num_repetitions(indices,i)
            for j in xrange(num_repetitions):
                new_list.write(image_path + '\n')





