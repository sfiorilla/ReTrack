import os
import shutil

import argparse
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
'''
python generate_mot_patches_adapted.py --data_path ./MOT17/ --save_path ./MOT17/ --include_only_categories_list 1 --include_only_marked_one --min_conf_prob 0.0
'''

# For argparse, define a custom argument type for a list of integers. 
# See: https://www.geeksforgeeks.org/how-to-pass-a-list-as-a-command-line-argument-with-argparse/
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def generate_trajectories(file_path, groundTrues, args):
    f = open(file_path, 'r')

    lines = f.read().split('\n')
    values = []
    for l in lines:
        split = l.split(',')
        if len(split) < 2:
            break
        numbers = [float(i) for i in split]

        if args.include_only_marked_one and int(numbers[6]) != 1:
            continue

        if int(numbers[7]) not in args.include_only_categories_list:
            continue

        if numbers[8] < args.min_conf_prob:
            continue


        values.append(numbers)
        

    values = np.array(values, np.float_)

    if groundTrues:
        # values = values[values[:, 6] == 1, :]  # Remove ignore objects
        # values = values[values[:, 7] == 1, :]  # Pedestrian only
        values = values[values[:, 8] > 0.4, :]  # visibility only

    values = np.array(values)
    values[:, 4] += values[:, 2]
    values[:, 5] += values[:, 3]

    return values


def make_parser():
    parser = argparse.ArgumentParser("MOTChallenge ReID dataset")

    # NOTE: Both path must finish with a slash, e.g. .my/mot/dataset/
    parser.add_argument("--data_path", default="./DanceTrack", help="path to MOT-type data, must finish with a /")
    parser.add_argument("--save_path", default="./mot-reid_datasets/DanceTrack", help="Path to save the MOT-ReID dataset, must finish with a /")

    parser.add_argument('--include_only_categories_list', type=list_of_ints, default=1)
    parser.add_argument('--include_only_marked_one', action="store_true") # NOTE: This one is stringer than --include_only_categories_list
    parser.add_argument('--min_conf_prob', default=0.0, type=float)


    return parser


def split_data(args):
    # Create folder for outputs
    save_path = args.save_path[:-1] + "_ReID"

    # Adjust the save folder name
    #if args.include_only_marked_one:
    #    save_path = save_path + "__only_1s"
    #else:
    #    save_path = save_path + "__0s_and_1s"

    #save_path = save_path + "__categ"
    #for c in args.include_only_categories_list:
    #    save_path = save_path + "_" + str(c)

    #save_path = save_path + "__min_conf_" + str(args.min_conf_prob)
    
    shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path, exist_ok=True)
    
    dataset_name = args.data_path.split('/')[-2]

    train_save_path = os.path.join(save_path, 'bounding_box_train')
    os.makedirs(train_save_path, exist_ok=True)
    test_save_path = os.path.join(save_path, 'bounding_box_test')
    os.makedirs(test_save_path, exist_ok=True)
    
    # Get gt data
    data_path = os.path.join(args.data_path, 'train')

    if dataset_name == 'MOT17':
        seqs = [f for f in os.listdir(data_path) if 'FRCNN' in f]
    else:
        seqs = os.listdir(data_path)

    seqs.sort()

    id_offset = 0

    for seq in seqs:
        print(seq)
        print(id_offset)

        ground_truth_path = os.path.join(data_path, seq, 'gt/gt.txt')
        gt = generate_trajectories(ground_truth_path, groundTrues=True, args=args)  # f, id, x_tl, y_tl, x_br, y_br, ...

        images_path = os.path.join(data_path, seq, 'img1')
        img_files = os.listdir(images_path)
        img_files.sort()

        num_frames = len(img_files)
        max_id_per_seq = 0
        for f in range(num_frames):

            img = cv2.imread(os.path.join(images_path, img_files[f]))
            if img is None:
                print("ERROR: Receive empty frame ({})".format(os.path.join(images_path, img_files[f])))
                continue

            H, W, _ = np.shape(img)
            det = gt[f + 1 == gt[:, 0], 1:].astype(np.int_)

            for d in range(np.size(det, 0)):
                id_ = det[d, 0]
                x1 = det[d, 1]
                y1 = det[d, 2]
                x2 = det[d, 3]
                y2 = det[d, 4]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, W)
                y2 = min(y2, H)

                # patch = cv2.cvtColor(img[y1:y2, x1:x2, :], cv2.COLOR_BGR2RGB)
                patch = img[y1:y2, x1:x2, :]

                max_id_per_seq = max(max_id_per_seq, id_)

                # plt.figure()
                # plt.imshow(patch)
                # plt.show()

                fileName = (str(id_ + id_offset)).zfill(7) + '_' + seq + '_' + (str(f)).zfill(7) + '_acc_data.jpg'

                if f < num_frames // 2:
                    cv2.imwrite(os.path.join(train_save_path, fileName), patch)
                else:
                    cv2.imwrite(os.path.join(test_save_path, fileName), patch)

        id_offset += max_id_per_seq

        return test_save_path


import os
import numpy as np
import shutil
np.random.seed(42)

def create_query_set(test_path="./MOT17_ReID/bounding_box_test/" ):

    imgs = os.listdir(test_path)
    # Group images by person ID
    id_to_imgs = {}
    for img in imgs:
        pid = int(img.split("_")[0])
        if pid not in id_to_imgs.keys():
            id_to_imgs[pid] = []
        id_to_imgs[pid].append(img)

    ##Count stats
    unique_ids = list(id_to_imgs.keys())
    ids_size=len(unique_ids)
    freqs = [len(id_to_imgs[pid]) for pid in unique_ids]
    w_probs = np.array(freqs) / sum(freqs)
    
    # Sample person IDs
    sampled_ids = np.random.choice(unique_ids, p=w_probs, size=(ids_size//2), replace=False)
    
    # For each sampled_id, sample the 0.1% of their image in the testset
    query_sampled_names=[]
    for pid in sampled_ids:
        pid_images = id_to_imgs[pid]
        samples_no = max(1, int(0.1*len(pid_images)))
        names = np.random.choice(pid_images, samples_no, replace=False)
        query_sampled_names.extend(names)
        
    #Move files in the folder
    query_set_path = "./MOT17_ReID/query_set"
    os.makedirs(query_set_path, exist_ok=True)

    for img in query_sampled_names:
        source = os.path.join(test_path, img)
        destination = os.path.join(query_set_path, img)
        shutil.move(source,destination) 




if __name__ == "__main__":
    args = make_parser().parse_args()

    print("Generating Training and Test...")
    test_path = split_data(args)
    print("Generating Query sets...")
    create_query_set(test_path)
    print("Process ended successfully.")
