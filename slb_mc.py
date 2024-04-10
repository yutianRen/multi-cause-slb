import argparse
import collections
import csv
import json
import os
import pickle
import random
import time
import warnings
from datetime import datetime

import xgboost as xgb
#import lightgbm as lgb
import numpy as np
import torch
import yaml
from geomloss import SamplesLoss
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split


from nets.trainer import Trainer


os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    #torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def drift_analysis(d1, d2, loss):
    """
    Source: https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_transport_blur.html#sphx-glr-auto-examples-sinkhorn-multiscale-plot-transport-blur-py
    """
    Loss =  SamplesLoss(loss, p=2, blur=0.05) #sinkhorn is Wasserstein Distance. blur is a paramter
    joint_dist = torch.tensor(d1)
    joint_dist2 = torch.tensor(d2)

    return Loss(joint_dist, joint_dist2).item()


def add_noise(y, rate=0.5, num_classes=8):
    total_num = len(y)
    rand_ind = random.sample(range(total_num), int(total_num*rate))

    for idx in rand_ind:
        rand_c = rng.randint(0, num_classes)
        while rand_c == y[idx]:
            rand_c = rng.randint(0, num_classes)
        y[idx] = rand_c

    return y


def add_itm_noise(gt_it, mean, var):

    it = rng.normal(mean, var)
    sign = 1 if rng.random() >= 0.5 else -1

    new_it = gt_it + it * sign

    return new_it



class Processor:
    """
    It achieves several functions:
    1) load the raw data files generated from the simulation.
    2) train, validate, and test the self-labeling method under the nested cross-validation loop.
    3) generate npy files of training(lb, slb), val, and test set for testing on TorchSSL.

    """
    def __init__(self, args):

        '''
        output dir structure
        --out_dir
            --out_data_dir_name
                --result.txt file
                --npy folder
        '''

        self.rand_seed = args.rand_seed

        data_dir_name = os.path.basename(args.data_path)
        if args.add_data:
            data_dir_name += '_addi_25'
        print('data dirt name: ', data_dir_name)
        args.data_dir_name = data_dir_name
        args.out_dir_path = os.path.join(args.out_dir_path, args.data_dir_name)
        if not os.path.exists(args.out_dir_path):
            os.makedirs(args.out_dir_path)

        result_file_name = 'results_x{}z{}t{}_{}'.format(str(args.x_offset_vel),
                                                         str(args.z_offset_vel),
                                                         '1',
                                                         args.data_dir_name)
        print('use pretrain: ', args.use_pretrain)
        if not args.use_pretrain:
            result_file_name += '__nopretrain'
        if args.add_effect_noise:
            result_file_name += 'effect_noise{}'.format(args.noise_level)
        if args.add_itm_noise:
            print('add itm noise')
            result_file_name += 'itm_noise_t2_{}'.format(args.itm_mean)

        result_file_name += '_seed{}.txt'.format(self.rand_seed)
        args.result_file_path = os.path.join(args.out_dir_path, result_file_name)

        args.work_dir = args.result_file_path[:-4] # os.path.join(args.out_dir_path, result_file_name)

        self.args = args
        print('result file path: ', self.args.result_file_path)

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.args.print_log:
            with open(self.args.result_file_path, 'a') as f:
                print(s, file=f)

    def load_data(self, path, isPlot=False):
        with open(path, 'rb') as f:
            dset = pickle.load(f)
        indices = list(range(len(dset[0])))
        return dset, np.array(indices)

    def esd(self, output_pos_sample):
        x1 = output_pos_sample[0]
        z1 = output_pos_sample[2]
        x2 = output_pos_sample[7]
        z2 = output_pos_sample[9]

        dist = ((x1-x2)**2 + (z1-z2)**2)**(0.5)
        ag = np.arctan2((z2-z1), (x2-x1)) * 180 / np.pi
        th = 7
        label = 0

        if 90 < ag <= 180:
            if dist <= th:
                label = 0
            else:
                label = 1
        elif 0 < ag <= 90:
            if dist <= th:
                label = 2
            else:
                label = 3
        elif -90 < ag <= 0:
            if dist <= th:
                label = 4
            else:
                label = 5
        else:
            if dist <= th:
                label = 6
            else:
                label = 7
        return label

    def cross_validation(self, indices, dataset, retrain_dset, add_dset, add_indices):

        x_vel = self.args.x_offset_vel
        z_vel = self.args.z_offset_vel


        kf = StratifiedKFold(n_splits=3, random_state=self.rand_seed, shuffle=True)
        kf_lb_slb = StratifiedKFold(n_splits=5, random_state=self.rand_seed, shuffle=True)

        input_pos = dataset[0]
        output_pos = dataset[1]
        duration = dataset[2]
        pos_arrs = dataset[3]
        labels = dataset[5]
        print('class dist', collections.Counter(labels))

        input_pos_ret = retrain_dset[0]
        output_pos_ret = retrain_dset[1]
        duration_ret = retrain_dset[2]
        pos_arrs_ret = retrain_dset[3]
        labels_ret = retrain_dset[5]

        pretrain_cls_score = []
        itm_score_1 = []
        itm_mae_1 = []
        itm_score_2 = []
        itm_mae_2 = []
        slb_score = collections.defaultdict(list)
        gt_score = collections.defaultdict(list)

        offset_list = [[], []]
        ball_dist_dev = []

        npy_path = os.path.join(self.args.out_dir_path, f"{self.args.data_dir_name}_npy_seed{self.rand_seed}")

        outer_fold_idx = 0


        for train_index, test_index in kf.split(indices, labels[indices]):

            self.print_log('*'*30 + f'Outer fold {outer_fold_idx} started'+ '*'*30)

            testset_indices = indices[test_index]
            input_pos_test = input_pos_ret[testset_indices]
            output_pos_test = output_pos_ret[testset_indices]
            duration_1_test = duration_ret[testset_indices, 0]
            duration_2_test = duration_ret[testset_indices, 1]
            classes_test = labels_ret[testset_indices]
            print("test set size: ", len(input_pos_test))

            ########################################
            if self.args.save_npy:
                fold_path = os.path.join(npy_path, f'fold_{outer_fold_idx}')
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)
                np.save(os.path.join(fold_path, f'data_test_{outer_fold_idx}.npy'), input_pos_test)
                with open(os.path.join(fold_path, f'label_test_{outer_fold_idx}.pkl'), 'wb') as f:
                    pickle.dump(classes_test, f)
            ########################################

            # 1000
            trainset_indices = indices[train_index]

            inner_fold_idx = 0
            for slb_index, pre_index in kf_lb_slb.split(trainset_indices, labels[trainset_indices]):
                self.print_log('*'*30 + f'Inner fold {inner_fold_idx} started' + '*'*30)

                pre_indices = trainset_indices[pre_index]
                input_pos_pre = input_pos[pre_indices]
                output_pos_pre = output_pos[pre_indices]
                duration_1_pre = duration[pre_indices, 0]
                duration_2_pre = duration[pre_indices, 1]
                classes_pre = labels[pre_indices]
                pos_arrs_pre = pos_arrs[pre_indices]

                # self-label datsaet
                slb_indices = trainset_indices[slb_index]
                input_pos_slb = input_pos_ret[slb_indices]
                output_pos_slb = output_pos_ret[slb_indices]
                duration_1_slb = duration_ret[slb_indices, 0]
                duration_2_slb = duration_ret[slb_indices, 1]
                classes_slb = labels_ret[slb_indices]
                pos_arrs_slb = pos_arrs_ret[slb_indices]


                subset_size = 360
                total_num_ssets = 25

                # process additional dataset
                if self.args.add_data:
                    reordered_add_indices = []
                    to_split_indices = add_indices
                    for _ in range(int(len(to_split_indices) / subset_size) - 1):
                        curr_size = len(to_split_indices)
                        ratio = subset_size / curr_size
                        to_split_indices, selected_add_indices = train_test_split(to_split_indices,
                                                                                  test_size=ratio,
                                                                                  shuffle=True,
                                                                                  random_state=0,
                                                                                  stratify=add_dset[5][to_split_indices])
                        reordered_add_indices += selected_add_indices.tolist()

                    reordered_add_indices += to_split_indices.tolist()

                    input_pos_slb = np.concatenate((input_pos_slb[:subset_size*5],
                                                    add_dset[0][reordered_add_indices],
                                                    input_pos_slb[subset_size * 5:]
                                                    ))
                    output_pos_slb = np.concatenate((output_pos_slb[:subset_size*5],
                                                     add_dset[1][reordered_add_indices],
                                                     output_pos_slb[subset_size * 5:]
                                                     ))
                    duration_1_slb = np.concatenate((duration_1_slb[:subset_size*5],
                                                   add_dset[2][reordered_add_indices, 0],
                                                     duration_1_slb[subset_size * 5:]
                                                   ))
                    duration_2_slb = np.concatenate((duration_2_slb[:subset_size*5],
                                                     add_dset[2][reordered_add_indices, 1],
                                                     duration_2_slb[subset_size * 5:]
                                                     ))
                    pos_arrs_slb = np.concatenate((pos_arrs_slb[:subset_size*5],
                                                   add_dset[3][reordered_add_indices],
                                                   pos_arrs_slb[subset_size * 5:]
                                                   ))
                    classes_slb = np.concatenate((classes_slb[:subset_size*5],
                                                  add_dset[5][reordered_add_indices],
                                                  classes_slb[subset_size * 5:]
                                                  ))

                print('shapes: ', input_pos_slb.shape, output_pos_slb.shape, classes_slb.shape,
                      pos_arrs_slb.shape, output_pos_slb[0, -1])

                if self.args.add_data:
                    input_pos_val = input_pos_slb[subset_size*total_num_ssets:]
                    classes_val = classes_slb[subset_size*total_num_ssets:]
                else:
                    input_pos_val = input_pos_slb[subset_size * 5:]
                    classes_val = classes_slb[subset_size * 5:]
                print('validatoin size: ', input_pos_val.shape, pre_indices.shape, slb_indices.shape)

                if self.args.save_npy:
                    inside_fold_path = os.path.join(fold_path, f'training_fold_{inner_fold_idx}')
                    if not os.path.exists(inside_fold_path):
                        os.makedirs(inside_fold_path)
                    np.save(os.path.join(inside_fold_path, f'data_lb_{inner_fold_idx}.npy'), input_pos_pre)
                    with open(os.path.join(inside_fold_path, f'label_lb_{inner_fold_idx}.pkl'), 'wb') as f:
                        pickle.dump(classes_pre, f)

                    np.save(os.path.join(inside_fold_path, f'data_val_{inner_fold_idx}.npy'), input_pos_val)
                    with open(os.path.join(inside_fold_path, f'label_val_{inner_fold_idx}.pkl'), 'wb') as f:
                        pickle.dump(classes_val, f)

                # interaction time model (ITM) pretraining and testing
                itm_c1 = xgb.XGBRegressor(objective ='reg:squarederror',
                                          learning_rate = 0.1, reg_lambda=40, reg_alpha=1,
                                          n_estimators = 1000, random_state=1, nthread=10)

                print('max duration 1: ', np.max(duration_1_pre))
                itm_c1.fit(output_pos_pre, duration_1_pre)
                itm_train_score = itm_c1.score(output_pos_pre, duration_1_pre)
                itm_train_mae = mean_absolute_error(duration_1_pre, itm_c1.predict(output_pos_pre))
                itm_test_score = itm_c1.score(output_pos_test, duration_1_test)
                itm_test_mae = mean_absolute_error(duration_1_test, itm_c1.predict(output_pos_test))
                print('gbdt train score: ', itm_train_score, 'mae: ',
                      itm_train_mae)
                print('gbdt test score: ', itm_test_score,
                      itm_test_mae)
                print('feature importance: ', itm_c1.feature_importances_)
                itm_mae_1.append(itm_test_mae)
                itm_score_1.append(itm_test_score)


                itm_c2 = xgb.XGBRegressor(objective ='reg:squarederror',
                                          learning_rate = 0.1, reg_lambda=20, reg_alpha=0,
                                          n_estimators = 1000, random_state=1, nthread=10)

                itm_c2.fit(output_pos_pre, duration_2_pre)
                itm_train_score = itm_c2.score(output_pos_pre, duration_2_pre)
                itm_train_mae = mean_absolute_error(duration_2_pre, itm_c2.predict(output_pos_pre))
                itm_test_score = itm_c2.score(output_pos_test, duration_2_test)
                itm_test_mae = mean_absolute_error(duration_2_test, itm_c2.predict(output_pos_test))
                print('gbdt train score: ', itm_train_score, 'mae: ',
                      itm_train_mae)
                print('gbdt test score: ', itm_test_score,
                      itm_test_mae)
                print('feature importance: ', itm_c2.feature_importances_)
                itm_mae_2.append(itm_test_mae)
                itm_score_2.append(itm_test_score)


                # an independent pretrain of the slb classifier to get the pretrain score
                if self.args.use_pretrain:
                    init_seed(self.args.seed)
                    cls_pre = Trainer(self.args, outer_fold_idx, inner_fold_idx, 0, 'pre')
                    cls_pre.load_data(input_pos_pre,
                                      classes_pre,
                                      input_pos_test,
                                      classes_test,
                                      input_pos_val,
                                      classes_val)
                    cls_pre.start()
                    cls_pre_score = cls_pre.best_acc
                else:
                    cls_pre_score = 0

                pretrain_cls_score.append(cls_pre_score)
                print('initial cls score: ', cls_pre_score)


                for slb_i in range(1, total_num_ssets+1):
                    
                    lo, hi = 0, slb_i * subset_size

                    if self.args.save_npy:
                        np.save(os.path.join(inside_fold_path,
                                f'data_slb_{inner_fold_idx}_{slb_i*subset_size}.npy'), input_pos_slb[:hi])
                        with open(os.path.join(inside_fold_path, f'label_slb_{inner_fold_idx}_{slb_i*subset_size}.pkl'), 'wb') as f:
                            pickle.dump(classes_slb[:hi], f)

                    # Conduct self-labeling based on trained ITM and generate self-labeled
                    # data-label pairs.
                    mislabeled = 0
                    self_labeled_input_pos, self_labeled_labl = [], []
                    dist_slb_gt_list = []

                    for di in range(hi):
                        if_label = self.esd(output_pos_slb[di])
                        label = classes_slb[di]
                        if if_label != label:
                            print('infered label unequal')
                        self_labeled_labl.append(if_label)

                    # add noise to label
                    if args.add_effect_noise:
                        self_labeled_labl = add_noise(self_labeled_labl, rate=args.noise_level, num_classes=8)

                    for di in range(hi):

                        ball_1_output_sample = output_pos_slb[di].reshape((1, -1))
                        if args.add_effect_noise:
                            ball_1_output_sample[:, -1] = self_labeled_labl[di]
                        #print('ball 1: ', ball_1_output_sample)
                        it1 = int(itm_c1.predict(ball_1_output_sample)[0])
                        if args.add_itm_noise:
                            it1 = int(add_itm_noise(it1, args.itm_mean, args.itm_mean/2))

                        ball_2_output_sample = output_pos_slb[di].reshape((1, -1))
                        if args.add_effect_noise:
                            ball_2_output_sample[:, -1] = self_labeled_labl[di]
                            #print(ball_2_output_sample, self_labeled_labl[di])
                        it2 = int(itm_c2.predict(ball_2_output_sample)[0])
                        if args.add_itm_noise:
                            it2 = int(add_itm_noise(it2, args.itm_mean, args.itm_mean/2))

                        slb_feature = []
                        ball_id = 0
                        duration_list = [duration_1_slb, duration_2_slb]
                        balls_pos = [(), ()]
                        balls_new_pos = [(), ()]
                        for it in [it1, it2]:
                            duration_slb = duration_list[ball_id]

                            if it > duration_slb[di]:
                                selected_pos = pos_arrs_slb[di, ball_id][0]

                                time_diff_penalty = it - duration_slb[di] #- 1

                                mislabeled += 1

                                x_sign = 1 if rng.random() >= 0.5 else -1
                                z_sign = 1 if rng.random() >= 0.5 else -1
                                x_vel = x_vel # 0.0025
                                z_vel = z_vel # 0.0025
                                x_offset = x_sign * (time_diff_penalty*x_vel)
                                z_offset = z_sign * (time_diff_penalty*z_vel)
                                # print('offset: ', x_offset, z_offset)
                                # print('penalty', time_diff_penalty)
                                balls_pos[ball_id] = (selected_pos[0], selected_pos[2])

                                selected_pos[0] += x_offset
                                selected_pos[2] += z_offset
                                selected_pos[3:6] = [x_vel * (-x_sign), 0, z_vel * (-z_sign)]

                                balls_new_pos[ball_id] = (selected_pos[0], selected_pos[2])

                                offset_list[ball_id].append((x_offset**2+z_offset**2)**0.5)
                            else:
                                selected_pos = pos_arrs_slb[di, ball_id][-it]

                            slb_feature += selected_pos
                            ball_id += 1

                        slb_feature.append(-it1 + it2)

                        x1, y1, z1 = slb_feature[0:3]
                        x2, y2, z2 = slb_feature[6:9] # 6:9 when velocity is added
                        dist = ((x1-x2)**2 + (z1-z2)**2 + (y1-y2)**2)**(0.5)
                        dist_plane = ((x1-x2)**2 + (z1-z2)**2)**(0.5)
                        slb_feature.extend([dist, x1-x2, y1-y2, z1-z2, dist_plane]) #, math.atan2(dist_plane, y2-y1)])

                        # label = classes_slb[di]
                        self_labeled_input_pos.append(slb_feature[12:])
                        # self_labeled_labl.append(label)

                        gt_input_sample = input_pos_slb[di]
                        dist_slb_gt = np.linalg.norm(np.array(slb_feature[12:]) - gt_input_sample)
                        dist_slb_gt_list.append(dist_slb_gt)

                        if balls_pos[0] and balls_pos[1]:
                            x1, z1 = balls_pos[0][0], balls_pos[0][1]
                            x2, z2 = balls_pos[1][0], balls_pos[1][1]
                            old_dist = ((x1-x2)**2 + (z1-z2)**2)**(0.5)

                            x1, z1 = balls_new_pos[0][0], balls_new_pos[0][1]
                            x2, z2 = balls_new_pos[1][0], balls_new_pos[1][1]
                            new_dist = ((x1-x2)**2 + (z1-z2)**2)**(0.5)

                            if old_dist <= 1.1:
                                if old_dist != 0:
                                    ball_dist_dev.append(abs(new_dist-old_dist) / old_dist)
                                    if ball_dist_dev[-1] > 10615086:
                                        print('anomaly: ', ball_dist_dev[-1])
                                        ball_dist_dev.pop()


                    # self-labeling training and test. Combined with pre set or no.
                    init_seed(self.args.seed)
                    cls_slb = Trainer(self.args, outer_fold_idx, inner_fold_idx, slb_i, 'slb')
                    if self.args.use_pretrain:
                        boost_train_set = np.concatenate((input_pos_pre, np.array(self_labeled_input_pos)))
                        boost_train_label = np.concatenate((classes_pre, np.array(self_labeled_labl)))
                    else:
                        boost_train_set = np.array(self_labeled_input_pos)
                        boost_train_label = np.array(self_labeled_labl)

                    cls_slb.load_data(boost_train_set, boost_train_label,
                                      input_pos_test, classes_test,
                                    input_pos_val,
                                    classes_val)
                    cls_slb.start()
                    slb_test_score = cls_slb.best_acc
                    slb_score[slb_i].append(slb_test_score)


                    
                    # # fully supervised training and test.
                    init_seed(self.args.seed)     # re-init the random see
                    cls_fs = Trainer(self.args, outer_fold_idx, inner_fold_idx, slb_i, 'fs')
                    if self.args.use_pretrain:
                        cls_fs.load_data(np.concatenate((input_pos_pre, input_pos_slb[:hi])),
                                       np.concatenate((classes_pre, classes_slb[:hi])),
                                       input_pos_test,
                                       classes_test,
                                       input_pos_val,
                                       classes_val)
                    else:
                        cls_fs.load_data(input_pos_slb[:hi],
                                       classes_slb[:hi],
                                       input_pos_test,
                                       classes_test,
                                       input_pos_val,
                                       classes_val)
                    cls_fs.start()
                    gt_score[slb_i].append(cls_fs.best_acc)
                    # del cls_fs
                    
                    print('$'*60)

                inner_fold_idx += 1
                self.print_log('*'*30 + f'Inner fold {inner_fold_idx-1} finished' + '*'*30)

            outer_fold_idx += 1
            self.print_log('*'*30 + f'Outer fold {outer_fold_idx-1} finished' + '*'*30)


        self.print_log(f'Data path: {self.args.data_path}')
        if pretrain_cls_score:
            self.print_log(f'pretrain_cls_score: {pretrain_cls_score}')
            self.print_log(f'averaged pretrain cls score: {sum(pretrain_cls_score)/len(pretrain_cls_score)}')
            std = np.std(pretrain_cls_score)
            self.print_log(f'std pretrain cls score: {std}')
        self.print_log(f'ITM score 1 list: {itm_score_1}')
        self.print_log(f'ITM score 2 list: {itm_score_2}')
        if itm_score_1:
            self.print_log(f'averaged ITM score: {sum(itm_score_1)/len(itm_score_1)}')
        self.print_log(f'ITM MAE list: {itm_mae_1}')
        if itm_mae_1:
            self.print_log(f'averaged ITM MAE: {sum(itm_mae_1)/len(itm_mae_1)}')
        if itm_score_2:
            self.print_log(f'averaged ITM score: {sum(itm_score_2)/len(itm_score_2)}')
        self.print_log(f'ITM MAE list: {itm_mae_2}')
        if itm_mae_2:
            self.print_log(f'averaged ITM MAE: {sum(itm_mae_2)/len(itm_mae_2)}')
        self.print_log(f'slb score list: {slb_score}')
        self.print_log(f'gt score list: {gt_score}')

        self.print_log(f'slb ave scores:')
        if slb_score:
            for slb_i in range(1, max(slb_score) + 1):
                if slb_score and slb_score[slb_i]:
                    self.print_log(f'{slb_i*100} data: {sum(slb_score[slb_i]) / len(slb_score[slb_i])}')
                    std = np.std(slb_score[slb_i])
                    self.print_log(f'{slb_i * 100} std: {std}')
        self.print_log(f'gt ave scores:')
        if gt_score:
            for slb_i in range(1, max(gt_score) + 1):
                if gt_score and gt_score[slb_i]:
                    self.print_log(f'{slb_i*100} data: {sum(gt_score[slb_i]) / len(gt_score[slb_i])}')
                    std = np.std(gt_score[slb_i])
                    self.print_log(f'{slb_i * 100} std: {std}')

        if offset_list:
            for offset in offset_list:
                self.print_log(f'averaged offset distance: {sum(offset) / len(offset)}')

        if ball_dist_dev:
            self.print_log(f'averaged ball_dist_dev: {sum(ball_dist_dev) / len(ball_dist_dev)}, '
                           f'{sum(ball_dist_dev)}, {len(ball_dist_dev)}')


    def measure_distance(self, d1, d2):
        input_pos = d1[0]
        input_pos_pert = d2[0]
        classes = d1[3].reshape((len(d1[3]), 1))
        classes_pert = d2[3].reshape((len(d2[3]), 1))

        p = np.concatenate((input_pos, classes), axis=1)
        q = np.concatenate((input_pos_pert, classes_pert), axis=1)

        self.print_log(f"Gaussian MMD distance w/ wind: {drift_analysis(p, q, 'gaussian')}")
        self.print_log(f"Energy distance w/ wind: {drift_analysis(p, q, 'energy')}")
        self.print_log(f"Sinkhorn distance w/ wind: {drift_analysis(p, q, 'sinkhorn')}")


    def process(self):
        # load datasets
        program_start_time = datetime.now()
        self.print_log(f"Start Time: {program_start_time}")
        pretrain_dataset, pre_indices = self.load_data(self.args.pretrain_path)
        dataset, indices = self.load_data(self.args.data_path)
        if self.args.add_data:
            add_dset, add_indices = self.load_data(self.args.add_data)

        # init seed and shuffle the data indices
        init_seed(self.args.seed)
        rng.shuffle(indices)
        if self.args.add_data:
            rng_add.shuffle(add_indices)

        # run the training and test in cross validation.
        self.cross_validation(pre_indices, pretrain_dataset, dataset, add_dset, add_indices)

        # measure the distance with the disturbed dataset
        #self.measure_distance(pretrain_dataset, dataset)

        program_end_time = datetime.now()
        self.print_log(f"End Time: {program_end_time}")
        self.print_log(f"Execution Time: {program_end_time - program_start_time}")


if __name__ == '__main__':

    rng = np.random.RandomState(7)
    rng_add = np.random.RandomState(7)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path', type=str, default='', required=True)
    parser.add_argument('--data_path', type=str, default='', required=True)
    parser.add_argument('--out_dir_path', type=str, default='', required=True)
    parser.add_argument('--rand_seed', type=int, default='0', required=True)
    parser.add_argument('--add_data', type=str, default='', required=False)

    parser.add_argument('--use_pretrain', type=int, default=1, required=False)
    parser.add_argument('--x_offset_vel', type=float, default=0.0025, required=False)
    parser.add_argument('--z_offset_vel', type=float, default=0.0025, required=False)

    parser.add_argument('--add_effect_noise', type=int, default=0, required=False)
    parser.add_argument('--noise_level', type=float, default=0.075, required=False)
    parser.add_argument('--add_itm_noise', type=int, default=0, required=False)
    parser.add_argument('--itm_mean', type=float, default=30, required=False)

    parser.add_argument('--save_npy', type=int, default=0, required=False)



    # load MLP network args
    with open('nets/net_config.yaml', 'r') as f:
        net_config = yaml.full_load(f)
    parser.set_defaults(**net_config)
    args = parser.parse_args()

    p = Processor(args)
    p.process()


