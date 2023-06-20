import torch
from .LoadData import *
from .ubr import UBR_SA

from .MB4CTR import MB4CTR
import numpy as np
from datetime import datetime
from torch.autograd import Variable
from .utils.evaluation import evaluate
from .preprocess import Config
from sklearn.metrics import roc_auc_score, log_loss


def to_var(x, device, volatile=False):
    if device != 'cpu':
    # if torch.cuda.is_available():
        x = x.to(device)
    return Variable(x, volatile=volatile)


def limit(x):
    return (x - (torch.min(x))) / (torch.max(x) - torch.min(x)) + 1e-7

def train(args):
    # Hyper Parameters
    device = args.device
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    negative_sample_size = args.negative_sample_size
    n_sample = args.n_sample
    rank_task_weight = 0.9

    print("batch_size: ", batch_size)
    print("learning_rate: ", learning_rate)
    print("negative_sample_size: ", negative_sample_size)
    data_name = args.dataset

    # in ['sku', 'bh', 'cid3', 'gap', 'dwell']
    micro_item_list = Config.get_micro_item_list(args.micro_mode)

    print("Loading data...", data_name)
    print("Micro_item_list: ", micro_item_list)

    if data_name == 'Computers':
        num_users = 60242
        num_items = 62006
        f_number = 5
        train_dataset, test_dataset, test_label = LoadData.get_data(data_name)
    elif data_name == 'Applicances':
        num_users = 86549
        num_items = 59999
        f_number = 5
        train_dataset, test_dataset, test_label = LoadData.get_data(data_name)
    elif data_name == 'UserBehaviors':
        num_users = 117362
        num_items = 2579724
        f_number = 4
        train_dataset, test_dataset, test_label = LoadData.get_data(data_name)
        print("train/test/: {:d}/{:d}".format(len(train_dataset), len(test_dataset)))
    print("==================================================================================")
    # return train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset


    # train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset = process_data(data_name)

    # ubr = UBR_SA(num_items - 30, 1590, 768, f_number, 'word2Vec')
    mb4ctr = MB4CTR(data_name, num_items - 30, negative_sample_size, device)

    if device != 'cpu':
        mb4ctr.to(device)

    optimizer = torch.optim.Adam(mb4ctr.parameters(), lr=learning_rate)

    print("==================================================================================")
    print("Training Start..", datetime.now())

    for epoch in range(num_epochs):
        for idx, (user_id, macro, micro) in enumerate(train_dataset):
            micro = to_var(micro, device)
            macro = to_var(macro, device)
            user_id = to_var(user_id, device)
            rank_pos_outputs = mb4ctr(user_id, macro, micro)

            # Calculate AUC
            auc = roc_auc_score(test_label, rank_pos_outputs)

            # Calculate log loss
            logloss = log_loss(test_label, rank_pos_outputs)
            optimizer.zero_grad()
            logloss.backward()
            optimizer.step()
            if idx % 100 == 0:
                print(" loss: {}, time: {}".format(logloss, datetime.now()))
        print("epoch: ", epoch, datetime.now())
        print("==================================================================================")
        print("Begin Ranking Prediction..")
        with torch.no_grad():
            k = [5, 10, 20]
            precision = [0.0] * len(k)
            mrr = [0.0] * len(k)
            recall = [0.0] * len(k)  # H@N
            apks = 0.0
            precision_u = [[], [], []]
            recall_u = [[], [], []]
            for i, user in enumerate(range(num_users)):
                true_results = test_dataset.get_user_items(user)
                user_torch = to_var(torch.LongTensor([user + 1]), device)
                user_emb = mb4ctr.user_embedding(user_torch)
                user_emb = user_emb.squeeze()
                user_lv = mb4ctr.user_latent_vector(user_torch).squeeze()
                user_emb = torch.cat((user_emb, user_lv))
                 pred_long_term_results = user_emb
                item_bias = mb4ctr.item_bias(to_var(torch.LongTensor(range(num_items + 1)), device)).squeeze()
                trans_item_bias = torch.nn.Linear(num_items + 1, 1)
                if device != 'cpu':
                    trans_item_bias = trans_item_bias.to(device)
                pred_results = pred_long_term_results + mb4ctr.user_bias(user_torch).squeeze() + trans_item_bias(
                    item_bias)

                # remove visited items
                visited_items = train_dataset.get_user_items(user)
                pred_results[visited_items] = -np.inf
                pred_results[num_items] = -np.inf
                sub_precision, sub_recall, apk, sub_precision_u, sub_recall_u, ndcg, sub_mrr = evaluate(true_results,
                                                                                                        pred_results, ks=k)
                precision = [precision[i] + sub_precision[i] for i in range(len(k))]
                recall = [recall[i] + sub_recall[i] for i in range(len(k))]
                mrr = [mrr[i] + sub_mrr[i] for i in range(len(k))]
                apks = apks + apk
                if i % 10000 == 0:
                    print(
                        'Tested Users {0}: test result precision@5 {1:.5f}, recall@5 {2:.5f}, mrr@5 {3:.5f}, precision@10 {4:.5f}, '
                        'recall@10 {5:.5f}, mrr@10 {6:.5f}, precision@20 {7:.5f}, recall@20 {8:.5f}, mrr@20 {9:.5f}, '
                        'MAP {10:.5f}, time: {11}'.format(i,
                                                        precision[
                                                            0] / num_users,
                                                        recall[
                                                            0] / num_users,
                                                        mrr[0] / num_users,
                                                        precision[
                                                            1] / num_users,
                                                        recall[
                                                            1] / num_users,
                                                        mrr[1] / num_users,
                                                        precision[
                                                            2] / num_users,
                                                        recall[
                                                            2] / num_users,
                                                        mrr[2] / num_users,
                                                        apks / num_users,
                                                        datetime.now()))
            print(
                'Tested Users {0}: test result precision@5 {1:.5f}, recall@5 {2:.5f}, mrr@5 {3:.5f}, precision@10 {4:.5f}, '
                'recall@10 {5:.5f}, mrr@10 {6:.5f}, precision@20 {7:.5f}, recall@20 {8:.5f}, mrr@20 {9:.5f}, '
                'MAP {10:.5f}, time: {11}'.format(i,
                                                precision[
                                                    0] / num_users,
                                                recall[
                                                    0] / num_users,
                                                mrr[0] / num_users,
                                                precision[
                                                    1] / num_users,
                                                recall[
                                                    1] / num_users,
                                                mrr[1] / num_users,
                                                precision[
                                                    2] / num_users,
                                                recall[
                                                    2] / num_users,
                                                mrr[2] / num_users,
                                                apks / num_users,
                                                datetime.now()))
            #          if epoch < 5:
            #                 pickle.dump([precision, recall], gzip.open("epoch_{0}_results.p",'wb'))
            if (apks / num_users) > best_map:
                best_p5 = precision[0] / num_users
                best_p10 = precision[1] / num_users
                best_p20 = precision[2] / num_users
                best_r5 = recall[0] / num_users
                best_r10 = recall[1] / num_users
                best_r20 = recall[2] / num_users
                best_m5 = mrr[0] / num_users
                best_m10 = mrr[1] / num_users
                best_m20 = mrr[2] / num_users
                best_map = apks / num_users

            print("==================================================================================")
            print("Testing End..")

    print("==================================================================================")
    print("Training End..")

    print("batch_size: ", batch_size)
    print("negative_sample_size: ", negative_sample_size)

    print("learning_rate: ", learning_rate)
    print("Loading data...", data_name)
    print("Property: ", micro_item_list)
