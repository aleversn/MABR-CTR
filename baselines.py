import random
from sklearn.metrics import log_loss, roc_auc_score
from deepctr_torch.models import *
import argparse
import numpy as np
import pandas as pd
from LoadData import *

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--dataset", type=str, default='lastfm')
    parse.add_argument("--device", type=str, default='cuda:0')
    # parse.add_argument("--num_epochs", type=int, default=30)
    # parse.add_argument("--dropout", type=float, default=0.6)
    parse.add_argument("--model", type=str, default='din')
    args = parse.parse_args()
    device = args.device
    dataset = args.dataset
    # model_name = args.model
    # batch_list = [128, 256, 512]
    # use_cuda = True
    # if use_cuda and torch.cuda.is_available():
    #     print('cuda ready...')
    #     device = 'cuda:3'
    # for model_name in ['xdeepfm']:
    for model_name in ['din', 'dien', 'DeepFM', 'difm', 'ifm', 'onn', 'xdeepfm', 'autoInt', 'dcn', 'afm', 'nfm',
                       'wdl', 'ccpm', 'pnn']:
    # for model_name in ['dien', 'difm']:
        print("model is: {}, dataset is: {}".format(model_name, dataset))
        batches, drops, aucs, loss = [], [], [], []
        batch_size, dropout, l2_reg_dnn, l2_reg_embedding, l2_reg_linear, l2_reg_cross, alpha, = 16, 0, 0, 0, 0, 0, 0
        l2_reg_att, att_fac, bi_dropout = 0.0, 0, 0.0
        optimizer, cross_param, gru_type, kernel_type = 'sgd', '', '', ''
        for i in range(5):
            if model_name == 'dien':
                train_x, train_y, test_x, test_y, feature_columns, behavior_feature_list = \
                    LoadData.get_data(dataset, use_neg=True)
                model = DIEN(feature_columns, behavior_feature_list, dnn_dropout=dropout, l2_reg_dnn=l2_reg_dnn,
                             l2_reg_embedding=l2_reg_embedding, gru_type=gru_type, alpha=alpha,
                             use_negsampling=False, use_bn=True, device=device, seed=random.randint(1000, 9999))
            elif model_name == 'din':
                train_x, train_y, test_x, test_y, feature_columns, behavior_feature_list = \
                    LoadData.get_data(dataset, use_neg=True)
                model = DIN(feature_columns, behavior_feature_list, device=device, dnn_dropout=dropout,
                            l2_reg_dnn=l2_reg_dnn, l2_reg_embedding=l2_reg_embedding,
                            att_weight_normalization=True, dnn_use_bn=True, seed=random.randint(1000, 9999))
            elif model_name == 'DeepFM':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = DeepFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                               dnn_dropout=dropout, l2_reg_dnn=l2_reg_dnn, l2_reg_embedding=l2_reg_embedding,
                               l2_reg_linear=l2_reg_linear,
                               device=device, dnn_use_bn=True, seed=random.randint(1000, 9999))
            elif model_name == 'difm':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = DIFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                             l2_reg_dnn=l2_reg_dnn, l2_reg_linear = l2_reg_linear, l2_reg_embedding=l2_reg_embedding,
                             dnn_dropout=dropout,
                             att_head_num=3, dnn_use_bn=True, device=device, seed=random.randint(1000, 9999))
            elif model_name == 'ifm':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = IFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                            dnn_dropout=dropout, l2_reg_dnn=l2_reg_dnn, l2_reg_embedding=l2_reg_embedding,
                            l2_reg_linear=l2_reg_embedding,
                            dnn_use_bn=True, device=device, seed=random.randint(1000, 9999))
            elif model_name == 'onn':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = ONN(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                            dnn_dropout=dropout, l2_reg_dnn=l2_reg_dnn, l2_reg_embedding=l2_reg_embedding,
                            l2_reg_linear=l2_reg_embedding,
                            dnn_use_bn=True, device=device, seed=random.randint(1000, 9999))
            elif model_name == 'xdeepfm':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = xDeepFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                                dnn_dropout=dropout, l2_reg_dnn=l2_reg_dnn, l2_reg_embedding=l2_reg_embedding,
                                l2_reg_linear=l2_reg_embedding,
                                cin_layer_size=[], device=device, seed=random.randint(1000, 9999))
            elif model_name == 'autoInt':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = AutoInt(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                                dnn_dropout=dropout, l2_reg_dnn=l2_reg_dnn, l2_reg_embedding=l2_reg_embedding,
                                att_layer_num=0, dnn_use_bn=True, device=device, seed=random.randint(1000, 9999))
            elif model_name == 'dcn':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = DCN(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns, device=device,
                            dnn_dropout=dropout, l2_reg_dnn=l2_reg_dnn, l2_reg_embedding=l2_reg_embedding,
                            l2_reg_cross=l2_reg_cross, cross_parameterization=cross_param,
                            dnn_use_bn=True, seed=random.randint(1000, 9999))
            elif model_name == 'afm':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = AFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                            afm_dropout=dropout, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding,
                            attention_factor=att_fac, l2_reg_att=l2_reg_att,
                            device=device, seed=random.randint(1000, 9999))
            elif model_name == 'nfm':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = NFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                            dnn_dropout=dropout, l2_reg_dnn=l2_reg_dnn, l2_reg_embedding=l2_reg_embedding,
                            l2_reg_linear=l2_reg_linear, bi_dropout=bi_dropout,
                            device=device, seed=random.randint(1000, 9999))
            elif model_name == 'wdl':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = WDL(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                            dnn_dropout=dropout, l2_reg_dnn=l2_reg_dnn, l2_reg_embedding=l2_reg_embedding,
                            l2_reg_linear=l2_reg_linear,
                            dnn_use_bn=True, device=device, seed=random.randint(1000, 9999))
            elif model_name == 'pnn':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = PNN(dnn_feature_columns=feature_columns, device=device, l2_reg_dnn=l2_reg_dnn,
                            kernel_type=kernel_type, l2_reg_embedding=l2_reg_embedding, dnn_dropout=dropout,
                            seed=random.randint(1000, 9999))
            elif model_name == 'ccpm':
                train_x, train_y, test_x, test_y, feature_columns, _ = LoadData.get_data(dataset, use_neg=False)
                model = CCPM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                             dnn_dropout=dropout, l2_reg_dnn=l2_reg_dnn, l2_reg_embedding=l2_reg_embedding,
                             l2_reg_linear=l2_reg_linear,
                             dnn_use_bn=True, device=device, seed=random.randint(1000, 9999))
            model.compile(optimizer, 'binary_crossentropy', metrics=['binary_crossentropy', 'auc'])
            history = model.fit(train_x, train_y, batch_size=batch_size, epochs=5, verbose=2,
                                validation_split=0, shuffle=False)
            pred_ans = model.predict(test_x, batch_size)
            print("")
            logloss = round(log_loss(test_y, pred_ans), 4)
            auc = round(roc_auc_score(test_y, pred_ans), 4)
            batches.append(batch_size)
            drops.append(dropout)
            aucs.append(auc)
            loss.append(logloss)
            print("{} batch_size={}, dropout={}, test LogLoss: {}, test AUC: {}".
                  format(i, batch_size, dropout, logloss, auc))
        batches.append(batch_size)
        drops.append(dropout)
        aucs.append(str(round(np.mean(aucs[-5:]), 4)) + '±' + str(round(np.std(aucs[-5:]), 4)))
        loss.append(str(round(np.mean(loss[-5:]), 4)) + '±' + str(round(np.std(loss[-5:]), 4)))
        output = {'batch_size': batches, 'dropout': drops, 'auc': aucs, 'logloss': loss}
        output = pd.DataFrame(output)
        output.to_excel('../{}/{}_results.xlsx'.format(dataset, model_name))
print('all batch is done!')
