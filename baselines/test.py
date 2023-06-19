import sys
from deepctr_torch.models import *
sys.path.append("/home/cwt/project/idea2/preprocess")
from sklearn.metrics import log_loss, roc_auc_score
import optuna
import random
import sys
import math
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
sys.path.append("/home/cwt/project/idea2/preprocess")
from FileTool import *
from SessionItemBase import *


def list_to_array(x, max_len):
    n_array = np.zeros([len(x), max_len], dtype=np.int64)
    for i, j in enumerate(x):
        n_array[i][0:len(j)] = j
    return n_array


def get_neg_sample(sku_list, cate_list, topsku_list):
    pos_list = []
    for i in range(len(sku_list)):
        pos_list.append(str(sku_list[i] + '+' + str(cate_list[i])))
    tmp = list(set(topsku_list) - set(pos_list))
    neg = tmp[random.randrange(0, len(tmp))]
    neg = neg.split('+')
    return [int(neg[0]), int(neg[1])]


def get_dataset_detail_lastfm(data, topSku, maxlen):
    predict_item_id, predict_art_id, predict_ans = [], [], []
    pos_length = []
    max_pos_len = maxlen
    item_ids, artist_ids, skips, timestamps = [], [], [], []
    for i in range(len(data)):
        item_id, art_id, skip, timestamp = [], [], [], []
        for j in range(len(data[i]) - 1):
            # item = SessionItemBase(data[i][j])
            item = data[i][j].split('+')
            item_id.append(item[1])
            art_id.append(item[0])
            skip.append(item[2])
            timestamp.append(item[3])
        q = random.random()
        # print(q)
        if q > 0.5:
            item = data[i][-1].split('+')
            predict_item_id.append(item[1])
            predict_art_id.append(item[0])
            if int(item[3]) < 2:
                predict_ans.append(0)
            else:
                predict_ans.append(1)
        else:
            item = get_neg_sample(item_id, art_id, topSku)
            predict_item_id.append(item[0])
            predict_art_id.append(item[1])
            predict_ans.append(0)
        pos_length.append(len(item_id))
        item_ids.append(item_id)
        artist_ids.append(art_id)
        skips.append(list(map(int, skip)))
        timestamps.append(timestamp)
    # print(predict_ans)
    dict = {'item_id': np.array(list(map(int, predict_item_id))), 'skip': np.ones(len(predict_item_id)),
            'art_id': np.array(list(map(int, predict_art_id))),
            'hist_item_id': list_to_array(item_ids, max_pos_len),
            'hist_art_id': list_to_array(artist_ids, max_pos_len),
            'hist_skip': list_to_array(skips, max_pos_len), 'pos_length': np.array(pos_length),
            'hist_timestamp': list_to_array(timestamps, max_pos_len)}
    return dict, np.array(list(map(int, predict_ans)))


def get_dataset_detail(data, label, max_len):
    predict_item_id, predict_cate_id, predict_ans = [], [], []
    pos_length, neg_length = [], []
    max_pos_len, max_neg_len = 0, 0
    item_ids, cate_ids, behaviors, gaps, dwells = [], [], [], [], []
    neg_item_ids, neg_cate_ids, neg_behaviors, neg_gaps, neg_dwells = [], [], [], [], []
    for i in range(len(data)):
        item_id, cate_id, behavior, gap, dwell = [], [], [], [], []
        neg_item_id, neg_cate_id, neg_behavior, neg_gap, neg_dwell = [], [], [], [], []
        for j in range(len(data[i]) - 1):
            item = SessionItemBase(data[i][j])
            if label[i][j] == '1':
                item_id.append(item.sku)
                cate_id.append(item.cid3)
                behavior.append(item.bh)
                gap.append(item.gap)
                dwell.append(item.dwell)
            else:
                neg_item_id.append(item.sku)
                neg_cate_id.append(item.cid3)
                neg_behavior.append(item.bh)
                neg_gap.append(item.gap)
                neg_dwell.append(item.dwell)
        item = SessionItemBase(data[i][-1])
        predict_item_id.append(item.sku)
        predict_cate_id.append(item.cid3)
        predict_ans.append(label[i][-1])
        pos_length.append(len(item_id))
        neg_length.append(len(neg_item_id))
        item_ids.append(list(map(int, item_id)))
        cate_ids.append(list(map(int, cate_id)))
        behaviors.append(list(map(int, behavior)))
        gaps.append(list(map(int, gap)))
        dwells.append(list(map(int, dwell)))
        if max_pos_len < len(item_id):
            max_pos_len = len(item_id)
        if max_neg_len < len(neg_item_id):
            max_neg_len = len(neg_item_id)
        neg_item_ids.append(list(map(int, neg_item_id)))
        neg_cate_ids.append(list(map(int, neg_cate_id)))
        neg_behaviors.append(list(map(int, neg_behavior)))
        neg_gaps.append(list(map(int, neg_gap)))
        neg_dwells.append(list(map(int, neg_dwell)))
    print("max_pos_len is: {}, max_neg_len is: {}".format(max_pos_len, max_neg_len))
    dict = {'item_id': np.array(list(map(int, predict_item_id))), 'gap': np.zeros(len(predict_item_id)),
            'cate_id': np.array(list(map(int, predict_cate_id))), 'neg_length': np.array(neg_length),
            'behavior': np.zeros(len(predict_item_id)), 'dwell': np.zeros(len(predict_item_id)),
            'hist_item_id': list_to_array(item_ids, max_len), 'hist_cate_id': list_to_array(cate_ids, max_len),
            'hist_behavior': list_to_array(behaviors, max_len), 'hist_gap': list_to_array(gaps, max_len),
            'hist_dwell': list_to_array(dwells, max_len), 'neg_hist_gap': list_to_array(neg_gaps, max_len),
            'neg_hist_item_id': list_to_array(neg_item_ids, max_len), 'pos_length': np.array(pos_length),
            'neg_hist_cate_id': list_to_array(neg_cate_ids, max_len),
            'neg_hist_behavior': list_to_array(neg_behaviors, max_len),
            'neg_hist_dwell': list_to_array(neg_dwells, max_len)}
    return dict, np.array(list(map(int, predict_ans)))


def get_dataset_detail2(data, topSku):
    predict_item_id, predict_cate_id, predict_ans = [], [], []
    pos_length = []
    max_pos_len = 761
    item_ids, cate_ids, behaviors, timestamps = [], [], [], []
    for i in range(len(data)):
        item_id, cate_id, behavior, timestamp = [], [], [], []
        if random.random() > 0.5:
            for j in range(len(data[i]) - 1):
                # item = SessionItemBase(data[i][j])
                item = data[i][j].split('+')
                item_id.append(item[0])
                cate_id.append(item[1])
                behavior.append(item[2])
                timestamp.append(item[3])

            item = data[i][-1].split('+')
            predict_item_id.append(item[0])
            predict_cate_id.append(item[1])
            predict_ans.append(1)
            pos_length.append(len(item_id))
            item_ids.append(item_id)
            cate_ids.append(cate_id)
            behaviors.append(list(map(int, behavior)))
            timestamps.append(timestamp)
        else:
            for j in range(len(data[i]) - 1):
                # item = SessionItemBase(data[i][j])
                item = data[i][j].split('+')
                item_id.append(item[0])
                cate_id.append(item[1])
                behavior.append(item[2])
                timestamp.append(item[3])

            item = get_neg_sample(item_id, cate_id, topSku)
            predict_item_id.append(item[0])
            predict_cate_id.append(item[1])
            predict_ans.append(0)
            pos_length.append(len(item_id))
            item_ids.append(item_id)
            cate_ids.append(cate_id)
            behaviors.append(list(map(int, behavior)))
            timestamps.append(timestamp)
        # if max_pos_len < len(item_id):
        #     max_pos_len = len(item_id)
    # print("max_pos_len is: {}".format(max_pos_len))
    dict = {'item_id': np.array(list(map(int, predict_item_id))), 'behavior': np.ones(len(predict_item_id)),
            'cate_id': np.array(list(map(int, predict_cate_id))),
            'hist_item_id': list_to_array(item_ids, max_pos_len),
            'hist_cate_id': list_to_array(cate_ids, max_pos_len),
            'hist_behavior': list_to_array(behaviors, max_pos_len), 'pos_length': np.array(pos_length),
            'hist_timestamp': list_to_array(timestamps, max_pos_len)}
    return dict, np.array(list(map(int, predict_ans)))


def dataset_split(data, split_size):
    train, test = [], []
    max_train, max_test = 0, 0
    for line in data:
        length = math.ceil(split_size*len(line))
        train.append(line[:length])
        test.append(line[length:])
        if length > max_train:
            max_train = length
        if len(line) - length > max_test:
            max_test = len(line) - length
    print("max_train_len is: {}, max_test_len is: {}".format(max_train, max_test))
    return train, test


def get_dataset(folder, data_path, label_path, max_len):
    file_data = os.path.join(folder, data_path)
    file_label = os.path.join(folder, label_path)
    data = FileTool.read_file_to_list_list(file_data)
    label = FileTool.read_file_to_list_list(file_label)
    # split data and label to train and test
    # train, test = train_test_split(data, test_size=0.2, random_state=2022)
    # train_label, test_label = train_test_split(label, test_size=0.2, random_state=2022)
    train, test = dataset_split(data, 0.5)
    train_label, test_label = dataset_split(label, 0.5)
    train_x, train_y = get_dataset_detail(train, train_label, max_len)
    test_x, test_y = get_dataset_detail(test, test_label, max_len)
    return train_x, train_y, test_x, test_y


def get_data_userbehavior(folder, train_path, test_path):
    file_train = os.path.join(folder, train_path)
    file_test = os.path.join(folder, test_path)
    train = FileTool.read_file_to_list_list(file_train, sep='\t')
    test = FileTool.read_file_to_list_list(file_test, sep='\t')
    topSku = FileTool.read_line_to_list_str(os.path.join(folder, 'top5000sku'))
    # split data and label to train and test
    # train, test = train_test_split(data, test_size=0.2, random_state=2022)
    # train_label, test_label = train_test_split(label, test_size=0.2, random_state=2022)
    # train, test, _ = LoadData.split_userbehavior(data)
    # valid = LoadData.convert_to_id(valid)
    # FileTool.write_file_list_list(os.path.join(folder, 'train'), train)
    # FileTool.write_file_list_list(os.path.join(folder, 'test'), test)
    # FileTool.write_file_list_list(os.path.join(folder, 'valid'), valid, ' ')
    train_x, train_y = get_dataset_detail2(train, topSku)
    test_x, test_y = get_dataset_detail2(test, topSku)
    return train_x, train_y, test_x, test_y


def load_data_lastfm(file_path):
    data = []
    listlist = FileTool.read_file_to_list_list(file_path)
    for line in listlist:
        line = line[0]
        items = line[1:-2].split(',')
        data.append(items)
    return data


def get_data_lastfm(folder, file_path, topSku_path):
    file_data = os.path.join(folder, file_path)
    data = load_data_lastfm(file_data)
    topSku = FileTool.read_line_to_list_str(os.path.join(folder, topSku_path))
    # split data and label to train and test
    # train, test = train_test_split(data, test_size=0.2, random_state=2022)
    # train_label, test_label = train_test_split(label, test_size=0.2, random_state=2022)
    train, test = dataset_split(data, 0.5)
    train_x, train_y = get_dataset_detail_lastfm(train, topSku, 86290)
    test_x, test_y = get_dataset_detail_lastfm(test, topSku, 86290)
    return train_x, train_y, test_x, test_y


def get_data(data_name, use_neg=False, hash_flag=False):
    train, test = {}, {}
    train_y, test_y, feature_columns, behavior_feature_list = [], [], [], []
    if data_name == 'Computers':
        root = '/home/cwt/project/data/Computers/'
        data_path = 'session.SBCGD'
        label_path = 'labels'
        train, train_y, test, test_y = get_dataset(root, data_path, label_path, max_len=25)
        feature_columns = [SparseFeat('item_id', vocabulary_size=62006 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('cate_id', vocabulary_size=86 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('behavior', vocabulary_size=10 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('gap', vocabulary_size=5 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('dwell', vocabulary_size=5 + 1, embedding_dim=30, use_hash=hash_flag)]
        feature_columns += [
            VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=62006 + 1, embedding_dim=30,
                                        embedding_name='item_id'), maxlen=25, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=86 + 1, embedding_dim=30,
                                        embedding_name='cate_id'), maxlen=25, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_behavior', vocabulary_size=10 + 1, embedding_dim=30,
                                        embedding_name='behavior'), maxlen=25, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_dwell', vocabulary_size=5 + 1, embedding_dim=30,
                                        embedding_name='dwell'), maxlen=25, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_gap', vocabulary_size=5 + 1, embedding_dim=30,
                                        embedding_name='gap'), maxlen=25, length_name="pos_length")]
        if use_neg:
            feature_columns += [
                VarLenSparseFeat(SparseFeat('neg_hist_item_id', vocabulary_size=62006 + 1, embedding_dim=30,
                                            embedding_name='item_id'), maxlen=25, length_name="neg_length"),
                VarLenSparseFeat(SparseFeat('neg_hist_cate_id', vocabulary_size=86 + 1, embedding_dim=8,
                                            embedding_name='cate_id'), maxlen=25, length_name="neg_length"),
                VarLenSparseFeat(SparseFeat('neg_hist_behavior', vocabulary_size=10 + 1, embedding_dim=5,
                                            embedding_name='behavior'), maxlen=25, length_name="neg_length"),
                VarLenSparseFeat(SparseFeat('neg_hist_dwell', vocabulary_size=5 + 1, embedding_dim=5,
                                            embedding_name='dwell'), maxlen=25, length_name="neg_length"),
                VarLenSparseFeat(SparseFeat('neg_hist_gap', vocabulary_size=5 + 1, embedding_dim=5,
                                            embedding_name='gap'), maxlen=25, length_name="neg_length")]
        behavior_feature_list = ["item_id", "behavior", "cate_id", "gap", "dwell"]
    elif data_name == 'Applicances':
        root = '/home/cwt/project/data/Applicances/'
        data_path = 'session.SBCGD'
        label_path = 'labels'
        train, train_y, test, test_y = get_dataset(root, data_path, label_path, max_len=75)
        feature_columns = [SparseFeat('item_id', vocabulary_size=59999 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('cate_id', vocabulary_size=91 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('behavior', vocabulary_size=10 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('gap', vocabulary_size=5 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('dwell', vocabulary_size=5 + 1, embedding_dim=30, use_hash=hash_flag)]
        feature_columns += [
            VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=59999 + 1, embedding_dim=30,
                                        embedding_name='item_id'), maxlen=75, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=91 + 1, embedding_dim=30,
                                        embedding_name='cate_id'), maxlen=75, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_behavior', vocabulary_size=10 + 1, embedding_dim=30,
                                        embedding_name='behavior'), maxlen=75, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_dwell', vocabulary_size=5 + 1, embedding_dim=30,
                                        embedding_name='dwell'), maxlen=75, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_gap', vocabulary_size=5 + 1, embedding_dim=30,
                                        embedding_name='gap'), maxlen=75, length_name="pos_length")]
        if use_neg:
            feature_columns += [
                VarLenSparseFeat(SparseFeat('neg_hist_item_id', vocabulary_size=59999 + 1, embedding_dim=30,
                                            embedding_name='item_id'), maxlen=75, length_name="neg_length"),
                VarLenSparseFeat(SparseFeat('neg_hist_cate_id', vocabulary_size=91 + 1, embedding_dim=8,
                                            embedding_name='cate_id'), maxlen=75, length_name="neg_length"),
                VarLenSparseFeat(SparseFeat('neg_hist_behavior', vocabulary_size=10 + 1, embedding_dim=5,
                                            embedding_name='behavior'), maxlen=75, length_name="neg_length"),
                VarLenSparseFeat(SparseFeat('neg_hist_dwell', vocabulary_size=5 + 1, embedding_dim=5,
                                            embedding_name='dwell'), maxlen=75, length_name="neg_length"),
                VarLenSparseFeat(SparseFeat('neg_hist_gap', vocabulary_size=5 + 1, embedding_dim=5,
                                            embedding_name='gap'), maxlen=75, length_name="neg_length")]
        behavior_feature_list = ["item_id", "behavior", "cate_id", "gap", "dwell"]
    elif data_name == 'UserBehaviors':
        root = '/home/cwt/project/data/UserBehavior/'
        # data_path = 'UserBehavior'
        train_path = 'train'
        test_path = 'test'
        train, train_y, test, test_y = get_data_userbehavior(root, train_path, test_path)
        feature_columns = [SparseFeat('item_id', vocabulary_size=2579724 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('cate_id', vocabulary_size=8512 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('behavior', vocabulary_size=4 + 1, embedding_dim=30, use_hash=hash_flag)
                           ]
        feature_columns += [
            VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=2579724 + 1, embedding_dim=30,
                                        embedding_name='item_id'), maxlen=761, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=8512 + 1, embedding_dim=30,
                                        embedding_name='cate_id'), maxlen=761, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_behavior', vocabulary_size=4 + 1, embedding_dim=30,
                                        embedding_name='behavior'), maxlen=761, length_name="pos_length")
        ]
        behavior_feature_list = ["item_id", "behavior", "cate_id"]
    elif data_name == 'lastfm':
        root = '/home/cwt/project/data/lastfm-dataset-1K/'
        data_path = 'valid'
        topSku_path = 'topSku'
        train, train_y, test, test_y = get_data_lastfm(root, data_path, topSku_path)
        feature_columns = [SparseFeat('item_id', vocabulary_size=64181 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('art_id', vocabulary_size=70657 + 1, embedding_dim=30, use_hash=hash_flag),
                           SparseFeat('skip', vocabulary_size=3, embedding_dim=30, use_hash=hash_flag)
                           ]
        feature_columns += [
            VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=64181 + 1, embedding_dim=30,
                                        embedding_name='item_id'), maxlen=86290, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_art_id', vocabulary_size=70657 + 1, embedding_dim=30,
                                        embedding_name='art_id'), maxlen=86290, length_name="pos_length"),
            VarLenSparseFeat(SparseFeat('hist_skip', vocabulary_size=3, embedding_dim=30,
                                        embedding_name='skip'), maxlen=86290, length_name="pos_length")
        ]
        # behavior_feature_list = ["item_id", "cate_id", "behavior"]
        behavior_feature_list = ["item_id", "art_id", "skip"]
    train_x = {name: train[name] for name in get_feature_names(feature_columns)}
    test_x = {name: test[name] for name in get_feature_names(feature_columns)}
    return train_x, train_y, test_x, test_y, feature_columns, behavior_feature_list


def run_trail(config):
    print("model is: {}, dataset is: {}".format(config["model"], config["dataset"]))
    batch_size = 16
    if config["model"] in ['dien', 'din']:
        train_x, train_y, test_x, test_y, feature_columns, behavior_feature_list = get_data(config["dataset"],
                                                                                            use_neg=False)
    else:
        train_x, train_y, test_x, test_y, feature_columns, _ = get_data(config["dataset"], use_neg=False)
    if config["model"] == "dien":
        model = DIEN(feature_columns, behavior_feature_list, dnn_dropout=config["dnn_dropout"], use_bn=True,
                     gru_type=config["gru_type"], use_negsampling=False, l2_reg_dnn=config["l2_reg_dnn"],
                     device=config["device"], alpha=config["alpha"], l2_reg_embedding=config["l2_reg_embedding"])
    elif config["model"] == "din":
        model = DIN(feature_columns, behavior_feature_list, dnn_dropout=config["dnn_dropout"], device=config["device"],
                    att_weight_normalization=True, dnn_use_bn=True, l2_reg_dnn=config["l2_reg_dnn"],
                    l2_reg_embedding=config["l2_reg_embedding"])
    elif config["model"] == 'DeepFM':
        model = DeepFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                       device=config["device"], dnn_dropout=config["dnn_dropout"], l2_reg_dnn=config["l2_reg_dnn"],
                       l2_reg_embedding=config["l2_reg_embedding"], l2_reg_linear=config["l2_reg_linear"],
                       dnn_use_bn=True)
    elif config["model"] == 'difm':
        model = DIFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns, att_head_num=5,
                     device=config["device"], dnn_dropout=config["dnn_dropout"], l2_reg_dnn=config["l2_reg_dnn"],
                     l2_reg_embedding=config["l2_reg_embedding"], l2_reg_linear=config["l2_reg_linear"],
                     dnn_use_bn=True)
    elif config["model"] == 'ifm':
        model = IFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                    device=config["device"], dnn_dropout=config["dnn_dropout"], l2_reg_dnn=config["l2_reg_dnn"],
                    l2_reg_embedding=config["l2_reg_embedding"], l2_reg_linear=config["l2_reg_linear"], dnn_use_bn=True)
    elif config["model"] == 'onn':
        model = ONN(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                    device=config["device"], dnn_dropout=config["dnn_dropout"], l2_reg_dnn=config["l2_reg_dnn"],
                    l2_reg_embedding=config["l2_reg_embedding"], l2_reg_linear=config["l2_reg_linear"], dnn_use_bn=True)
    elif config["model"] == 'xdeepfm':
        model = xDeepFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns, cin_layer_size=[],
                        device=config["device"], dnn_dropout=config["dnn_dropout"], l2_reg_dnn=config["l2_reg_dnn"],
                        l2_reg_embedding=config["l2_reg_embedding"], l2_reg_linear=config["l2_reg_linear"])
    elif config["model"] == 'autoInt':
        model = AutoInt(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns, att_layer_num=0,
                        l2_reg_dnn=config["l2_reg_dnn"], l2_reg_embedding=config["l2_reg_embedding"], dnn_use_bn=True,
                        device=config["device"], dnn_dropout=config["dnn_dropout"])
    elif config["model"] == 'dcn':
        model = DCN(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns, device=config["device"],
                    l2_reg_dnn=config["l2_reg_dnn"], l2_reg_embedding=config["l2_reg_embedding"], dnn_use_bn=True,
                    dnn_dropout=config["dnn_dropout"], cross_parameterization=config["cross_parameterization"],
                    l2_reg_cross=config["l2_reg_cross"],)
    elif config["model"] == 'afm':
        model = AFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                    attention_factor=config["attention_factor"], l2_reg_linear=config["l2_reg_linear"],
                    l2_reg_embedding=config["l2_reg_embedding"], l2_reg_att=config["l2_reg_att"],
                    device=config["device"], afm_dropout=config["dnn_dropout"])
    elif config["model"] == 'nfm':
        model = NFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                    l2_reg_linear=config["l2_reg_linear"], l2_reg_embedding=config["l2_reg_embedding"],
                    l2_reg_dnn=config["l2_reg_dnn"], bi_dropout=config["bi_dropout"],
                    device=config["device"], dnn_dropout=config["dnn_dropout"])
    elif config["model"] == 'wdl':
        model = WDL(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                    device=config["device"], dnn_dropout=config["dnn_dropout"], l2_reg_dnn=config["l2_reg_dnn"],
                    l2_reg_embedding=config["l2_reg_embedding"], l2_reg_linear=config["l2_reg_linear"])
    elif config["model"] == 'pnn':
        model = PNN(dnn_feature_columns=feature_columns, device=config["device"], dnn_dropout=config["dnn_dropout"],
                    l2_reg_dnn=config["l2_reg_dnn"], l2_reg_embedding=config["l2_reg_embedding"], use_outter=True,
                    kernel_type=config["kernel_type"])
    elif config["model"] == 'ccpm':
        model = CCPM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                    device=config["device"], dnn_dropout=config["dnn_dropout"], l2_reg_dnn=config["l2_reg_dnn"],
                    l2_reg_embedding=config["l2_reg_embedding"], l2_reg_linear=config["l2_reg_linear"], dnn_use_bn=True)
    model.compile(config["optimizer"], "binary_crossentropy", metrics=["binary_crossentropy", "auc"])
    history = model.fit(train_x, train_y, batch_size=batch_size, epochs=5, verbose=0, validation_split=0,
                        shuffle=False)
    pred_ans = model.predict(test_x, batch_size)
    auc = round(roc_auc_score(test_y, pred_ans), 4)
    logloss = round(log_loss(test_y, pred_ans), 4)
    print('test auc is: {}, test logloss is: {}'.format(auc, logloss))
    return auc


def objective(trial):
    params = {
        "l2_reg_dnn": trial.suggest_categorical("l2_reg_dnn", [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]),
        # "l2_reg_cross": trial.suggest_categorical("l2_reg_cross", [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]),
        "l2_reg_linear": trial.suggest_categorical("l2_reg_linear", [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]),
        # "l2_reg_att": trial.suggest_categorical("l2_reg_att", [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]),
        "l2_reg_embedding": trial.suggest_categorical("l2_reg_embedding",
                                                      [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]),
        "dnn_dropout": trial.suggest_float("dnn_dropout", 0.0, 0.7),
        # "bi_dropout": trial.suggest_float("bi_dropout", 0.0, 0.7),
        # "attention_factor": trial.suggest_int("attention_factor", 1, 10),
        # "alpha": trial.suggest_float("alpha", 0, 1.0),
        # "gru_type": trial.suggest_categorical("gru_type", ["GRU", "AUGRU", "AIGRU", "AGRU"]),
        # "kernel_type": trial.suggest_categorical("kernel_type", ["mat", "vec", "num"]),
        # "dataset": "Computers",
        # "dataset": "Applicances",
        "dataset": "lastfm",
        "device": "cuda:0",
        "model": "wdl",
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adagrad", "sgd"]),
        # "cross_parameterization": trial.suggest_categorical("cross_parameterization", ["vector", "matrix"]),
    }

    return run_trail(params)


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), storage='sqlite:///db.sqlite3')
study.optimize(objective, n_trials=100)
