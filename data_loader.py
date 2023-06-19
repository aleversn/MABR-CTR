from functools import reduce
from torch.utils import data
from torch.utils.data.dataloader import default_collate
import torch
import random as rnd
from .preprocess.Data import *
from .preprocess.FileTool import *


class MicroBehaviorDataset(data.Dataset):
    def __init__(self, file_name, data_name, micro_item_list, emb_dic_file_path, map_file_bottom_id_to_itemsId):
        if data_name == 'Computers':
            root = '../data/Computers_8/'
            self.num_items = 93140
            self.seq_len_max = 30
        elif data_name == 'Applicances':
            root = '../data/Applicances/'
            self.num_items = 16514
            self.seq_len_max = 40
        elif data_name == 'Computers_sample':
            root = '../data/Computer/'
            self.num_items = 2255
        self.micro_item_list = micro_item_list
        self.emb_wgts_micro_items_dict = []
        self.neg_items_path = root + "neg_item.SBCGD"
        for item in self.micro_item_list:
            self.emb_wgts_micro_items_dict.append(torch.nn.Embedding.from_pretrained(torch.tensor(
                    np.loadtxt(root + item + ".reidx"))))
        [file_name, self.map_file_bottom_id_to_itemsId, self.emb_dic_file_path] = \
            FileTool.add_folder_file_list(root, [file_name, map_file_bottom_id_to_itemsId, emb_dic_file_path])
        # get split item and micro behavior
        self.emb_dic = torch.nn.Embedding.from_pretrained(torch.tensor(np.loadtxt(self.emb_dic_file_path)))

        [self.dataset, self.micro_actions, self.num_users] = Data.load_data_trans_bottomId_to_microItemsId(file_name,
                                                                                    self.map_file_bottom_id_to_itemsId,
                                                                                    self.micro_item_list)

        self.neg_items = FileTool.read_file_to_list_list(self.neg_items_path)

        self.transmicro = [torch.nn.Linear(5, 1), torch.nn.Linear(8, 1), torch.nn.Linear(5, 1),torch.nn.Linear(5, 1)]

        self.mode = 'pos'
        self.word_dim = 53




    def __getitem__(self, idx):
        # loading user data
        items = self.emb_wgts_micro_items_dict[0](torch.tensor(self.micro_actions[0][idx])).float()
        micro = self.get_user_micro_action(idx)
        micro = torch.stack(micro).float()
        return idx + 1, items, micro

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.dataset)

    def limit(self, x):
        return (x - (torch.min(x))) / (torch.max(x) - torch.min(x))

    def str2int(self, s):
        return reduce(lambda x, y: x * 10 + y,
                      map(lambda s: {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s],
                          s))

    def get_user_micro_action(self, idx):
        micro_actions = []
        for i in range(len(self.micro_item_list)-1):
            micro_action = self.micro_actions[i+1][idx]
            micro_actions.append(self.transmicro[i](self.emb_wgts_micro_items_dict[i+1](torch.tensor(micro_action)).float()).squeeze())
        # micro_actions = np.array(micro_actions)
        return micro_actions


    def get_neg_items(self, user_id, batch_size):
        neg_items, neg_actions, actions = [], [], []
        for idx in user_id:
            item_ids = rnd.sample(range(len(self.neg_items[idx-1])), batch_size)
            item_ids.sort()
            micro_action = []
            neg_action = [self.neg_items[idx-1][id] for id in item_ids]
            items = [self.str2int(item.split('+')[0]) for item in neg_action]
            actions.append([self.str2int(item.split('+')[1]) for item in neg_action])
            actions.append([self.str2int(item.split('+')[2]) for item in neg_action])
            actions.append([self.str2int(item.split('+')[3]) for item in neg_action])
            actions.append([self.str2int(item.split('+')[4]) for item in neg_action])

            neg_items.append(self.emb_wgts_micro_items_dict[0](torch.tensor(items)))
            for i in range(len(self.micro_item_list) - 1):
                micro_action.append(self.transmicro[i](
                    self.emb_wgts_micro_items_dict[i + 1](torch.tensor(actions[i])).float()).squeeze())
            neg_actions.append(torch.stack(micro_action))

        neg_items = torch.stack(neg_items).float()
        neg_actions = torch.stack(neg_actions).float()

        return neg_items, neg_actions

    def get_user_review(self, user_id):
        if user_id in self.user_review:
            user_reviews = self.user_review[user_id]
            review_len = len(user_reviews)
            if review_len < self.max_reviews:
                pad_len = self.max_reviews - review_len
                pad_vector = np.zeros((pad_len, self.word_dim))
                user_reviews = np.concatenate((user_reviews, pad_vector), axis=0)
            else:
                user_reviews = user_reviews[:self.max_reviews]
        else:
            user_reviews = np.zeros((self.max_reviews, self.word_dim))
        return user_reviews
    
    def get_user_review_scores(self, user_id):
        user_review_scores = []
        for score_dict in self.user_review_score_dict:
            if user_id in score_dict:
                user_review_score = score_dict[user_id]
                scores_len = len(user_review_score)
                if scores_len < self.max_reviews:
                    pad_len = self.max_reviews - scores_len
                    user_review_score = np.concatenate((user_review_score, np.asarray([0.0] * pad_len)))
                else:
                    user_review_score = user_review_score[:self.max_reviews]
            else:
                user_review_score = np.zeros(self.max_reviews)
            user_review_scores.append(user_review_score)
        user_review_scores = np.array(user_review_scores)
        return user_review_scores
    
    def get_item_review(self, item_id):
        item_reviews = self.item_review[item_id]
        review_len = len(item_reviews)
        if review_len < self.max_reviews:
            pad_len = self.max_reviews - review_len
            pad_vector = np.zeros((pad_len, self.word_dim))
            item_reviews = np.concatenate((item_reviews, pad_vector), axis=0)
        else:
            item_reviews = item_reviews[:self.max_reviews]
        return item_reviews
    
    def get_item_review_scores(self, item_id):
        item_review_scores = []
        for score_dict in self.item_review_score_dict:
            if item_id in score_dict:
                item_review_score = score_dict[item_id]
                score_len = len(item_review_score)
                if score_len < self.max_reviews:
                    pad_len = self.max_reviews - score_len
                    item_review_score = np.concatenate((item_review_score, np.asarray([0.0] * pad_len)))
                else:
                    item_review_score = item_review_score[:self.max_reviews]
            else:
                item_review_score = np.zeros(self.max_reviews)
            item_review_scores.append(item_review_score)
        item_review_scores = np.array(item_review_scores)
        return item_review_scores
    
    def get_user_items(self, user_id):
        #df = self.user_dataset_group.get_group(user_id)
        # user_items = list(df['business_id'])
        user_items = self.micro_actions[0][user_id]
        return user_items
    
    def pad_data(self, user_item_list, max_len, pad_id):
        result_list = np.zeros((len(user_item_list), max_len))
        result_list[:] = pad_id
        # left pad
        for i in range(len(result_list)):
            result_list[i, (max_len-len(user_item_list[i])):] = user_item_list[i]
        result_list = result_list.astype(int)
        return result_list
        
        
def my_collate(batch):

    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def get_loader(file_name, data_name, micro_item_list, total_emb_path, map_file_bottom_id_to_itemsId, batch_size=100, shuffle=False, num_workers=0):
    """Builds and returns Dataloader."""
    dataset = MicroBehaviorDataset(file_name, data_name, micro_item_list, total_emb_path, map_file_bottom_id_to_itemsId)
    data_loader = data.DataLoader(dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate)
    return data_loader, dataset
