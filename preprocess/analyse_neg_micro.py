from Config import *
from FileTool import *
from SessionItemBase import *
from functools import reduce
import pandas as pd
import numpy as np
folder = Config.folder
file_in = Config.file_data_raw
file_out_topsku = Config.topsku
file_out_lines = Config.file_data_raw_topsku
min_sku_cnt = Config.min_cnt_sku_limit
file_out = Config.file_data_raw_topsku_len + '30'


def str2int(s):
    return reduce(lambda x, y: x * 10 + y,
                  map(lambda s: {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s], s))


# FileTool.func_begin('count_neg_item...')
# listlist = FileTool.read_file_to_list_list(os.path.join(folder, file_out), Config.file_sep)
#
# print("len30 user num: ", len(listlist))
# neg_5_len_15_list, neg_5_len_10_list = [], []
# neg_10_len_5_list, neg_10_len_10_list = [], []
# neg_15_len_8_list, neg_15_len_10_list, neg_15_len_9_list = [], [], []
# neg_15_len_7_list, neg_15_len_6_list, neg_15_len_5_list = [], [], []
# valid_line = []
# count_neg_5_len_15_list, count_neg_5_len_10_list = 0, 0
# count_neg_5_len_11_list, count_neg_5_len_12_list = 0, 0
# count_neg_5_len_13_list, count_neg_5_len_14_list = 0, 0
# for line in listlist:  # a line is a user's all session
#     tmp = []
#     for i in range(len(line) - 1):  # a unit is a session
#         item = SessionItemBase(line[i])
#         if item.bh in ('1', '2', '3', '5'):
#             if str2int(item.dwell) < 5 and SessionItemBase(line[i + 1]).sku != item.sku:
#                 #
#                 tmp.append(line[i])
#
#     # if len(tmp) > 0:
#
#         # if (len(set_line) - len(set_tmp) >= 30):
#
#     if len(tmp) >= 10:
#         neg_5_len_10_list.append(tmp)
#         valid_line.append(line)
#         # count_neg_5_len_10_list += 1
#     # if len(tmp) >= 11:
#     #     count_neg_5_len_11_list += 1
#     # if len(tmp) >= 12:
#     #     count_neg_5_len_12_list += 1
#     # if len(tmp) >= 13:
#     #     count_neg_5_len_13_list += 1
#     # if len(tmp) >= 14:
#     #     count_neg_5_len_14_list += 1
#     # if len(tmp) >= 15:
#     #     count_neg_5_len_15_list += 1
#         # line_split_neg = [x for x in line if x not in tmp]
#         # if len(line_split_neg) >= 30:
#         #     neg_15_len_10_list.append(tmp)
#         #     valid_line.append(line_split_neg)
#     # if len(tmp) >= 9:
#     #     neg_15_len_9_list.append(tmp)
#     # if len(tmp) >= 8:
#     #     neg_15_len_8_list.append(tmp)
#     # if len(tmp) >= 7:
#     #     neg_15_len_7_list.append(tmp)
#     # if len(tmp) >= 6:
#     #     neg_15_len_6_list.append(tmp)
#     # if len(tmp) >= 5:
#     #     neg_15_len_5_list.append(tmp)
# print("len(valid_line)", len(valid_line))
# print("len(neg_5_len_10_list): ", len(neg_5_len_10_list), " ratio: ", len(neg_5_len_10_list)/len(listlist))
# # print("len(neg_5_len_11_list): ", count_neg_5_len_11_list, " ratio: ", count_neg_5_len_11_list/len(listlist))
# # print("len(neg_5_len_12_list): ", count_neg_5_len_12_list, " ratio: ", count_neg_5_len_12_list/len(listlist))
# # print("len(neg_5_len_13_list): ", count_neg_5_len_13_list, " ratio: ", count_neg_5_len_13_list/len(listlist))
# # print("len(neg_5_len_14_list): ", count_neg_5_len_14_list, " ratio: ", count_neg_5_len_14_list/len(listlist))
# # print("len(neg_5_len_15_list): ", count_neg_5_len_15_list, " ratio: ", count_neg_5_len_15_list/len(listlist))
# FileTool.write_file_list_list(os.path.join(folder, 'neg_5_len_10_list'), neg_5_len_10_list, Config.file_sep)
# # FileTool.write_file_list_list(os.path.join(folder, 'neg_15_len_10_list'), neg_15_len_10_list, Config.file_sep)
# FileTool.write_file_list_list(os.path.join(folder, 'valid_line'), valid_line, Config.file_sep)
# FileTool.func_end("count_neg_items_list_dwell_5")

# listlist = FileTool.read_file_to_list_list(os.path.join(Config.folder, 'session.SBCGD'), Config.file_sep)
# max_sku, max_bh, max_cid3, max_gap, max_dwell = 0, 0, 0, 0, 0
# for line in listlist:
#     for unit in line:
#         item = SessionItemBase(unit)
#         if str2int(item.sku) > max_sku:
#             max_sku = str2int(item.sku)
#         if str2int(item.bh) > max_bh:
#             max_bh = str2int(item.bh)
#         if str2int(item.cid3) > max_cid3:
#             max_cid3 = str2int(item.cid3)
#         if str2int(item.gap) > max_gap:
#             max_gap = str2int(item.gap)
#         if str2int(item.dwell) > max_dwell:
#             max_dwell = str2int(item.dwell)
#
# print("max_sku: {}".format(max_sku))
# print("max_bh: {}".format(max_bh))
# print("max_cid: {}".format(max_cid3))
# print("max_gap: {}".format(max_gap))
# print("max_dwell: {}".format(max_dwell))

# data = pd.read_excel('/home/cwt/project/idea2/test.xlsx')
# aucs = data['auc']
# loss = data['logloss']
# for i in range(int(len(aucs)/6)-1):
#     auc = aucs[i*6 : (i+1)*6].values
#     logloss = loss[i*6 : (i+1)*6].values
#     auc_avg = round(np.mean(auc[:-1]), 4)
#     auc_std = round(np.std(auc[:-1]), 4)
#     loss_avg = round(np.mean(logloss[:-1]), 4)
#     loss_std = round(np.std(logloss[:-1]), 4)
#     auc_result = list(map(float, auc[-1].split('±')))
#     loss_result = list(map(float, logloss[-1].split('±')))
#     if auc_avg == auc_result[0] and auc_std == auc_result[1]:
#         if loss_avg == loss_result[0] and loss_std == loss_result[1]:
#             continue
#         else:
#             print('{}, logloss: {}'.format(i+1, logloss))
#     else:
#         print('{}, auc: {}'.format(i+1, auc))
#         break
# print('done!')

# listlist = FileTool.read_file_to_list_list(os.path.join(Config.folder, 'labels'))
# print('validline is: {}'.format(len(listlist)))
# max_line, max_pos, max_neg, count = 0, 0, 0, 0
# for list in listlist:
#     pos, neg = 0, 0
#     for item in list:
#         if item == '1':
#             pos += 1
#         else:
#             neg += 1
#     if pos > max_pos:
#         max_pos = pos
#     if neg > max_neg:
#         max_neg = neg
#     if pos + neg > max_line:
#         max_line = pos+neg
#     if pos + neg > 100:
#         # print('pos is: {}, neg is: {}, total is: {}'.format(pos, neg, pos+neg))
#         count += 1
# print('long user num is: {}'.format(count))
# print('max_line: {}, max_pos: {}, max_neg: {}'.format(max_line, max_pos, max_neg))
# output = {'batch_size': [1,1,1], 'dropout': [0.6,0.6,0.6],}
# output = pd.DataFrame(output)
# output.to_excel('/home/cwt/project/idea2/output/dien_{}_{}_results.xlsx'.format('Computers',128))
label_list = FileTool.read_file_to_list_list(os.path.join(Config.folder, "labels"))
listlist = FileTool.read_file_to_list_list(os.path.join(Config.folder, "session.SBCGD"))
if len(label_list) != len(listlist):
    print("len(label_list) != len(listlist)")
else:
    for i in range(len(label_list)):
        if len(label_list[i]) != len(listlist[i]):
            print("len(label_list{}) != len(listlist{})".format(i, i))
    print("all equation!")