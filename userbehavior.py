import pandas as pd
from FileTool import *
from SessionItemBase import *
from StructureTool import *

root_path = "/home/cwt/project/data/UserBehavior/"
file_path = 'valid'
listlist = FileTool.read_file_to_list_list(os.path.join(root_path, file_path))
print('user num is: {}'.format(len(listlist)))
sku_dict = {}
total_num = 0
for line in listlist:
    total_num += len(line)
    for unit in line:
        item = unit.split('+')
        StructureTool.addDict(sku_dict, item[0])
print('sku num is: {}'.format(len(sku_dict)))
print('total behaviors: {}'.format(total_num))
