from FileTool import *
from StructureTool import *
root_path = "/home/cwt/project/data/lastfm-dataset-1K/"
source_file_path = root_path + "valid"
dict_Sku = {}

def order_dict(dicts, n):
    result = []
    result1 = []
    p = sorted([(k, v) for k, v in dicts.items()], reverse=True)
    s = set()
    for i in p:
        s.add(i[1])
    for i in sorted(s, reverse=True)[:n]:
        for j in p:
            if j[1] == i:
                result.append(j)
    for r in result:
        result1.append(r[0])
    return result1

listlist = FileTool.read_file_to_list_list(source_file_path, lineItemFilt=',')
for line in listlist:
    line = line[0]
    items = line[1:-2].split(',')
    for item in items:
        unit = item.split('+')
        StructureTool.addDict(dict_Sku, str(unit[1] + '+' + unit[0]))
topSku = order_dict(dict_Sku, 10000)
FileTool.write_file_listStr(root_path + 'topSku', topSku)


