from Preprocess import *
from ProcessTrain import *
from Config import *
import os
#from PlotRes.Plot import *

class Process():
    def __init__(self):
        pass

    @classmethod
    def build_base_data(cls):
        # filter out the items with fewer than 50 occurrences, and transfer time to id
        Preprocess.build_raw_data()
        # # transfer sku.uniq, bh.raw, cid3.raw, dwell.id, gap.id to xx.w2v, get the embedding with word2vec
        Preprocess.get_embedding_items()
        # # transfer xx.w2v to xx.mapping and xx.reidx, .reidx is norm w2v vector value, .map???
        Preprocess.get_data_item_mapping_items()
        # # Preprocess.get_recall_topsku()
        ProcessTrain.get_data_session_train_itemid(Config.folder, Config.file_data_src, "session", mode="SBCT")
        pass

    @classmethod
    def build_train_data(cls):
        # Config.file_data_src: JD_xxx.topsku.lenxx, get session.SBCGD
        ProcessTrain.get_data_session_train_itemid(Config.folder, Config.file_data_src, "session", mode="SBCGD")
        # from session.SBCGD get session.SBCGD.id and session.SBCGD.id.mapping
        Preprocess.get_file_to_id_mapping(Config.folder, "session.SBCGD", "session.SBCGD.id", "session.SBCGD.id.mapping")
        # get session.SBCGD.id.train and session.SBCGD.id.test
        ProcessTrain.split_data_train_test(Config.folder, 'session.SBCGD.id', "session.SBCGD.id.mapping", -1, Config.train_ratio)
        # get sesseion.SBCGD.id.lenxx.train, sesseion.SBCGD.id.lenxx.test, sesseion.SBCGD.id.lenxx.train.div
        ProcessTrain.format_data_train_test(Config.folder, 'session.SBCGD.id', "session.SBCGD.id.mapping", Config.seq_len)
        # get session.SBCGD.id.lenxx.SBCGD.train, session.SBCGD.id.lenxx.SBCGD.test, session.SBCGD.id.lenxx.SBCGD.train.div
        ProcessTrain.get_file_micro_items_sequence(Config.folder, "session.SBCGD.id", "session.SBCGD.id.mapping", Config.seq_len)
        #
        ProcessTrain.get_file_micro_items_sequence_train_data(Config.folder, "session.SBCGD.id")
        pass

    @classmethod
    def analyse_data(cls):
        FileTool.func_begin("analyse_data")
        Preprocess.get_data_statis()
        #ProcessTrain.analyse_file_data_sbcd()
        FileTool.func_end("analyse_data")
        pass

    @classmethod
    def start(cls):
        Process.build_base_data()  # base data
        # Process.build_train_data() # train data
        # Process.analyse_data()
        pass

if __name__ == "__main__":
    Process.start()


