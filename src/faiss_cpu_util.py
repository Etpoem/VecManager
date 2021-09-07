# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   panjf
# @Time     :   2021/2/22 9:09
# @Desc     :   cpu 版本 faiss 引擎
# -------------------------------------------------------------
import os
import toml
import faiss
import logging
import pickle
import numpy as np
from threading import Timer
from pathlib import Path


logger = logging.getLogger('main.faiss')
cfg = toml.load(Path(__file__).parents[1] / 'config.toml')
index_file_path = Path(__file__).parents[1] / 'data/indexs'
index_file_path.mkdir(exist_ok=True, parents=True)
_ADD_TIMEOUT = cfg['faiss']['add_timeout']
_REMOVE_TIMEOUT = cfg['faiss']['remove_timeout']
_SUB_INDEX_SIZE = cfg['faiss']['sub_index_size']
_N_LIST = cfg['faiss']['nlist']
_N_PROBE = cfg['faiss']['nprobe']
_DIMENSION = cfg['faiss']['dimension']


class FaissManager(object):
    def __init__(self, db_name, max_size, is_multiple_gpus=False):
        """
        :param db_name:             string --
        :param max_size:            int --
        :param is_mutiple_gpus:     bool -- 此项无效，为兼容gpu版本而保留
        """
        self.db_name = db_name
        self.max_size = max_size
        self.is_multiple_gpus = is_multiple_gpus
        # 创建 index
        self.cpu_index = faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlatL2(_DIMENSION),
            _DIMENSION,
            _N_LIST,
            faiss.ScalarQuantizer.QT_fp16,
            faiss.METRIC_L2
        )
        self.sub_cpu_index = faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlatL2(_DIMENSION),
            _DIMENSION,
            _N_LIST,
            faiss.ScalarQuantizer.QT_fp16,
            faiss.METRIC_L2
        )
        # index 训练
        with open(Path(__file__).parents[1] / 'sample_features.pkl', 'rb') as f:
            sample_vectors = pickle.load(f)
        self.cpu_index.train(sample_vectors)
        self.sub_cpu_index.train(sample_vectors)
        del sample_vectors
        # 声明持久化文件路径及名称
        self.cpu_index_file = index_file_path / f'{self.db_name}.index'
        self.sub_cpu_index_file = index_file_path / f'sub_{self.db_name}.index'
        # 初始化向量库修改的后处理程序
        self.add_post_timer = Timer(_ADD_TIMEOUT, self._add_post_process)
        self.remove_post_timer = Timer(_REMOVE_TIMEOUT, self._remove_post_process, [False, False])
        self.warning = None
        self._reload_index()

    def __len__(self):
        return self.cpu_index.ntotal + self.sub_cpu_index.ntotal

    def _reload_index(self):
        if self.cpu_index_file.is_file():
            self.cpu_index = faiss.read_index(str(self.cpu_index_file))
            logger.info(f'{self.db_name} index 重载完成')
        if self.sub_cpu_index_file.is_file():
            self.sub_cpu_index = faiss.read_index(str(self.sub_cpu_index_file))
            logger.info(f'{self.db_name} sub_index 重载完成')

    def _add_post_process(self):
        """
        添加向量的后处理
        sub_index 内的向量数量大于指定数量后，与主index进行合并，然后持久化保存
        sub_index 内的向量数量未到达指定数量时，对sub_index 进行持久化保存
        :return:
        """
        if self.sub_cpu_index.ntotal >= _SUB_INDEX_SIZE:
            faiss.merge_into(self.cpu_index, self.sub_cpu_index, shift_ids=False)
            if self.sub_cpu_index_file.exists():
                os.remove(str(self.sub_cpu_index_file))
            faiss.write_index(self.cpu_index, str(self.cpu_index_file))
        else:
            faiss.write_index(self.sub_cpu_index, str(self.sub_cpu_index_file))

    def _remove_post_process(self, is_main_modify, is_sub_modify):
        """
        删除向量的后处理操作，对有改动的index 进行向量持久化
        :param is_main_modify:      bool -- 主 index 是否发生变化
        :param is_sub_modify:       bool -- sub index 是否发生变化
        :return:
        """
        if is_main_modify:
            faiss.write_index(self.cpu_index, str(self.cpu_index_file))
        if is_sub_modify:
            faiss.write_index(self.sub_cpu_index, str(self.sub_cpu_index_file))

    def add_id_vector(self, feature_id, vector):
        """
        添加特征向量及对应的id
        :param feature_id:      int --
        :param vector:          list -- [512D]
        :return:
                string -- success or error info
        """
        feature_id = np.array([feature_id])
        vector = np.array([vector]).astype('float32')
        try:
            if self.add_post_timer.is_alive():
                self.add_post_timer.cancel()
            if self.remove_post_timer.is_alive():
                self.remove_post_timer.cancel()
            self.sub_cpu_index.add_with_ids(vector, feature_id)
            self.add_post_timer = Timer(_ADD_TIMEOUT, self._add_post_process)
            self.add_post_timer.start()
            return 'success'
        except Exception as e:
            logger.exception(f'{self.db_name} 添加向量到faiss出现异常')
            return str(e)

    def add_id_vector_batch(self, feature_ids, vectors):
        """
                添加特征向量及对应的id
                :param feature_ids:      int --  [n] id 为 int64
                :param vectors:          list -- [n, dimension]
                :return:
                        string -- success or error info
                """
        feature_id = np.array(feature_ids)
        vector = np.array(vectors).astype('float32')
        try:
            if self.add_post_timer.is_alive():
                self.add_post_timer.cancel()
            if self.remove_post_timer.is_alive():
                self.remove_post_timer.cancel()
            self.sub_cpu_index.add_with_ids(vector, feature_id)
            self.add_post_timer = Timer(_ADD_TIMEOUT, self._add_post_process)
            self.add_post_timer.start()
            return 'success'
        except Exception as e:
            logger.exception(f'{self.db_name} 添加向量到faiss出现异常')
            return str(e)

    def remove_vectors(self, feature_ids):
        """
        根据feature_id移除特征向量
        :param feature_ids:         list --
        :return:
        """
        feature_ids = np.array(feature_ids)
        is_main_modify = False
        is_sub_modify = False
        if self.add_post_timer.is_alive():
            self.add_post_timer.cancel()
        if self.remove_post_timer.is_alive():
            self.remove_post_timer.cancel()
        main_org_total = self.cpu_index.ntotal
        sub_org_total = self.sub_cpu_index.ntotal
        self.cpu_index.remove_ids(feature_ids)
        self.sub_cpu_index.remove_ids(feature_ids)
        if self.cpu_index.ntotal < main_org_total:
            is_main_modify = True
        if self.sub_cpu_index.ntotal < sub_org_total:
            is_sub_modify = True
        self.remove_post_timer = Timer(_REMOVE_TIMEOUT, self._remove_post_process, [is_main_modify, is_sub_modify])
        self.remove_post_timer.start()

    def search(self, vectors, k=20, nprobe=_N_PROBE):
        """
        向量批量搜索
        :param vectors:     list -- [n, 512]
        :param k:           int -- top k, default 20
        :param nprobe:      int -- 进行搜索的聚簇数量， 默认全部搜索
        :return:
                list -- [n, k] feature_ids
                list -- [n, k] similar score
        """
        vectors = np.array(vectors).astype('float32')
        # set nprobe
        self.cpu_index.nprobe = nprobe
        self.sub_cpu_index.nprobe = nprobe

        # search
        main_d, main_i = self.cpu_index.search(vectors, k)
        sub_d, sub_i = self.sub_cpu_index.search(vectors, k)
        # 去除feature_id 为-1的项, 找到第一项为-1的位置，然后进行切片
        minus_main_index = np.where(main_i[0] == -1)[0]
        if len(minus_main_index) != 0:
            main_d = main_d[:, 0:minus_main_index[0]]
            main_i = main_i[:, 0:minus_main_index[0]]
        minus_sub_index = np.where(sub_i[0] == -1)[0]
        if len(minus_sub_index) != 0:
            sub_d = sub_d[:, 0:minus_sub_index[0]]
            sub_i = sub_i[:, 0:minus_sub_index[0]]

        # 搜索结果汇总
        main_d = np.concatenate((main_d, sub_d), axis=1)
        main_i = np.concatenate((main_i, sub_i), axis=1)
        # 按照距离排序，按照k值切片，距离转换为相似度
        if len(main_i) != 0:
            for i in range(len(main_i)):
                order_index = main_d[i].argsort()
                main_i[i] = main_i[i][order_index]
                main_d[i] = main_d[i][order_index]
            if len(main_i[0]) > k:
                main_i = main_i[:, 0:k]
                main_d = main_d[:, 0:k]
            main_d = np.array([_distance_to_score(x) for x in main_d])
        return main_i.tolist(), main_d.tolist()

    def remove_index(self):
        del self.cpu_index
        del self.sub_cpu_index
        if self.cpu_index_file.is_file():
            os.remove(str(self.cpu_index_file))
        if self.sub_cpu_index_file.is_file():
            os.remove(str(self.sub_cpu_index_file))
        logger.info(f'{self.db_name} index 删除完成')


def _distance_to_score(distances):
    """
    将向量之间的欧式距离映射为相似度
    :param distances:  list
    :return: list
    """
    scores = []
    for distance in distances:
        if distance <= 1.2:
            score = 7.0 / (7.0 + distance)
        elif distance < 1.9:
            score = -1 * distance + 2.05
        elif distance >= 1.9:
            score = 0.10
        scores.append(float(f'{score:.4f}'))
    return scores
