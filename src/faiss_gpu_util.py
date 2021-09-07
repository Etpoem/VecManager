# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   panjf
# @Time     :   2020/12/14 16:38
# @Desc     :   GPU 版本 faiss 引擎
# -------------------------------------------------------------
import os
import toml
import faiss
import logging
import pickle
import subprocess
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
_FREE_MEMORY = cfg['faiss']['free_memory']
_N_LIST = cfg['faiss']['nlist']
_N_PROBE = cfg['faiss']['nprobe']
_DIMENSION = cfg['faiss']['dimension']
_GPU_DEVICE = [int(i) for i in cfg['service']['gpu_device'].split(',')]


class FaissManager(object):
    def __init__(self, db_name, max_size, is_multiple_gpus=False):
        """
        :param db_name:         string --
        :param max_size:        int --
        :param is_multiple_gpus:   bool -- 是否使用多gpu
        """
        self.db_name = db_name
        self.max_size = max_size
        self.is_multiple_gpus = is_multiple_gpus
        # 创建 cpu_index
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
        # 选择运行的gpu序号,即选择拥有最大剩余显存的显卡
        gpu_free_memory = free_memory()
        self.gpu_device = gpu_free_memory.index(max(gpu_free_memory))
        # 声明Gpu参数空间，用于设置nprobe
        self.gpu_parameter = faiss.GpuParameterSpace()
        # 创建 gpu_index
        self.gpu_index = _my_index_cpu_to_gpu(self.cpu_index, self.gpu_device, self.is_multiple_gpus)
        self.sub_gpu_index = _my_index_cpu_to_gpu(self.sub_cpu_index, self.gpu_device, False)
        self.gpu_parameter.set_index_parameter(self.sub_gpu_index, 'nprobe', _N_PROBE)
        # 声明持久化文件路径及名称
        self.cpu_index_file = index_file_path / f'{self.db_name}.index'
        self.sub_cpu_index_file = index_file_path / f'sub_{self.db_name}.index'
        # 初始化向量库修改的后处理程序
        self.add_post_timer = Timer(_ADD_TIMEOUT, self._add_post_process)
        self.remove_post_timer = Timer(_REMOVE_TIMEOUT, self._remove_post_process, [False, False])
        self.warning = None
        self._reload_index()

    def __len__(self):
        return self.cpu_index.ntotal + self.sub_gpu_index.ntotal

    def _reload_index(self):
        if self.cpu_index_file.is_file():
            self.cpu_index = faiss.read_index(str(self.cpu_index_file))
            self.gpu_index = _my_index_cpu_to_gpu(self.cpu_index, self.gpu_device, self.is_multiple_gpus)
            logger.info(f'{self.db_name} index 重载完成')
        if self.sub_cpu_index_file.is_file():
            self.sub_cpu_index = faiss.read_index(str(self.sub_cpu_index_file))
            self.sub_gpu_index = _my_index_cpu_to_gpu(self.sub_cpu_index, self.gpu_device, False)

    def _add_post_process(self):
        """
        添加向量后的后处理
        sub_index数量大于指定数量后，与主index进行合并，
        然后载入gpu_index, 并进行向量持久化，删除sub_cpu_index_file文件；
        sub_index未到达指定数量数量时，对sub_index进行持久化
        :return:
        """
        if self.sub_gpu_index.ntotal >= _SUB_INDEX_SIZE:
            self.sub_cpu_index = faiss.index_gpu_to_cpu(self.sub_gpu_index)
            self.sub_gpu_index.reset()
            faiss.merge_into(self.cpu_index, self.sub_cpu_index, shift_ids=False)
            if self.sub_cpu_index_file.exists():
                os.remove(str(self.sub_cpu_index_file))
            faiss.write_index(self.cpu_index, str(self.cpu_index_file))
            del self.gpu_index
            self.gpu_index = _my_index_cpu_to_gpu(self.cpu_index, self.gpu_device, self.is_multiple_gpus)
            # self.sub_gpu_index.reset()
            if free_memory(self.gpu_device) < _FREE_MEMORY:
                self.warning = f'GPU显存不足，需要预留{_FREE_MEMORY}MiB 作为计算空间'
        else:
            self.sub_cpu_index = faiss.index_gpu_to_cpu(self.sub_gpu_index)
            faiss.write_index(self.sub_cpu_index, str(self.sub_cpu_index_file))

    def _remove_post_process(self, is_main_modify, is_sub_modify):
        """
        删除向量的后处理操作， 包括向量持久化,以及载入GPU
        :param is_main_modify:      bool -- 主库是否发生变化
        :param is_sub_modify:       bool -- 子库是否发生变化
        :return:
        """
        if is_main_modify:
            faiss.write_index(self.cpu_index, str(self.cpu_index_file))
            self.gpu_index = _my_index_cpu_to_gpu(self.cpu_index, self.gpu_device, self.is_multiple_gpus)
        if is_sub_modify:
            faiss.write_index(self.sub_cpu_index, str(self.sub_cpu_index_file))
            self.sub_gpu_index = _my_index_cpu_to_gpu(self.sub_cpu_index, self.gpu_device, False)

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
        if self.warning is None:
            try:
                if self.add_post_timer.is_alive():
                    self.add_post_timer.cancel()
                self.sub_gpu_index.add_with_ids(vector, feature_id)
                if self.remove_post_timer.is_alive():
                    # 这里是因为remove的后处理会把cpu_index reload 到 gpu_index
                    self.sub_cpu_index.add_with_ids(vector, feature_id)
                self.add_post_timer = Timer(_ADD_TIMEOUT, self._add_post_process)
                self.add_post_timer.start()
                return 'success'
            except Exception as e:
                logger.exception(f'{self.db_name} 添加向量到faiss 出现一异常')
                return str(e)
        else:
            if free_memory(self.gpu_device) >= _FREE_MEMORY:
                self.warning = None
                self.sub_gpu_index.add_with_ids(vector, feature_id)
                return 'success'
            else:
                return self.warning

    def add_id_vector_batch(self, feature_ids, vectors):
        """
        批量添加特征向量及对应的id
        :param feature_ids:     list -- [n] id 为 int64
        :param vectors:         list -- [n, dimension]
        :return:
                    string -- success or error info
        """
        feature_ids = np.array(feature_ids)
        vectors = np.array(vectors).astype('float32')
        if self.warning is None:
            try:
                if self.add_post_timer.is_alive():
                    self.add_post_timer.cancel()
                self.sub_gpu_index.add_with_ids(vectors, feature_ids)
                if self.remove_post_timer.is_alive():
                    self.sub_cpu_index.add_with_ids(vectors, feature_ids)
                self.add_post_timer = Timer(_ADD_TIMEOUT, self._add_post_process)
                self.add_post_timer.start()
                return 'success'
            except Exception as e:
                logger.exception('添加向量到faiss出现异常')
                return str(e)
        else:
            if free_memory(self.gpu_device) >= _FREE_MEMORY:
                self.warning = None
                self.sub_gpu_index.add_with_ids(vectors, feature_ids)
                return 'success'
            else:
                return self.warning

    def remove_vectors(self, feature_ids):
        """
        根据feature_id移除特征向量
        :param feature_ids:         list --
        :return:
        """
        feature_ids = np.array(feature_ids)
        is_main_modify = False
        is_sub_modify = True        # 实践发现，删除后 ntotal 貌似不会立即发生变化
        if self.remove_post_timer.is_alive():
            self.remove_post_timer.cancel()
        self.sub_cpu_index = faiss.index_gpu_to_cpu(self.sub_gpu_index)
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
        向量批量搜索，
        :param vectors:     list -- [n, 512]
        :param k:           int -- top k, default 20
        :param nprobe:      int -- 进行搜索的聚簇数量， 默认全部搜索
        :return:
                list -- [n, k] feature_ids
                list -- [n, k] similar score
        """
        vectors = np.array(vectors).astype('float32')
        # set nprobe
        self.gpu_parameter.set_index_parameter(self.gpu_index, 'nprobe', nprobe)

        main_d, main_i = self.gpu_index.search(vectors, k)
        sub_d, sub_i = self.sub_gpu_index.search(vectors, k)
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
        del self.gpu_index
        del self.sub_gpu_index
        del self.cpu_index
        del self.sub_cpu_index
        if self.sub_cpu_index_file.is_file():
            os.remove(str(self.sub_cpu_index_file))
        if self.cpu_index_file.is_file():
            os.remove(str(self.cpu_index_file))
        logger.info(f'{self.db_name} index 删除完成')


def _my_index_cpu_to_gpu(cpu_index, gpu_device=0, is_multiple_gpus=False):
    """
    index 加载到gpu，支持多卡分布加载
    :param cpu_index:           faiss index object --
    :param gpu_device:          int -- 加载的gpu卡序号，当使用多卡时，向量数据会分布到所有的卡，此项无效
    :param is_multiple_gpus:    bool -- 是否采用多卡分布
    :return:
                faiss gpu index object
    """
    if is_multiple_gpus:
        gpu_number = faiss.get_num_gpus()
        clone_option = faiss.GpuMultipleClonerOptions()
        clone_option.shard = True
        clone_option.useFloat16 = True
        resources = [faiss.StandardGpuResources() for i in range(gpu_number)]
        for res in resources:
            res.noTempMemory()
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        for i, res in zip(range(gpu_number), resources):
            vdev.push_back(i)
            vres.push_back(res)
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, cpu_index, clone_option)
        gpu_index.referenced_objects = resources
    else:
        res = faiss.StandardGpuResources()
        res.noTempMemory()
        clone_option = faiss.GpuClonerOptions()
        clone_option.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_device, cpu_index, clone_option)

    return gpu_index


def free_memory(gpu_device=None):
    """
    开启一个子进程， 通过‘nvidia-smi'命令获取剩余显存
    :param: gpu_device      int -- gpu设备序号，指faiss内的序号而不是真正物理机上的序号
    :return:
            list or int -- 剩余显存
    """
    command = 'nvidia-smi --query-gpu=memory.free --format=csv'
    ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, encoding='ascii')
    all_free_values = [int(x.split()[0]) for x in ret.stdout.split('\n')[1:-1]]
    free_value_choose = [all_free_values[i] for i in _GPU_DEVICE]
    if gpu_device is not None:
        return free_value_choose[gpu_device]
    else:
        return free_value_choose


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
