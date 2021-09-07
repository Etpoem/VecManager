# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   panjf
# @Time     :   2021/5/12 14:16
# @Desc     :   
# -------------------------------------------------------------
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import numpy as np
from faiss_util import FaissManager


def faiss_search_test():
    db_index = FaissManager('cb', 5000000, False)

    print('开始生成数据')
    feature_ids = range(0, 2500000)
    vectors = np.random.random([2500000, 512]).astype('float32')
    print('数据生成完成')

    print('添加')
    db_index.add_id_vector_batch(feature_ids, vectors)
    for i in range(10):
        time.sleep(1)
        print(i)
    del feature_ids
    del vectors
    print(f'size : {len(db_index)}')

    for i in range(100):
        since = time.time()
        search_vector = np.random.random([2, 512]).astype('float32')
        use = time.time() - since
        print(f'use time {use * 1000} ms')


if __name__ == "__main__":
    faiss_search_test()
