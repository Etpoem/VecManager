# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   panjf
# @Time     :   2020/12/21 9:25
# @Desc     :   实现人脸库相关信息记录及更新，人脸库名称，容量，备注信息，是否使用多gpu
# -------------------------------------------------------------
import json
from pathlib import Path

data_path = Path(__file__).parent[1] / 'data'
data_path.mkdir(exist_ok=True)


class DbinfoManager(object):
    def __init__(self):
        self.db_info_file = data_path / 'db_info.json'

    def create_db(self, db_name, max_size, info, is_multiple_gpus=False):
        """
        记录人脸库信息
        :param db_name:     string --
        :param max_size:    int --
        :param info:        string --
        :param is_multiple_gpus: Bool --
        :return:
        """
        if not self.db_info_file.exists():
            with self.db_info_file.open('w') as f:
                db = {
                    db_name: {
                        'db_name': db_name,
                        'max_size': max_size,
                        'info': info,
                        'is_multiple_gpus': is_multiple_gpus
                    }
                }
                json.dump(db, f, indent=4)
        else:
            with self.db_info_file.open('r') as f:
                db = json.load(f)
                db[db_name] = {
                    'db_name': db_name,
                    'max_size': max_size,
                    'info': info,
                    'is_multiple_gpus': is_multiple_gpus
                }
            with self.db_info_file.open('w') as f:
                json.dump(db, f, indent=4)

    def remove_db(self, db_name):
        """
        删除人脸库信息
        :param db_name:     string --
        :return:
        """
        with self.db_info_file.open('r') as f:
            db = json.load(f)
            del db[db_name]
        with self.db_info_file.open('w') as f:
            json.dump(db, f, indent=4)

    def update_db(self, db_name, max_size):
        """
        更新该库最大容量
        :param db_name:     string --
        :param max_size:    int --
        :return:
        """
        with self.db_info_file.open('r') as f:
            db = json.load(f)
            db[db_name]['max_size'] = max_size
        with self.db_info_file.open('w') as f:
            json.dump(db, f, indent=4)

    def get_all_db(self):
        """
        获取人脸库名称和最大容量，用于构建faiss index
        :return:
                    list -- db_names
                    list -- max_sizes
        """
        with self.db_info_file.open('r') as f:
            db = json.load(f)
            db_names = list(db.keys())
            if len(db_names) == 0:
                return [], [], []
            else:
                max_sizes = []
                is_multiple_gpus = []
                for db_name in db_names:
                    max_sizes.append(db[db_name]['max_size'])
                    is_multiple_gpus.append(db[db_name]['is_multiple_gpus'])
                return db_names, max_sizes, is_multiple_gpus

    def get_info(self, db_name):
        """
        获取指定人脸库备注信息
        :param db_name:     string --
        :return:
        """
        with self.db_info_file.open('r') as f:
            db = json.load(f)
        info = db[db_name]['info']
        return info
