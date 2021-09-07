# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   panjf
# @Time     :   2020/12/18 16:22
# @Desc     :   Mysql 数据库操作代码
# -------------------------------------------------------------
import toml
import json
import time
import pymysql
import logging
from DBUtils.PersistentDB import PersistentDB
from pathlib import Path

logger = logging.getLogger('main.mysql')
cfg = toml.load(Path(__file__).parents[1] / 'config.toml')

_HOST = cfg['mysql']['host']
_PORT = cfg['mysql']['port']
_DATABASE = cfg['mysql']['database']


class MysqlManager(object):
    def __init__(self):
        self.pool = PersistentDB(creator=pymysql,
                                 ping=0,
                                 host=_HOST,
                                 port=_PORT,
                                 user=cfg['mysql']['user'],
                                 password=cfg['mysql']['password'])
        self._connect_check()
        del self.pool
        self.pool = PersistentDB(creator=pymysql,
                                 ping=0,
                                 host=_HOST,
                                 port=_PORT,
                                 user=cfg['mysql']['user'],
                                 password=cfg['mysql']['password'],
                                 database=_DATABASE)
        self.table_prefix = cfg['mysql']['table_prefix']
        # self.main_table = cfg['mysql']['main_table']
        logger.info(f'MySQL internal 初始化完成 ，表名前缀 {self.table_prefix}')

    def _connect_check(self):
        """
        检查 MySQL 服务端是否可用，连接失败后进行多次重连接，睡眠时间递增
        :return:
        """
        retry_count = 5
        sleep_time = 10
        for i in range(retry_count):
            if i < retry_count - 1:
                try:
                    conn = self.pool.connection()
                    cursor = conn.cursor()
                    cursor.execute("CREATE DATABASE IF NOT EXISTS {}".format(_DATABASE))
                    cursor.execute("USE {}".format(_DATABASE))
                    conn.commit()
                    conn.close()
                    logger.info(f'MySQL {_HOST}:{_PORT}/{_DATABASE} 连接成功！')
                    break
                except Exception as e:
                    logger.exception(f'MySQL {_HOST}:{_PORT}/{_DATABASE} 连接失败')
                    logger.error(f'{sleep_time} 秒之后进行重新连接')
                    time.sleep(sleep_time)
                    sleep_time += 10
            else:
                conn = self.pool.connection()
                conn.close()

    def create_table(self, db_name):
        """
        :param db_name:     string --
        :return:
                if success:
                            string -- 'success'
                else:
                            string -- error info
        """
        table_name = f'{self.table_prefix}{db_name}'
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE {} ("
                           "face_id BIGINT PRIMARY KEY,"
                           "face_vector BLOB NOT NULL,"
                           "face_coord BLOB NOT NULL,"
                           "face_info BLOB ,"
                           "create_datetime DATETIME NOT NULL)"
                           "ENGINE=MyISAM DEFAULT CHARSET=utf8".format(table_name))
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f'人脸信息表 {table_name} 创建成功！')
            return 'success'
        except Exception as e:
            logger.exception(f'人脸信息表 {table_name} 创建失败！')
            return str(e)

    def drop_table(self, db_name):
        """
        :param db_name:     string --
        :return:
                if success:
                            string -- 'success'
                else:
                            string -- error info
        """
        table_name = f'{self.table_prefix}{db_name}'
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            cursor.execute("DROP TABLE {}".format(table_name))
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f'人脸信息表 {table_name} 删除完成')
            return 'success'
        except Exception as e:
            logger.exception(f'人脸信息表 {table_name} 删除失败')
            return str(e)

    def insert_data(self, db_name, face_id, face_vector, face_coord, face_info, create_datetime):
        """
        :param db_name:         string --
        :param face_id:         int --
        :param face_vector:     list -- [512]
        :param face_coord:      list -- [x1, y1, x2, y2]
        :param face_info:       string --
        :param create_datetime:     datetime.datetime -- format '%Y-%m-%d %H:%M:%S'
        :return:
                if success:
                            string -- 'success'
                else:
                            string -- error info
        """
        table_name = f'{self.table_prefix}{db_name}'
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            face_vector = json.dumps(face_vector)
            face_coord = json.dumps(face_coord)
            cursor.execute("INSERT INTO {} "
                           "(face_id, face_vector, face_coord, face_info, create_datetime) "
                           "VALUES (%s, %s, %s, %s, %s)".format(table_name),
                           (face_id, face_vector, face_coord, face_info, create_datetime))
            conn.commit()
            cursor.close()
            conn.close()
            return 'success'
        except Exception as e:
            conn.rollback()
            logger.exception(f'face_id:{face_id} 人脸信息插入失败')
            return str(e)

    def insert_data_batch(self, db_name, face_data):
        """
        :param db_name:     string
        :param face_data:   list -- [(face_id, face_vector, face_coord, face_info, create_datetime), ...]
        :return:
                string -- 'success' or error info
        """
        table_name = f'{self.table_prefix}{db_name}'
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            sql = "INSERT INTO {} " \
                  "(face_id, face_vector, face_coord, face_info, create_datetime) " \
                  "VALUES (%s, %s, %s, %s, %s)".format(table_name)
            cursor.executemany(sql, face_data)
            conn.commit()
            cursor.close()
            conn.close()
            return 'success'
        except Exception as e:
            conn.rollback()
            logger.exception(f'{db_name} 表 插入数据发生异常')
            return str(e)

    def delete_data(self, db_name, face_id):
        """
        :param db_name:     string --
        :param face_id:     int --
        :return:
                if success:
                            string -- 'success'
                else:
                            string -- error info
        """
        table_name = f'{self.table_prefix}{db_name}'
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            ret = cursor.execute("DELETE FROM {} "
                                 "WHERE face_id=%s".format(table_name), (face_id,))
            conn.commit()
            cursor.close()
            conn.close()
            if ret == 0:
                return f'feature_id: {face_id} 不存在'
            else:
                return 'success'
        except Exception as e:
            conn.rollback()
            logger.exception(f'feature_id: {face_id} 删除失败')
            return str(e)

    def delete_data_batch(self, db_name, face_ids):
        """
        :param db_name:         string --
        :param face_ids:        list --
        :return:
                string -- 'success' or error info
        """
        table_name = f'{self.table_prefix}{db_name}'
        face_ids_str = ','.join([str(i) for i in face_ids])
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM {0} "
                           "WHERE face_id in ({1})".format(table_name, face_ids_str))
            conn.commit()
            cursor.close()
            conn.close()
            return 'success'
        except Exception as e:
            conn.rollback()
            logger.exception(f'{table_name}表 删除 ({face_ids_str}) 发生异常')
            return str(e)

    def select_coord_info(self, db_name, face_ids):
        """
        :param db_name:         string --
        :param face_ids:         list --
        :return:
                dict -- {
                            str(feature_id) : {
                                                'feature_id' :  int,
                                                'faceCoord':    list,
                                                'info'          string }
                            ......
                        }
        """
        table_name = f'{self.table_prefix}{db_name}'
        face_ids_str = ','.join([str(i) for i in face_ids])
        face_ids.insert(0, 'face_id')
        fiels_str = ','.join([str(i) for i in face_ids])
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            cursor.execute("SELECT face_id, face_coord, face_info "
                           "FROM {0} "
                           "WHERE face_id in ({1}) "
                           "ORDER BY FIELD({2})".format(table_name, face_ids_str, fiels_str))
            result_dict = {}
            for data in cursor.fetchall():
                result_dict[str(data[0])] = {
                    'feature_id': int(data[0]),
                    'faceCoord': json.loads(data[1]),
                    'info': str(data[2], encoding='utf-8')
                }
            cursor.close()
            conn.close()
            return result_dict
        except Exception as e:
            logger.error(f'获取 {face_ids} 信息失败')
            logger.error(str(e))

    def select_vector_coord_info(self, db_name, face_id):
        """
        :param db_name:     string --
        :param face_id:     int --
        :return:
                list -- face_vector
                list -- face_coord
                string -- face_info
        """
        table_name = f'{self.table_prefix}{db_name}'
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            cursor.execute("SELECT face_vector, face_coord, face_info "
                           "FROM {} "
                           "WHERE face_id=%s".format(table_name), (face_id,))
            data = cursor.fetchone()
            cursor.close()
            conn.close()
            if data is not None:
                face_vector = json.loads(data[0])
                face_coord = json.loads(data[1])
                face_info = str(data[2], encoding='utf-8')
                return 'success', face_vector, face_coord, face_info
            else:
                error_info = f'{db_name}: {face_id} 不存在'
                return error_info, '', '', ''
        except Exception as e:
            logger.exception(f'获取{db_name}:{face_id}人脸信息失败')
            return str(e), '', '', ''

    def get_id_by_date(self, db_name, begin_date, end_date):
        """
        获取指定时间段内的人脸特征id
        :param db_name:         string --
        :param begin_date:      datetime object -- format '%Y-%m-%d %H:%M:%S'
        :param end_date:        datetime object -- format '%Y-%m-%d %H:%M:%S'
        :return:
                list -- face_ids
        """
        table_name = f'{self.table_prefix}{db_name}'
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            cursor.execute("SELECT face_id "
                           "FROM {} "
                           "WHERE create_datetime BETWEEN %s AND %s".format(table_name), (begin_date, end_date))
            face_ids = [int(face_id[0]) for face_id in cursor.fetchall()]
            cursor.close()
            conn.close()
            return face_ids
        except Exception as e:
            logger.exception(f'从人脸信息表{db_name}获取{begin_date}到{end_date}的feature_id 失败')

    def delete_by_date(self, db_name, begin_date, end_date):
        """
        删除指定人脸库特定时间段的数据
        :param db_name:         string --
        :param begin_date:      datetime object -- format '%Y-%m-%d %H:%M:%S'
        :param end_date:        datetime object -- format '%Y-%m-%d %H:%M:%S'
        :return:
                if success:
                            string -- 'success'
                else:
                            string -- error info
        """
        table_name = f'{self.table_prefix}{db_name}'
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            ret = cursor.execute("DELETE FROM {} "
                                 "WHERE create_datetime BETWEEN %s AND %s".format(table_name), (begin_date, end_date))
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f'人脸库{db_name},删除了{begin_date}到{end_date}内共{ret}条记录')
            return 'success'
        except Exception as e:
            conn.rollback()
            logger.exception(f'{db_name} 删除{begin_date}到{end_date}数据发生错误')
            return str(e)

    # def main_table_delete_by_date(self, begin_date, end_date):
    #     """
    #     删除行人库主表特定时间段的数据
    #     :param begin_date:      datetime object -- format '%Y-%m-%d %H:%M:%S'
    #     :param end_date:        datetime object -- format '%Y-%m-%d %H:%M:%S'
    #     :return:
    #             if success:
    #                         string -- 'success'
    #             else:
    #                         string -- error info
    #     """
    #     try:
    #         conn = self.pool.connection()
    #         cursor = conn.cursor()
    #         ret = cursor.execute("DELETE FROM {} "
    #                              "WHERE create_datetime BETWEEN %s AND %s".format(self.main_table), (begin_date, end_date))
    #         conn.commit()
    #         cursor.close()
    #         conn.close()
    #         logger.info(f'删除行人库主表{self.main_table},从{begin_date} 到 {end_date} 内共{ret}条记录')
    #         return 'success'
    #     except Exception as e:
    #         conn.rollback()
    #         logger.exeception(f'删除行人库主表{self.main_table}, {begin_date}-{end_date} 数据发生异常')
    #         return str(e)

    # def select_person_ids(self, face_ids):
    #     """
    #     获取 face_ids 对应的person_id
    #     :param face_ids:    list --
    #     :return:
    #         dict -- {
    #                     str(feature_id): int(person_id),
    #                     ...
    #                 }
    #     """
    #     face_ids_str = ','.join([str(i) for i in face_ids])
    #     try:
    #         conn = self.pool.connection()
    #         cursor = conn.cursor()
    #         cursor.execute("SELECT face_id, person_id "
    #                        "FROM {} "
    #                        "WHERE face_id in ({})".format(self.main_table, face_ids_str))
    #         result_dict = {}
    #         for data in cursor.fetchall():
    #             result_dict[str(data[0])] = int(data[1])
    #         cursor.close()
    #         conn.close()
    #         return result_dict
    #     except Exception as e:
    #         logger.exception(f'获取 {face_ids} 的person_id 信息失败')

    # 获取人脸库所有ID
    def get_all_face_id(self, db_name):
        """
        获取指定人脸库下得所有 face_id
        :param db_name:     string --
        :return:
                list -- face_id list
        """
        table_name = f'{self.table_prefix}{db_name}'
        try:
            conn = self.pool.connection()
            cursor = conn.cursor()
            cursor.execute("SELECT face_id "
                           "FROM {}".format(table_name))
            face_ids = [int(face_id[0]) for face_id in cursor.fetchall()]
            cursor.close()
            conn.close()
            return face_ids
        except Exception as e:
            logger.exception(f'获取 face_id 失败')

