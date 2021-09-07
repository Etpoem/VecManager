# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   panjf
# @Time     :   2021/2/19 15:41
# @Desc     :   
# -------------------------------------------------------------
import os
import toml
config = toml.load('config.toml')
os.environ["CUDA_VISIBLE_DEVICES"] = config['service']['gpu_device']
import json
from flask import Flask, request, jsonify
from waitress import serve
from marshmallow import ValidationError

import val
from db_info_util import DbinfoManager
if config['service']['device'] == 'gpu':
    from faiss_gpu_util import FaissManager
else:
    from faiss_cpu_util import FaissManager
from mysql_util import MysqlManager
from log_util import setup_logger


logger = setup_logger(name='main')


class VectorSearchServer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.mysql_manager = MysqlManager()
        self.db_info_manager = DbinfoManager()

        self.flask_host = self.cfg['flask']['host']
        self.flask_port = self.cfg['flask']['port']

        # ------------------------ 创建默认库/恢复已存在的人脸库 -----------------------
        self.indexs = {}
        if not self.db_info_manager.db_info_file.exists():
            # 创建默认的人脸库
            default_db_names = cfg['service']['default_db_names']
            default_max_size = cfg['service']['default_max_size']
            default_db_info = cfg['service']['default_db_info']
            default_is_multiple_gpus = cfg['service']['default_is_multiple_gpus']
            for db_name, max_size, info, is_multiple_gpus in zip(default_db_names, default_max_size, default_db_info,
                                                                 default_is_multiple_gpus):
                self.indexs[db_name] = FaissManager(db_name, max_size, is_multiple_gpus)
                self.mysql_manager.create_table(db_name)
                self.db_info_manager.create_db(db_name, max_size, info, is_multiple_gpus)
            logger.info(f'默认人脸库 {default_db_names} 创建完成')
            del default_db_names
            del default_max_size
            del default_db_info
            del default_is_multiple_gpus
        else:
            pre_db_names, pre_max_size, pre_is_multiple_gpus = self.db_info_manager.get_all_db()
            if len(pre_db_names) != 0:
                for db_name, max_size, is_multiple_gpus in zip(pre_db_names, pre_max_size, pre_is_multiple_gpus):
                    self.indexs[db_name] = FaissManager(db_name, max_size, is_multiple_gpus)
                logger.info(f'{pre_db_names} 人脸库恢复完成')
            del pre_db_names
            del pre_max_size
            del pre_is_multiple_gpus
        self.db_names, _, _ = self.db_info_manager.get_all_db()

    def flask_run(self):
        app = Flask(__name__)

        @app.route('/faiss/db/create', methods=["POST"])
        def db_create():
            ret_info = {
                'result': [],
                'error': ''
            }
            # --------------参数校验-----------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbCreate().load(receive_param)
                db_name = receive_param['db_name']
                max_size = receive_param['max_size']
                info = receive_param['info']
                is_multiple_gpus = receive_param['is_multiple_gpus']
            except ValidationError as err:
                ret_info["error"] = str(err.messages)
                return jsonify(ret_info)
            # -----------------------------------------
            if db_name not in self.db_names:
                self.indexs[db_name] = FaissManager(db_name, max_size, is_multiple_gpus)
                self.mysql_manager.create_table(db_name)
                self.db_names.append(db_name)
                self.db_info_manager.create_db(db_name, max_size, info, is_multiple_gpus)
                result = {
                    'db_name': db_name,
                    'size': 0,
                    'max_size': max_size,
                    'info': info,
                    'is_multiple_gpus': is_multiple_gpus
                }
                ret_info['result'].append(result)
                logger.info(f'人脸库{db_name}创建完成')
            else:
                ret_info['error'] = f'人脸库 {db_name} 已经存在'

            return jsonify(ret_info)

        @app.route('/faiss/db/remove', methods=["POST"])
        def db_remove():
            ret_info = {
                'result': [],
                'error': ''
            }
            # ----------------参数校验-------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbRemove().load(receive_param)
                db_name = receive_param['db_name']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # --------------------------------------
            if db_name in self.db_names:
                result = {
                    'db_name': db_name,
                    'size': len(self.indexs[db_name]),
                    'max_size': self.indexs[db_name].max_size,
                    'info': self.db_info_manager.get_info(db_name)
                }
                self.mysql_manager.drop_table(db_name)
                self.indexs[db_name].remove_index()
                del self.indexs[db_name]
                self.db_info_manager.remove_db(db_name)
                self.db_names.remove(db_name)
                ret_info['result'].append(result)
                logger.info(f'人脸库 {db_name} 删除成功')
            else:
                ret_info['error'] = f'人脸库 {db_name} 不存在'

            return jsonify(ret_info)

        @app.route('/faiss/db/update', methods=["POST"])
        def db_update():
            ret_info = {
                'result': [],
                'error': ''
            }
            # ----------------参数校验-----------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbUpdate().load(receive_param)
                db_name = receive_param['db_name']
                max_size = receive_param['max_size']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ------------------------------------------
            if db_name in self.db_names:
                self.db_info_manager.update_db(db_name, max_size)
                self.indexs[db_name].max_size = max_size
                result = {
                    'db_name': db_name,
                    'size': len(self.indexs[db_name]),
                    'max_size': max_size,
                    'info': self.db_info_manager.get_info(db_name)
                }
                ret_info['result'].append(result)
                logger.info(f'人脸库 {db_name} 更新完成')
            else:
                ret_info['error'] = f'人脸库 {db_name} 不存在'

            return jsonify(ret_info)

        @app.route('/faiss/db/info', methods=["POST"])
        def db_info():
            ret_info = {
                'result': [],
                'error': ''
            }
            # --------------参数校验-------------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbInfo().load(receive_param)
                db_name = receive_param['db_name']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ------------------------------------------
            if db_name == '':
                result = [
                    {
                        'db_name': _db_name,
                        'size': len(self.indexs[_db_name]),
                        'max_size': self.indexs[_db_name].max_size,
                        'info': self.db_info_manager.get_info(_db_name),
                        'is_multiple_gpus': self.indexs[_db_name].is_multiple_gpus
                    }
                    for _db_name in self.db_names
                ]
                ret_info['result'] = result
                return jsonify(ret_info)
            elif db_name in self.db_names:
                result = {
                    'db_name': db_name,
                    'size': len(self.indexs[db_name]),
                    'max_size': self.indexs[db_name].max_size,
                    'info': self.db_info_manager.get_info(db_name),
                    'is_multiple_gpus': self.indexs[db_name].is_multiple_gpus
                }
                ret_info['result'].append(result)
                return jsonify(ret_info)
            else:
                ret_info['error'] = f'人脸库 {db_name} 不存在'
                return jsonify(ret_info)

        @app.route('/faiss/feature/delete', methods=["POST", "DELETE"])
        def face_delete():
            ret_info = {
                'result': [],
                'error': ''
            }
            # ---------------------参数校验-----------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbDelete().load(receive_param)
                db_name = receive_param['db_name']
                feature_id = receive_param['feature_id']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ----------------------------------------------
            if db_name in self.db_names:
                select_ret, face_vector, face_coord, face_info = self.mysql_manager.select_vector_coord_info(db_name,
                                                                                                             feature_id)
                if select_ret == 'success':
                    result = {
                        'feature_id': feature_id,
                        'feature': json.dumps(face_vector),
                        'faceCoord': face_coord,
                        'info': face_info
                    }
                    self.mysql_manager.delete_data(db_name, feature_id)
                    self.indexs[db_name].remove_vectors([feature_id])
                    ret_info['result'].append(result)
                    return jsonify(ret_info)
                else:
                    ret_info['error'] = select_ret
                    return jsonify(ret_info)
            else:
                ret_info['error'] = f'人脸库{db_name}不存在'
                return jsonify(ret_info)

        @app.route('/faiss/feature/delete_by_date', methods=["POST", "DELETE"])
        def face_delete_by_date():
            ret_info = {
                'result': [],
                'error': ''
            }
            # -------------参数校验-----------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbDeleteByDate().load(receive_param)
                db_name = receive_param['db_name']
                begin_time = receive_param['begin_time']
                end_time = receive_param['end_time']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ----------------------------------------
            if db_name in self.db_names:
                try:
                    passed_ids = self.mysql_manager.get_id_by_date(db_name, begin_time, end_time)
                    if len(passed_ids) != 0:
                        self.indexs[db_name].remove_vectors(passed_ids)
                        self.mysql_manager.delete_by_date(db_name, begin_time, end_time)
                        result = {
                            'total': len(passed_ids),
                            'feature_ids': passed_ids
                        }
                    else:
                        result = {
                            'total': 0,
                            'feature_ids': []
                        }
                    ret_info['result'].append(result)
                    return jsonify(ret_info)
                except Exception as e:
                    ret_info['error'] = str(e)
                    return jsonify(ret_info)
            else:
                ret_info['error'] = f'人脸库{db_name}不存在'
                return jsonify(ret_info)

        @app.route('/faiss/feature/delete_batch', methods=["POST", "DELETE"])
        def face_delete_batch():
            ret_info = {
                'error': ''
            }
            # ----------------------参数校验--------------------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbDeleteBatch().load(receive_param)
                db_name = receive_param['db_name']
                feature_ids = receive_param['feature_ids']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # --------------------------------------------------------
            try:
                mysql_ret = self.mysql_manager.delete_data_batch(db_name, feature_ids)
                if mysql_ret == 'success':
                    faiss_ret = self.indexs[db_name].remove_vectors(feature_ids)
                    if faiss_ret == 'success':
                        return jsonify(ret_info)
                    else:
                        ret_info['error'] = faiss_ret
                        return jsonify(ret_info)
                else:
                    ret_info['error'] = mysql_ret
                    return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        @app.route('/faiss/feature/get', methods=["POST"])
        def db_get():
            ret_info = {
                'result': [],
                'error': ''
            }
            # ---------------------参数校验---------------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbGet().load(receive_param)
                db_name = receive_param['db_name']
                feature_id = receive_param['feature_id']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ---------------------------------------------------
            if db_name in self.db_names:
                select_ret, face_vector, face_coord, face_info = self.mysql_manager.select_vector_coord_info(db_name,
                                                                                                             feature_id)
                if select_ret == 'success':
                    result = {
                        'feature_id': feature_id,
                        'feature': json.dumps(face_vector),
                        'faceCoord': face_coord,
                        'info': face_info
                    }
                    ret_info['result'].append(result)
                    return jsonify(ret_info)
                else:
                    ret_info['error'] = select_ret
                    return jsonify(ret_info)
            else:
                ret_info['error'] = f'人脸库{db_name}不存在'
                return jsonify(ret_info)

        @app.route('/faiss/feature/insert', methods=["POST"])
        def insert_feature():
            ret_info = {
                'result': [],
                'error': ''
            }
            # -----------------------参数校验--------------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbInsertFeature().load(receive_param)
                db_name = receive_param['db_name']
                feature_id = receive_param['feature_id']
                feature = json.loads(receive_param['feature'])
                face_coord = receive_param['faceCoord']
                info = receive_param['info']
                create_datetime = receive_param['create_datetime']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ---------------------------------------------------------
            if db_name in self.db_names:
                if len(self.indexs[db_name]) < self.indexs[db_name].max_size:
                    mysql_ret = self.mysql_manager.insert_data(db_name, feature_id, feature, face_coord, info,
                                                               create_datetime)
                    if mysql_ret == 'success':
                        insert_ret = self.indexs[db_name].add_id_vector(feature_id, feature)
                        if insert_ret == 'success':
                            result = {
                                'feature_id': feature_id,
                                'feature': json.dumps(feature),
                                'faceCoord': face_coord,
                                'info': info
                            }
                            ret_info['result'].append(result)
                            return jsonify(ret_info)
                        else:
                            ret_info['error'] = insert_ret
                            return jsonify(ret_info)
                    else:
                        ret_info['error'] = mysql_ret
                        return jsonify(ret_info)
                else:
                    ret_info['error'] = f'人脸库{db_name}容量已满，人脸添加失败'
                    return jsonify(ret_info)
            else:
                ret_info['error'] = f'人脸库 {db_name} 不存在'
                return jsonify(ret_info)

        @app.route('/faiss/feature/insert_batch', methods=['POST'])
        def insert_feature_batch():
            ret_info = {
                'error': ''
            }
            # -------------------参数校验-----------------------------
            receive_param = request.get_json()
            try:
                db_name = receive_param['db_name']
                face_data = receive_param['face_data']
                """
                face_data is a list type
                format:
                        [(face_id, face_vector, face_coord, face_info, create_datetime)]
                """
                face_ids_list = [int(data[0]) for data in face_data]
                face_vector_list = [json.loads(data[1]) for data in face_data]
            except Exception as err:
                ret_info['error'] = str(err)
                return jsonify(ret_info)
            # -------------------------------------------------------------
            try:
                # 将人脸信息存入mysql表
                mysql_ret = self.mysql_manager.insert_data_batch(db_name, face_data)
                if mysql_ret == 'success':
                    faiss_ret = self.indexs[db_name].add_id_vector_batch(face_ids_list, face_vector_list)
                    if faiss_ret == 'success':
                        return jsonify(ret_info)
                    else:
                        ret_info['error'] = faiss_ret
                        return jsonify(ret_info)
                else:
                    ret_info['error'] = mysql_ret
                    return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return jsonify(ret_info)

        @app.route('/faiss/feature/search', methods=["POST"])
        def search_feature():
            ret_info = {
                'result': [],
                'error': ''
            }
            # -----------------参数校验------------------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbSearchFeature().load(receive_param)
                db_name = receive_param['db_name']
                feature = json.loads(receive_param['feature'])
                top = receive_param['top']
                nprobe = receive_param['nprobe']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ---------------------------------------------------
            if db_name in self.db_names:
                face_ids, scores = self.indexs[db_name].search([feature], top, nprobe)
                face_ids, scores = face_ids[0], scores[0]
                if len(face_ids) != 0:
                    id_score_dict = {}
                    for i, s in zip(face_ids, scores):
                        id_score_dict[str(i)] = s
                    result_dict = self.mysql_manager.select_coord_info(db_name, face_ids)
                    for key in result_dict.keys():
                        result_dict[key]['score'] = id_score_dict[key]
                    for value in result_dict.values():
                        ret_info['result'].append(value)
                    return jsonify(ret_info)
                else:
                    return jsonify(ret_info)
            else:
                ret_info['error'] = f'人脸库 {db_name} 不存在'
                return jsonify(ret_info)

        @app.route('/faiss/feature/search_fast', methods=["POST"])
        def search_feature_fast():
            ret_info = {
                'result': [],
                'error': ''
            }
            # -----------------参数校验------------------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbSearchFeatureFast().load(receive_param)
                db_name = receive_param['db_name']
                feature = json.loads(receive_param['feature'])
                top = receive_param['top']
                nprobe = receive_param['nprobe']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # ---------------------------------------------------
            if db_name in self.db_names:
                face_ids, scores = self.indexs[db_name].search([feature], top, nprobe)
                face_ids, scores = face_ids[0], scores[0]
                if len(face_ids) != 0:
                    for face_id, score in zip(face_ids, scores):
                        ret_info['result'].append(
                            {
                                'feature_id': face_id,
                                'score': score
                            }
                        )
                    return jsonify(ret_info)
                else:
                    return jsonify(ret_info)
            else:
                ret_info['error'] = f'人脸库 {db_name} 不存在'
                return jsonify(ret_info)

        @app.route('/faiss/feature/search_fast_batch', methods=['POST'])
        def search_feature_fast_batch():
            ret_info = {
                'result': [],
                'error': ''
            }
            # --------------------参数校验--------------------------
            receive_param = request.get_json()
            try:
                receive_param = val.DbSearchFeatureFastBatch().load(receive_param)
                db_name = receive_param['db_name']
                features = json.loads(receive_param['features'])
                top = receive_param['top']
                nprobe = receive_param['nprobe']
            except ValidationError as err:
                ret_info['error'] = str(err.messages)
                return jsonify(ret_info)
            # --------------------------------------------------------
            try:
                face_ids, scores = self.indexs[db_name].search(features, top, nprobe)
                ret_info['result'].append(
                    {
                        'feature_ids': face_ids,
                        'scores': scores
                    }
                )
                return jsonify(ret_info)
            except Exception as e:
                ret_info['error'] = str(e)
                return ret_info

        @app.route('/faiss/state', methods=["GET"])
        def state():
            ret_info = {
                'state': 1
            }
            for db_name in self.db_names:
                if self.indexs[db_name].warning is not None:
                    ret_info['state'] = 0
                    logger.error(self.indexs[db_name].warning)
                    break
            try:
                conn = self.mysql_manager.pool.connection()
                conn.close()
            except Exception as e:
                logger.exception('mysql 异常')
                ret_info['state'] = 0
            return jsonify(ret_info)

        serve(app=app, host=self.flask_host, port=self.flask_port)


if __name__ == "__main__":
    service = VectorSearchServer(cfg=config)
    service.flask_run()

