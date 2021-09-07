# -*- coding: UTF-8 -*-
# -------------------------------------------------------------
# @Author   :   panjf
# @Time     :   2021/2/19 15:56
# @Desc     :   数据验证模块
# -------------------------------------------------------------
import datetime
import toml
from pathlib import Path
from marshmallow import Schema, fields, validate, EXCLUDE, ValidationError

cfg = toml.load(Path(__file__).parents[1] / 'config.toml')


class DbCreate(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    max_size = fields.Integer(required=True, validate=validate.Range(min=1, max=30000000))
    info = fields.String(missing='')
    is_multiple_gpus = fields.Boolean(missing=False)


class DbRemove(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)


class DbUpdate(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    max_size = fields.Integer(required=True, validate=validate.Range(min=1, max=30000000))


class DbInfo(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(missing='')


class DbDelete(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    feature_id = fields.Integer(required=True)


class DbDeleteByDate(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    begin_time = fields.DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    end_time = fields.DateTime(required=True, format='%Y-%m-%d %H:%M:%S')


class DbDeleteBatch(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    feature_ids = fields.List(fields.Integer(), required=True)


class DbGet(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    feature_id = fields.Integer(required=True)


class DbInsertFeature(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    feature_id = fields.Integer(required=True, validate=validate.Range(min=0, max=9223372036854775807))
    feature = fields.String(required=True)
    faceCoord = fields.List(fields.Integer(), required=True)
    info = fields.String(missing='')
    create_datetime = fields.DateTime(missing=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                      format='%Y-%m-%d %H:%M:%S')


class DbSearchFeature(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    feature = fields.String(required=True)
    top = fields.Integer(missing=5)
    nprobe = fields.Integer(missing=cfg['faiss']['nprobe'], validate=validate.Range(min=1, max=cfg['faiss']['nlist']))


class DbSearchFeatureFast(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    feature = fields.String(required=True)
    top = fields.Integer(missing=5)
    nprobe = fields.Integer(missing=cfg['faiss']['nprobe'], validate=validate.Range(min=1, max=cfg['faiss']['nlist']))


class DbSearchFeatureFastBatch(Schema):
    class Meta:
        unknown = EXCLUDE
    db_name = fields.String(required=True)
    features = fields.String(required=True)
    top = fields.Integer(missing=5)
    nprobe = fields.Integer(missing=cfg['faiss']['nprobe'] / 2,
                            validate=validate.Range(min=1, max=cfg['faiss']['nlist']))

