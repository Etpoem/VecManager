<h1 style="text-align:center"> 人脸搜索服务API文档 <h1>

- - - 


<!-- @import "[TOC]" {cmd="toc" depthFrom=2 depthTo=3 orderedList=false} -->

<!-- code_chunk_output -->

- [1-向量库库操作](#1-向量库库操作)
  - [1.1-获取向量库信息](#11-获取向量库信息)
  - [1.2-创建向量库](#12-创建向量库)
  - [1.3-删除向量库](#13-删除向量库)
  - [1.4-修改向量库容量大小](#14-修改向量库容量大小)
- [2-特征向量操作](#2-特征向量操作)
  - [2.1-单条向量添加](#21-单条向量添加)
  - [2.2-单条向量删除](#22-单条向量删除)
  - [2.3-单条向量获取](#23-单条向量获取)
  - [2.4-单条向量搜索](#24-单条向量搜索)
  - [2.5-单条向量搜索（fast)](#25-单条向量搜索fast)
  - [2.6-按时间段删除向量](#26-按时间段删除向量)
  - [2.7-批量删除向量](#27-批量删除向量)
  - [2.8-批量向量添加](#28-批量向量添加)
  - [2.9-批量向量搜索（fast)](#29-批量向量搜索fast)
- [3-其他](#3-其他)
  - [3.1-服务状态检测](#31-服务状态检测)

<!-- /code_chunk_output -->


- - - 

<div STYLE="page-break-after: always;"></div>

## 1-向量库库操作
### 1.1-获取向量库信息
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/faiss/db/info
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|否|向量库名称，不填写或为空字符时返回所有向量库信息|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则为空|
|result|json 数组|返回结果，json对象数组|

**result中字段说明**
|字段名|类型|描述|
|---|---|---|
|db_name|String|向量库名称|
|size|Integer|向量库现存向量数量|
|max_size|Integer|向量库可存储的最大向量数量|
|info|String|向量库备注信息|
|is_multiple_gpus|Bool|向量库是否使用多GPU,仅在有使用gpu版本的faiss时有效，使用cpu版本时该项不起作用|

#### 请求示例
~~~http
POST /faiss/db/info HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
  "db_name": ""
}
~~~
#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "info": "test0",
      "db_name": "database_0",
      "max_size": 10000,
      "size": 6,
      "is_multiple_gpus": false
    },
    {
      "info": "test1",
      "db_name": "database_1",
      "max_size": 10000,
      "size": 6,
      "is_multiple_gpus": true
    }
  ]
}
~~~

### 1.2-创建向量库
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/faiss/db/create
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称，唯一标识|
|max_size|Integer|是|向量库容量，范围限制1~30000000|
|is_multiple_gpus|Bool|否|向量库是否使用多GPU,默认False,仅在有使用gpu版本的faiss时有效，使用cpu版本时该项不起作用|
|info|String|否|向量库备注信息，默认空字符|


#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则为空字符|
|result|json数组|返回结果|

result中字段说明：
|字段名|类型|描述|
|---|---|---|
|db_name|String|向量库名称|
|size|Integer|向量库现存人脸数量|
|max_size|Integer|向量库可存储的最大人脸数量|
|is_multiple_gpus|Bool|向量库是否使用多GPU,仅在有使用gpu版本的faiss时有效，使用cpu版本时该项不起作用|
|info|String|向量库备注信息|


#### 请求示例
~~~http
POST /faiss/db/create HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
  "db_name": "database_2",
  "max_size": 10000,
  "is_multiple_gpus": false,
  "info": "database test 2"
}
~~~
#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "db_name": "database_2",
      "info": "database test 2",
      "max_size": 10000,
      "size": 0,
      "is_multiple_gpus": false
    }
  ]
}
~~~

### 1.3-删除向量库
#### 请求方式
- POST
#### 请求URL
    http；//{host}:{post}/faiss/db/remove
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|db_name|String|向量库名称|
|size|Integer|向量库的大小|
|max_size|Integer|向量库最大容量|
|is_multiple_gpus|Bool|向量库是否使用多GPU,仅在有使用gpu版本的faiss时有效，使用cpu版本时该项不起作用|
|info|String|向量库的备注信息|
#### 请求示例
~~~http
POST /faiss/db/remove HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
  "db_name": "database_2"
}
~~~
#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "db_name": "database_2",
      "info": "database test 2",
      "max_size": 10000,
      "size": 0,
      "is_multiple_gpus": false
    }
  ]
}
~~~

### 1.4-修改向量库容量大小
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/faiss/db/update
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
|max_size|Integer|是|向量库容量，范围限制1~30000000|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|db_name|String|向量库名称|
|size|Integer|向量库的大小|
|max_size|Integer|向量库最大容量|
|info|String|向量库的备注信息|
|is_multiple_gpus|Bool|向量库是否使用多GPU,仅在有使用gpu版本的faiss时有效，使用cpu版本时该项不起作用|

#### 请求示例
~~~http
POST /faiss/db/update HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
  "db_name": "database_0",
  "max_size": 1000
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
    {
      "db_name": "database_2",
      "info": "database test 2",
      "max_size": 1000,
      "size": 0,
      "is_multiple_gpus": false
    }
  ]
}
~~~

## 2-特征向量操作
### 2.1-单条向量添加
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/faiss/feature/insert
#### Headers
    Content-Type    application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
|feature_id|Integer|是|特征ID|
|faceCoord|Array|是|人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|feature|String|是|特征字符串|
|info|String|否|备注信息，默认空字符|
|create_datetime|String|否|创建的时间，默认为添加的时刻，格式为 '%Y-%m-%d %H:%M:%S'，例如‘2021-03-03 12:12:04'|

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|db_name|String|向量库名称|
|faceCoord|Array|人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|feature_id|Integer|特征ID|
|feature|String|特征字符串|
|info|String|备注信息|

#### 请求示例
~~~http
POST /faiss/feature/insert HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
  "db_name": "database_1",
  "faceCoord" : [100, 120, 200, 220],
  "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
  "feature_id": 100,
  "info": "test face"
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
  {
    "db_name": "database_1",
    "faceCoord" : [100, 120, 200, 220],
    "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
    "feature_id": 100,
    "info": "test face"
  }
  ]
}
~~~

### 2.2-单条向量删除
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/faiss/feature/delete
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
|feature_id|Integer|是|特征id|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|feature_id|Integer|特征ID|
|feature|String|特征字符串|
|faceCoord|Array|人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|info|String|备注信息|
#### 请求示例
~~~http
POST /faiss/feature/delete HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
	"db_name": "database_2",
	"feature_id": 22
}
~~~
#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
      "feature_id": 22,
      "info": "test face"
    }
  ]
}
~~~

### 2.3-单条向量获取
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/faiss/feature/get
#### Hearders
    Content-Type    application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
|feature_id|Integer|是|特征id|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|feature_id|Integer|特征ID|
|feature|String|特征字符串|
|faceCoord|Array|人脸位置坐标 [X_min, Y_min, X_max, Y_max]|
|info|String|备注信息|
#### 请求示例
~~~http
POST /faiss/feature/get HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
	"db_name": "database_2",
	"feature_id": 22
}
~~~
#### 返回结果示例
~~~json
{
  "error": "",
  "result": [
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
      "feature_id": 22,
      "info": "test face"
    }
  ]
}
~~~

### 2.4-单条向量搜索
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/faiss/feature/search
#### Header
    Content-Type application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
|feature|String|是|特征向量字符串|
|top|Integer|否|返回相似度最高的前top个结果，默认5|
|nprobe|Integer|否|聚簇搜索数量，1~128，默认128，越大搜索越精确，相对速度会更慢|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|feature_id|Integer|特征ID|
|score|Float|相似度 0~1，1为完全相似|
|faceCoord|Array|人脸框位置 [x_min, y_min, x_max, y_max]|
|info|String|备注信息|

#### 请求示例
~~~http
POST /faiss/feature/search HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
  "db_name": "database_1",
  "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
  "top": 2
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature_id": 100,
      "info": "test face",
      "socre": 0.9676
    },
    {
      "faceCoord" : [100, 120, 200, 220],
      "feature_id": 200,
      "info": "test face 2",
      "socre": 0.9376
    }
  ]
}
~~~

### 2.5-单条向量搜索（fast)
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/faiss/feature/search_fast
#### Header
    Content-Type application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
|feature|String|是|特征向量字符串|
|top|Integer|否|返回相似度最高的前top个结果，默认5|
|nprobe|Integer|否|聚簇搜索数量，1~128，默认128，越大搜索越精确，相对速度会更慢|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|feature_id|Integer|特征ID|
|score|Float|相似度 0~1，1为完全相似|
#### 请求示例
~~~http
POST /faiss/feature/search_fast HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
  "db_name": "database_1",
  "feature": "[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...]",
  "top": 2
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
    {
      "feature_id": 100,
      "socre": 0.9676
    },
    {
      "feature_id": 200,
      "socre": 0.9376
    }
  ]
}
~~~

### 2.6-按时间段删除向量
#### 请求方式
- POST or DELETE
#### 请求URL
    http://{host}:{port}/faiss/feature/delete_by_date
#### Headers
    Content-Type    application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
|begin_time|String|是|开始日期时间，格式为'%Y-%m-%d %H:%M:%S',如 '2021-03-13 10:30:30'|
|end_time|String|是|结束日期时间，格式为'%Y-%m-%d %H:%M:%S', 如 '2021-05-19 10:30:30'|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则为空|
|result|json数组|返回的结果|

result中的字段说明
|字段名|类型|描述|
|---|---|---|
|total|Integer|删除向量的总数量|
|feature_ids|Array|删除的人脸id|

#### 请求示例
~~~http
POST /faiss/feature/delete_by_date HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
    "db_name": "cbvsp",
    "begin_time": "2020-10-20 12:10:10",
    "end_time": "2021-01-02 12:10:10"
}
~~~

#### 返回结果示例
~~~json
{
    "error": "",
    "result": [
        {
            "total": 3,
            "feature_ids": [1, 2, 3]
        }
    ]
}
~~~

### 2.7-批量向量删除
#### 请求方式
- POST or DELETE
#### 请求URL
    http://{host}:{port}/faiss/feature/delete_batch
#### Headers
    Content-Type	application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
|feature_ids|Array|是|需要删除的特征id|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
#### 请求示例
~~~http
POST /faiss/feature/delete_batch HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
	"db_name": "database_2",
	"feature_id": [1, 2, 3]
}
~~~
#### 返回结果示例
~~~json
{
  "error": ""
}
~~~

### 2.8-批量向量添加
#### 请求方式
- POST or DELETE
#### 请求URL
    http://{host}:{port}/faiss/feature/insert_batch
#### Headers
    Content-Type    application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
|face_data|Array|是|[(face_id, face_vector, face_coord, face_info, create_datetime)]|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
#### 请求示例
~~~http
POST /faiss/feature/insert_batch HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
	"db_name": "database_2",
	"face_data": [
        (1367467, "[1.07346, 0.75875, ...]", "[10, 20, 110, 120]", "info_1")
    ]
}
~~~
#### 返回结果示例
~~~json
{
  "error": ""
}
~~~

### 2.9-批量向量搜索（fast)
#### 请求方式
- POST
#### 请求URL
    http://{host}:{port}/faiss/feature/search_fast_batch
#### Header
    Content-Type application/json
#### 请求参数
|字段名|类型|必需|描述|
|---|---|---|---|
|db_name|String|是|向量库名称|
|features|String|是|特征向量字符串|
|top|Integer|否|返回相似度最高的前top个结果，默认5|
|nprobe|Integer|否|聚簇搜索数量，1~128，默认128，越大搜索越精确，相对速度会更慢|
#### 响应结果
|字段名|类型|描述|
|---|---|---|
|error|String|错误信息，正常则返回空|
|result|json数组|返回结果|
result中字段说明
|字段名|类型|描述|
|---|---|---|
|feature_id|Integer|特征ID|
|score|Float|相似度 0~1，1为完全相似|
#### 请求示例
~~~http
POST /faiss/feature/search_fast HTTP/1.1
Host: 192.168.96.136:17100
Content-Type: application/json

{
  "db_name": "database_1",
  "feature": "[[0.09483230113983154, -0.006801479030400515, 0.05101810023188591, ...],[...]]",
  "top": 3
}
~~~
#### 返回结果示例
~~~json
{
  
  "error": "",
  "result": [
    {
        "feature_ids": [[105, 89, 77], [100, 102, 111]],
        "scores": [[0.95, 0.92, 0.90], [0.90, 0.89, 0.60]]
    }
  ]
}
~~~


## 3-其他
### 3.1-服务状态检测
#### 请求方式
- GET

#### 请求URL
    http://{host}:{port}/faiss/state

#### 响应结果
|字段名|类型|描述|
|---|---|---|
|state|Integer| 0或1，1表示服务正常，0表示服务出现异常|
