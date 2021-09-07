# VecManager
REST 风格的向量搜索与管理，向量搜索基于faiss
- 向量搜索支持使用 GPU 加速，量大时加速明显，不占用cpu资源
- 使用 mysql 进行元数据存储，每个向量可添加单独备注信息
- faiss 中数据使用 float16 半精度计算，节约存储和计算资源
- 良好的数据落盘机制，减少应用崩溃，关机断电带来的数据丢失情况
- 向量的添加和删除可实时生效
- 可创建和管理多个向量库

## 说明
faiss 是一个性能强大的向量相似性搜索库，提供了各种算法来加速向量的搜索。但在面对一些场景的时候有时候会显得力不从心，比如，index存储的是压缩量化后的向量数据，用户无法获取元数据；向量数据持久化方式是整个 index 进行写入，当向量多的时候，写入时间较长；GPU 和 CPU 的接口方法不完全相同；因此我对 faiss 进行了封装，使得向量管理更方便。

具体接口参见该项目的文档 [VecManager.md](VecManager.md)

## 部署说明
按照需求到 config.toml 中进行相关参数配置
可配置的参数，在文件中皆有注释给出相应说明

给出了 Dockerfile 可从一个 nvidia 基础镜像制作该部署镜像 

**docker 命令部署**
~~~bash
    docker run -d --restart=always -v 本地代码路径:/app --runtime=nvidia --net=host --name=vec_manager image_name python main.py
~~~

**使用 kubectl 命令部署**
~~~bash
    kubectl create -f VecManager.yaml
~~~


