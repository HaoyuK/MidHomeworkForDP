# 神经网络和深度学习课程-期中作业任务2：训练目标检测模型

## 仓库介绍

本项目为课程DATA620004——神经网络和深度学习期中作业任务2的代码仓库

* 任务2：在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3

* 基本要求：

  （1） 学习使用现成的目标检测框架——如mmdetection或detectron2——在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3；

  （2） 挑选4张测试集中的图像，通过可视化对比训练好的Faster R-CNN第一阶段产生的proposal box和最终的预测结果。
  
  （3） 搜集三张不在VOC数据集内包含有VOC中类别物体的图像，分别可视化并比较两个在VOC数据集上训练好的模型在这三张图片上的检测结果（展示bounding box、类别标签和得分）；


### 文件说明
```bash
├── data/  # data的路径
├── configs/ # 模型训练的config路径
├── tools/ # 模型训练、测试、log分析的代码路径
  ├── train.py # 模型训练的主代码
├── demo/ # 模型训练完毕，加载最优checkpoint对图片进行分析
├── log_images/ # 对log进行分析的文件夹
├── work_dirs / # 模型训练的workspace
│   └── faster-rcnn_r50_fpn_1x_voc
│       ├── faster-rcnn_r50_fpn_1x_voc.py   # 配置文件
│       ├── epoch_12.pth                    # 模型
├── visualize.py                            # 对图像进行目标检测结果的可视化
├── origin                                  # 需要进行检测的图像（目录）
├── final
│   └── vis                                 # 图像检测结果（目录）
```

## Requirements
本项目基于```mmdetection```框架训练Faster R-CNN和YOLO V3，参考[mmdetection]("https://mmdetection.readthedocs.io/en/latest/get_started.html")的官方文档对环境进行配置。以下为本项目使用的环境配置方法。

#### Prerequisites
```bash
# Step 1. Create a conda environment and activate it.
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# Step 2. Install PyTorch following official instructions, e.g.
conda install pytorch torchvision -c pytorch
```

#### Installation
```bash
# Step 0. Install MMEngine and MMCV using MIM.
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Step 1. Install MMDetection.
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

# Step 2. For visualization
pip install seaborn
pip install matplotlib
```

## 一、数据下载与处理
执行以下指令下载`VOC2007`和`VOC2012`数据集

```bash
python tools/misc/download_dataset.py --dataset-name voc2007 --unzip
python tools/misc/download_dataset.py --dataset-name voc2012 --unzip
```
下载好的数据会自动储存在`data/VOCdevkit/`下
## 二、 模型的训练

### config编写

本项目已经编写好训练Faster R-CNN和YOLO V3的config文件，储存的位置分别为
- Faster R-CNN
```
./configs/haoyu/faster-rcnn_r50_fpn_1x_voc.py
```
- YOLO V3
```
./configs/haoyu/yolov3_d53_8xb8-ms-608-273e_voc_0529.py
```

### 模型训练

进入仓库根目录，运行：

* Faster R-CNN
```bash
python tools/train.py configs/haoyu/faster-rcnn_r50_fpn_1x_voc.py
```
* YOLO V3
```bash
python tools/train.py configs/haoyu/yolov3_d53_8xb8-ms-608-273e_voc_0529.py 
  ```

生成的模型权重和log文件会自动保存在`work_dirs`文件夹中；本项目中保存了两个模型训练过程中的log文件在以下地址：
- Faster R-CNN
```
./work_dirs/faster-rcnn_r50_fpn_1x_voc/20240505_182859/20240505_182859.log
```
- YOLO V3
```
./work_dirs/yolov3_d53_8xb8-ms-608-273e_voc_0529/20240529_200011/20240529_200011.log 
```

## 二、训练log分析以及模型效果测试

### 1. 训练log分析

* Faster RCNN 模型训练时Loss变化情况
```bash
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster-rcnn_r50_fpn_1x_voc/20240505_182859/vis_data/20240505_182859.json --keys loss loss_cls loss_bbox --out ./log_images/fater_rcnn_train_losses.pdf --legend loss loss_cls loss_bbox
```
![Faster RCNN 模型训练时Loss变化情况](./log_images/fater_rcnn_train_losses.pdf)


* Faster RCNN 模型验证时mAP指标变化情况
```bash
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster-rcnn_r50_fpn_1x_voc/20240505_182859/vis_data/20240505_182859.json --keys mAP  --out ./log_images/yolov3_new_eval_map.pdf --legend mAP --eval-interval 1
```
![Faster RCNN 模型验证时mAP指标变化情况](./log_images/fater_rcnn_eval_map.pdf)

* YOLO V3 模型训练时Loss变化情况
```bash
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/yolov3_d53_8xb8-ms-608-273e_voc_0529/20240529_200011/vis_data/20240529_200011.json --keys loss loss_cls loss_conf loss_xy loss_wh --out ./log_images/yolov3_0529_train_losses.pdf --legend loss loss_cls loss_conf loss_xy loss_wh
```
![YOLO V3 模型训练时Loss变化情况](./log_images/yolov3_0529_train_losses.pdf)

* YOLO V3 模型验证时mAP指标变化情况
```bash
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/yolov3_d53_8xb8-ms-608-273e_voc_0529/20240529_200011/vis_data/20240529_200011.json --keys mAP  --out ./log_images/yolov3_0529_eval_map.pdf --legend mAP --eval-interval 1
```
![YOLO V3 模型验证时mAP指标变化情况](./log_images/yolov3_0529_eval_map.pdf)

### 2. 模型整体效果测试
先将下载的模型权重移至`work_dirs/`下

运行以下命令，在VOC2007的测试集上对模型性能进行测试：
* Faster RCNN 
```bash
python tools/test.py \
       work_dirs/faster-rcnn_r50_fpn_1x_voc/faster-rcnn_r50_fpn_1x_voc.py \ 
       work_dirs/faster-rcnn_r50_fpn_1x_voc/epoch_12.pth 
```

* YOLO V3 
```bash
python tools/test.py \
      work_dirs/yolov3_d53_8xb8-ms-608-273e_voc_0529/yolov3_d53_8xb8-ms-608-273e_voc_0529.py \
      work_dirs/yolov3_d53_8xb8-ms-608-273e_voc_0529/epoch_14.pth 
```

### 3. Proposal Box 与最终结果对比
visualize.py：对图像进行目标检测结果的可视化。自定义参数：

- 配置文件：`config_path`
- 模型：`checkpoint`
- 待检测图像目录：`./origin/`
- 检测结果图像保存目录：`./final`（结果自动保存在`./final/vis/`）
```bash
python visualize.py
```
### 4. Faster RCNN 与 YOLO V3 模型效果对比

具体见```demo/inference_haoyu.ipynb```


## 更多实验结果和细节详见报告
