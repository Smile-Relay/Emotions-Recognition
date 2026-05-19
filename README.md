# Smile Relay - Emotions Recognition Backend

Smile Relay 的后端核心，负责实时情绪识别、数据持久化以及硬件交互（打印服务）。

## 🚀 核心功能

- **实时情绪识别**：基于 Pytorch 实现的 mini-Xception 模型。
- **人脸检测与对齐**：使用 dnn/Haar Cascade 进行人脸检测，并结合 dlib 进行对齐预处理。
- **Web 服务**：基于 Flask 提供 API 接口，支持前端进行情绪检测请求、漂流瓶存储及查询。
- **打印排队系统**：集成了 CUPS 打印服务，支持将情绪卡片渲染并自动打印。
- **异步任务处理**：使用多线程处理截图和打印等耗时操作。

## 🛠️ 安装与运行

### 1. 安装依赖

```bash
pip3 install -r requirement.txt
```

*注意：如果需要使用 CUPS 打印功能，请确保系统已安装 `libcups2-dev`。*

### 2. 运行 Web 服务器

```bash
python3 web_detect.py
```

### 3. (可选) 摄像头演示

```bash
python3 camera_demo.py
```

## 📂 项目结构

- `web_detect.py`: Flask 应用核心，包含情绪识别、数据库操作和打印状态接口。
- `model/`: mini-Xception 模型的 Pytorch 实现。
- `face_detector/`: 人脸检测模块。
- `face_alignment/`: 基于 dlib 的人脸对齐预处理。
- `db_models/`: 数据库模型定义（基于 Peewee）。
- `checkpoint/`: 预训练模型权重。

## ⚙️ 打印服务说明

本项目支持通过 `cups` 进行自动化打印。
- 后端会启动无头浏览器（Pyppeteer）对生成的漂流瓶网页进行截图。
- 截图经过图像处理（添加边距等）后发送至系统默认打印机。
- 前端可通过 `/print_status` 接口实时监控打印队列状态。

## 📄 许可证

基于开源情绪识别框架进行二次开发。
