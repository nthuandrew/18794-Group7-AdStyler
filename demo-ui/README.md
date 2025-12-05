# AI图像生成器 Django项目

这是一个使用Django构建的AI图像生成器，集成了GPT4All进行提示词优化和Diffusion模型接口（目前为占位符实现）。

## 功能特性

- 用户输入文字描述和目标风格
- 使用GPT4All进行prompt engineering优化提示词
- 通过Diffusion模型接口生成图像（目前返回随机占位图）
- 美观的现代化Web界面
- 生成历史记录保存

## 使用Conda环境运行项目（推荐）

### 1. 创建Conda环境

```bash
# 进入项目目录
cd /Users/phi/Documents/ad-prompting-llm-demo-ui

# 使用environment.yml创建环境（推荐）
conda env create -f environment.yml

# 或者手动创建环境
conda create -n stylediffusion python=3.10 -y
conda activate stylediffusion
pip install -r requirements.txt
```

### 2. 激活环境并初始化数据库

```bash
# 激活conda环境
conda activate stylediffusion

# 创建数据库迁移
python manage.py makemigrations

# 应用迁移
python manage.py migrate
```

### 3. 运行开发服务器

```bash
# 方法1：使用启动脚本
./run.sh

# 方法2：手动运行
conda activate stylediffusion
python manage.py runserver
```

访问 http://127.0.0.1:8000 查看应用。

---

## 使用Python虚拟环境（备选方案）

### 1. 创建虚拟环境

```bash
# 进入项目目录
cd /Users/phi/Documents/ad-prompting-llm-demo-ui

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 创建数据库迁移

```bash
# 创建迁移文件
python manage.py makemigrations

# 应用迁移
python manage.py migrate
```

### 4. 创建超级用户（可选，用于访问Django管理后台）

```bash
python manage.py createsuperuser
```

### 5. 运行开发服务器

```bash
python manage.py runserver
```

访问 http://127.0.0.1:8000 查看应用。

## 项目结构

```
ad-prompting-llm-demo-ui/
├── manage.py
├── requirements.txt
├── environment.yml           # Conda环境配置文件
├── run.sh                    # 启动脚本
├── image_generator/          # Django项目配置
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── generator/                # 主应用
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── urls.py
│   ├── views.py
│   ├── services/            # 服务模块
│   │   ├── __init__.py
│   │   ├── llm_service.py   # GPT4All服务
│   │   └── diffusion_service.py  # Diffusion模型服务
│   └── templates/
│       └── generator/
│           └── index.html   # 前端页面
└── media/                    # 生成的图像存储目录（自动创建）
```

## API端点

### 生成图像
- **URL**: `/api/generate/`
- **方法**: POST
- **请求体**:
```json
{
    "user_text": "一只可爱的小猫",
    "target_style": "水彩画"
}
```
- **响应**:
```json
{
    "success": true,
    "image_url": "/media/generated_xxxxx.png",
    "processed_prompt": "优化后的提示词",
    "record_id": 1
}
```

### 获取历史记录
- **URL**: `/api/history/`
- **方法**: GET
- **响应**: 返回最近20条生成记录

## GPT4All模型配置

### 使用本地已下载的模型

如果你已经在GPT4All软件中下载了模型（如DeepSeek-R1-Distill-Qwen-7B），可以通过以下方式使用：

#### 方法1：通过环境变量（推荐）

```bash
# 设置环境变量
export GPT4ALL_MODEL="DeepSeek-R1-Distill-Qwen-7B"

# 运行服务器
conda activate stylediffusion
python manage.py runserver
```

#### 方法2：修改settings.py

编辑 `image_generator/settings.py`，修改 `GPT4ALL_MODEL_NAME`：

```python
GPT4ALL_MODEL_NAME = "DeepSeek-R1-Distill-Qwen-7B"  # 或使用完整路径
```

#### 方法3：使用完整路径

如果模型不在默认路径，可以使用完整路径：

```python
GPT4ALL_MODEL_NAME = "/path/to/your/model.gguf"
```

### 模型查找路径

程序会按以下顺序查找模型：
1. 使用模型名称在GPT4All默认路径查找
2. macOS: `~/Library/Application Support/GPT4All/`
3. Linux: `~/.local/share/GPT4All/`
4. 如果都找不到，会回退到默认的orca-mini-3b模型

### 注意事项

1. **模型格式**: 确保模型是GGUF格式
2. **硬件要求**: DeepSeek-R1-Distill-Qwen-7B是7B参数模型，需要足够的RAM（建议16GB+）
3. **首次加载**: 首次加载模型可能需要一些时间
4. **模型路径**: 如果模型在GPT4All软件中下载，通常会在上述默认路径中

## 注意事项

1. **GPT4All模型下载**: 如果使用默认模型，首次运行时会自动下载模型文件（约2-3GB），请确保网络连接正常。

2. **Diffusion模型接口**: 目前 `diffusion_service.py` 中的 `generate_image` 方法返回的是随机生成的占位图。要使用真实的diffusion模型，需要：
   - 安装相应的diffusion模型库（如Stable Diffusion）
   - 在 `generate_image` 方法中实现真实的模型调用

3. **媒体文件**: 生成的图像保存在 `media/` 目录中，确保该目录有写入权限。

4. **生产环境**: 部署到生产环境时，请：
   - 修改 `SECRET_KEY`
   - 设置 `DEBUG = False`
   - 配置合适的数据库
   - 设置静态文件和媒体文件的正确服务方式

## 开发说明

### 实现真实的Diffusion模型

编辑 `generator/services/diffusion_service.py`，在 `generate_image` 方法中替换占位符代码：

```python
def generate_image(self, prompt: str, width: int = 512, height: int = 512) -> Image.Image:
    # TODO: 在这里实现真实的diffusion model调用
    # 例如使用Stable Diffusion:
    # from diffusers import StableDiffusionPipeline
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # image = pipe(prompt, width=width, height=height).images[0]
    # return image
    pass
```

## 许可证

MIT License

