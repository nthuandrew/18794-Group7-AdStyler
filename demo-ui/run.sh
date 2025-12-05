#!/bin/bash
# 启动Django开发服务器的脚本

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stylediffusion

# 运行Django开发服务器
python manage.py runserver

