"""
Django settings for image_generator project.
"""

from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-change-this-in-production-12345'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'generator',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'image_generator.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'image_generator.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'zh-hans'

TIME_ZONE = 'Asia/Shanghai'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# GPT4All模型配置
# 建议使用完整路径以避免自动下载
# 如果设置为模型名称，程序会先在本地路径查找，找不到才会下载
GPT4ALL_MODEL_NAME = os.getenv('GPT4ALL_MODEL', 
    '/Users/phi/Library/Application Support/nomic.ai/GPT4All/DeepSeek-R1-Distill-Qwen-7B-Q4_0.gguf')

# Layout生成方式配置
# 可选值: "llm" (使用finetuned layout LLM) 或 "sample" (从训练数据随机采样)
LAYOUT_GENERATION_METHOD = os.getenv('LAYOUT_GENERATION_METHOD', 'sample')

# Layout LLM配置 (当LAYOUT_GENERATION_METHOD="llm"时使用)
LAYOUT_LLM_CHECKPOINT_PATH = os.getenv('LAYOUT_LLM_CHECKPOINT_PATH', 
    str(BASE_DIR.parent / 'layout-llm-finetuning' / 'finetune_layout_llm' / 'output_layout_llm' / 'checkpoint-500'))
LAYOUT_LLM_BASE_MODEL = os.getenv('LAYOUT_LLM_BASE_MODEL', 'Qwen/Qwen2.5-1.5B')
LAYOUT_PROB_MODEL_PATH = os.getenv('LAYOUT_PROB_MODEL_PATH',
    str(BASE_DIR.parent / 'layout-llm-finetuning' / 'train_layout_distribution' / 'layout_prob_model.joblib'))
LAYOUT_THRESHOLDS_PATH = os.getenv('LAYOUT_THRESHOLDS_PATH',
    str(BASE_DIR.parent / 'layout-llm-finetuning' / 'train_layout_distribution' / 'layout_thresholds.json'))

# Layout采样配置 (当LAYOUT_GENERATION_METHOD="sample"时使用)
TRAIN_LAYOUT_JSON_PATH = os.getenv('TRAIN_LAYOUT_JSON_PATH',
    str(BASE_DIR.parent / 'layout-llm-finetuning' / 'create_layout_dataset' / 'train_layout.json'))

