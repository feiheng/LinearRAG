# LinearRAG 工具函数模块
# 提供通用工具函数：MD5 哈希、LLM 调用、文本标准化、日志设置等

from hashlib import md5
from dataclasses import dataclass, field
from typing import List, Dict
import httpx
from openai import OpenAI
from collections import defaultdict
import multiprocessing as mp
import re
import string
import logging
import numpy as np
import os

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    计算内容的 MD5 哈希值作为唯一标识符
    
    Args:
        content (str): 要哈希的内容
        prefix (str): 哈希值前缀，用于区分不同类型的对象
        
    Returns:
        str: 带前缀的 MD5 哈希值
    """
    return prefix + md5(content.encode()).hexdigest()

class LLM_Model:
    """
    大语言模型封装类
    
    封装 OpenAI API 调用，提供统一的推理接口
    """
    def __init__(self, llm_model):
        """
        初始化 LLM 模型
        
        Args:
            llm_model (str): 模型名称，如 "gpt-4o-mini"
        """
        # 创建 HTTP 客户端，设置 60 秒超时
        http_client = httpx.Client(timeout=60.0, trust_env=False)
        
        # 初始化 OpenAI 客户端
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量获取 API 密钥
            base_url=os.getenv("OPENAI_BASE_URL"),  # 从环境变量获取 API 基础 URL
            http_client=http_client
        )
        
        # 配置 LLM 调用参数
        self.llm_config = {
            "model": llm_model,
            "max_tokens": 2000,  # 最大生成 token 数
            "temperature": 0,    # 温度为 0，确保输出确定性
        }
    
    def infer(self, messages):
        """
        执行模型推理
        
        Args:
            messages (List[Dict]): 对话消息列表，包含 role 和 content
            
        Returns:
            str: 模型生成的回复内容
        """
        response = self.openai_client.chat.completions.create(**self.llm_config, messages=messages)
        return response.choices[0].message.content

def normalize_answer(s):
    """
    标准化答案文本，用于精确匹配评估
    
    执行以下操作：
    1. 转为小写
    2. 移除标点符号
    3. 移除冠词 (a, an, the)
    4. 标准化空白字符
    
    Args:
        s: 输入答案（可以是任意类型）
        
    Returns:
        str: 标准化后的答案
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s) 
    
    def remove_articles(text):
        """移除冠词"""
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        """标准化空白字符"""
        return " ".join(text.split())
    
    def remove_punc(text):
        """移除标点符号"""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        """转为小写"""
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def setup_logging(log_file):
    """
    配置日志系统
    
    Args:
        log_file (str): 日志文件路径
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # 创建日志处理器：控制台 + 文件
    handlers = [logging.StreamHandler()]  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # 抑制 httpx/openai 的噪声日志（如 401 未授权等 HTTP 请求日志）
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

def min_max_normalize(x):
    """
    Min-Max 标准化，将数值缩放到 [0, 1] 范围
    
    Args:
        x: 输入数组
        
    Returns:
        ndarray: 标准化后的数组
    """
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # 处理所有值相同的情况（范围为 0）
    if range_val == 0:
        return np.ones_like(x)  # 返回全 1 数组
    
    return (x - min_val) / range_val
