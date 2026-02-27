# LinearRAG 嵌入存储模块
# 负责管理文本嵌入的存储和检索，使用 Parquet 格式持久化

from copy import deepcopy
from src.utils import compute_mdhash_id
import numpy as np
import pandas as pd
import os

class EmbeddingStore:
    """
    嵌入存储类
    
    管理文本及其嵌入向量的存储，支持增量更新和持久化
    使用 Parquet 格式存储数据，包含 hash_id、text 和 embedding 三列
    """
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        初始化嵌入存储
        
        Args:
            embedding_model: 嵌入模型实例
            db_filename (str): 数据库文件路径（Parquet 格式）
            batch_size (int): 批处理大小
            namespace (str): 命名空间前缀，用于生成 hash_id
        """
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace
        
        # 数据存储列表
        self.hash_ids = []
        self.texts = []
        self.embeddings = []
        
        # 索引字典，用于快速查找
        self.hash_id_to_text = {}
        self.hash_id_to_idx = {}
        self.text_to_hash_id = {}
        
        # 从磁盘加载已有数据
        self._load_data()
    
    def _load_data(self):
        """
        从 Parquet 文件加载数据到内存
        """
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["text"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            
            # 构建索引
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
            self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
            print(f"[{self.namespace}] 从 {self.db_filename} 加载了 {len(self.hash_ids)} 条记录")
        
    def insert_text(self, text_list):
        """
        插入新的文本列表，计算嵌入并存储
        
        Args:
            text_list (List[str]): 要插入的文本列表
        """
        # 为每个文本生成 hash_id
        nodes_dict = {}
        for text in text_list:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}
        
        all_hash_ids = list(nodes_dict.keys())
        
        # 找出已存在的和新增的 hash_id
        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]      
        
        # 只编码新增的文本
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        all_embeddings = self.embedding_model.encode(texts_to_encode, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size)
        
        # 插入新数据
        self._upsert(missing_ids, texts_to_encode, all_embeddings)

    def _upsert(self, hash_ids, texts, embeddings):
        """
        执行实际的插入操作
        
        Args:
            hash_ids (List[str]): hash_id 列表
            texts (List[str]): 文本列表
            embeddings (List[ndarray]): 嵌入向量列表
        """
        # 追加到列表
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        
        # 重建索引
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
        
        # 保存到磁盘
        self._save_data()

    def _save_data(self):
        """
        将数据保存到 Parquet 文件
        """
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "text": self.texts,
            "embedding": self.embeddings
        })
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        data_to_save.to_parquet(self.db_filename, index=False)
      
    def get_hash_id_to_text(self):
        """
        获取 hash_id 到文本的映射（深拷贝）
        
        Returns:
            Dict[str, str]: hash_id 到文本的映射
        """
        return deepcopy(self.hash_id_to_text)
    
    def encode_texts(self, texts):
        """
        编码文本为嵌入向量
        
        Args:
            texts (List[str]): 文本列表
            
        Returns:
            ndarray: 嵌入向量数组
        """
        return self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size)
    
    def get_embeddings(self, hash_ids):
        """
        根据 hash_id 获取嵌入向量
        
        Args:
            hash_ids (List[str]): hash_id 列表
            
        Returns:
            ndarray: 嵌入向量数组
        """
        if not hash_ids:
            return np.array([])
        # 查找索引
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        # 获取对应的嵌入
        embeddings = np.array(self.embeddings)[indices]
        return embeddings
