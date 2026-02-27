# LinearRAG 核心模块
# 实现基于线性图结构的检索增强生成（RAG）系统
# 支持两种检索方式：BFS 迭代和向量化矩阵计算

from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.ner import SpacyNER
import igraph as ig
import re
import logging
import torch
logger = logging.getLogger(__name__)


class LinearRAG:
    """
    LinearRAG 核心类
    
    实现基于图结构的检索增强生成系统，主要功能包括：
    1. 索引构建：段落嵌入、实体识别、图结构构建
    2. 检索：基于种子实体的图搜索或向量化检索
    3. 问答：使用 LLM 生成答案
    """
    def __init__(self, global_config):
        """
        初始化 LinearRAG
        
        Args:
            global_config (LinearRAGConfig): 全局配置对象
        """
        self.config = global_config
        logger.info(f"使用以下配置初始化 LinearRAG: {self.config}")
        
        # 设置检索方法
        retrieval_method = "向量化矩阵检索" if self.config.use_vectorized_retrieval else "BFS 迭代"
        logger.info(f"使用检索方法：{retrieval_method}")
        
        # 设置 GPU 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.config.use_vectorized_retrieval:
            logger.info(f"向量化检索使用设备：{self.device}")
        
        self.dataset_name = global_config.dataset_name
        
        # 加载嵌入存储
        self.load_embedding_store()
        
        # 初始化组件
        self.llm_model = self.config.llm_model
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        self.graph = ig.Graph(directed=False)  # 无向图

    def load_embedding_store(self):
        """
        加载三个嵌入存储：
        1. 段落嵌入存储
        2. 实体嵌入存储
        3. 句子嵌入存储
        """
        self.passage_embedding_store = EmbeddingStore(
            self.config.embedding_model, 
            db_filename=os.path.join(self.config.working_dir, self.dataset_name, "passage_embedding.parquet"), 
            batch_size=self.config.batch_size, 
            namespace="passage"
        )
        self.entity_embedding_store = EmbeddingStore(
            self.config.embedding_model, 
            db_filename=os.path.join(self.config.working_dir, self.dataset_name, "entity_embedding.parquet"), 
            batch_size=self.config.batch_size, 
            namespace="entity"
        )
        self.sentence_embedding_store = EmbeddingStore(
            self.config.embedding_model, 
            db_filename=os.path.join(self.config.working_dir, self.dataset_name, "sentence_embedding.parquet"), 
            batch_size=self.config.batch_size, 
            namespace="sentence"
        )

    def load_existing_data(self, passage_hash_ids):
        """
        加载已有的 NER 结果，支持增量处理
        
        Args:
            passage_hash_ids (Set[str]): 段落 hash_id 集合
            
        Returns:
            tuple: (existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids)
        """
        self.ner_results_path = os.path.join(self.config.working_dir, self.dataset_name, "ner_results.json")
        
        if os.path.exists(self.ner_results_path):
            # 加载已有的 NER 结果
            existing_ner_reuslts = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            
            # 计算需要处理的新段落
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def qa(self, questions):
        """
        对问题列表进行问答
        
        Args:
            questions (List[Dict]): 问题列表，每个问题包含 question 字段
            
        Returns:
            List[Dict]: 包含预测答案的问题列表
        """
        # 检索相关段落
        retrieval_results = self.retrieve(questions)
        
        # 系统提示：指导 LLM 进行推理
        system_prompt = """As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations."""
        
        all_messages = []
        
        # 构建每个问题的消息
        for retrieval_result in retrieval_results:
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            
            prompt_user = ""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user}
            ]
            all_messages.append(messages)
        
        # 并行调用 LLM 进行推理
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA 阅读（并行）"
            ))

        # 提取答案
        for qa_result, question_info in zip(all_qa_results, retrieval_results):
            try:
                # 从 "Answer: " 后提取答案
                pred_ans = qa_result.split('Answer:')[1].strip()
            except:
                # 如果解析失败，使用完整回复
                pred_ans = qa_result
            question_info["pred_answer"] = pred_ans
        
        return retrieval_results
        
    def retrieve(self, questions):
        """
        为每个问题检索相关段落
        
        Args:
            questions (List[Dict]): 问题列表
            
        Returns:
            List[Dict]: 检索结果，包含问题和相关段落
        """
        # 加载嵌入数据到内存
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        
        # 构建图的节点索引
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}

        # 如果使用向量化检索，预计算稀疏矩阵
        if self.config.use_vectorized_retrieval:
            logger.info("为向量化检索预计算稀疏邻接矩阵...")
            self._precompute_sparse_matrices()
            e2s_shape = self.entity_to_sentence_sparse.shape
            s2e_shape = self.sentence_to_entity_sparse.shape
            e2s_nnz = self.entity_to_sentence_sparse._nnz()
            s2e_nnz = self.sentence_to_entity_sparse._nnz()
            logger.info(f"矩阵构建完成：实体 - 句子 {e2s_shape}, 句子 - 实体 {s2e_shape}")
            logger.info(f"E2S 稀疏度：{(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}% (nnz={e2s_nnz})")
            logger.info(f"S2E 稀疏度：{(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}% (nnz={s2e_nnz})")
            logger.info(f"设备：{self.device}")

        retrieval_results = []
        
        # 为每个问题执行检索
        for question_info in tqdm(questions, desc="检索中"):
            question = question_info["question"]
            
            # 编码问题
            question_embedding = self.config.embedding_model.encode(
                question, normalize_embeddings=True, show_progress_bar=False, batch_size=self.config.batch_size
            )
            
            # 获取种子实体
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = self.get_seed_entities(question)
            
            if len(seed_entities) != 0:
                # 有种子实体：使用图搜索
                sorted_passage_hash_ids, sorted_passage_scores = self.graph_search_with_seed_entities(
                    question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
                )
                final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.hash_id_to_text[passage_hash_id] for passage_hash_id in final_passage_hash_ids]
            else:
                # 无种子实体：使用稠密段落检索
                sorted_passage_indices, sorted_passage_scores = self.dense_passage_retrieval(question_embedding)
                final_passage_indices = sorted_passage_indices[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.texts[idx] for idx in final_passage_indices]
            
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"]
            }
            retrieval_results.append(result)
        
        return retrieval_results
    
    def _precompute_sparse_matrices(self):
        """
        预计算稀疏邻接矩阵用于向量化检索
        
        使用 PyTorch 稀疏张量格式，在 GPU 上加速计算
        """
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # 构建实体到句子矩阵（提及矩阵）
        entity_to_sentence_indices = []
        entity_to_sentence_values = []
        
        for entity_hash_id, sentence_hash_ids in self.entity_hash_id_to_sentence_hash_ids.items():
            entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)
        
        # 构建句子到实体矩阵
        sentence_to_entity_indices = []
        sentence_to_entity_values = []
        
        for sentence_hash_id, entity_hash_ids in self.sentence_hash_id_to_entity_hash_ids.items():
            sentence_idx = self.sentence_embedding_store.hash_id_to_idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.entity_embedding_store.hash_id_to_idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)
        
        # 转换为 PyTorch 稀疏张量（COO 格式）
        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.device
            ).coalesce()
        else:
            self.entity_to_sentence_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences), device=self.device
            )
        
        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.device
            ).coalesce()
        else:
            self.sentence_to_entity_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities), device=self.device
            )
            
    def graph_search_with_seed_entities(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        """
        基于种子实体的图搜索
        
        Args:
            question_embedding: 问题嵌入向量
            seed_entity_indices: 种子实体索引列表
            seed_entities: 种子实体文本列表
            seed_entity_hash_ids: 种子实体 hash_id 列表
            seed_entity_scores: 种子实体分数列表
            
        Returns:
            tuple: (sorted_passage_hash_ids, sorted_passage_scores)
        """
        if self.config.use_vectorized_retrieval:
            # 向量化版本
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(
                question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
            )
        else:
            # BFS 迭代版本
            entity_weights, actived_entities = self.calculate_entity_scores(
                question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
            )
        
        # 计算段落分数
        passage_weights = self.calculate_passage_scores(question_embedding, actived_entities)
        
        # 合并实体和段落分数
        node_weights = entity_weights + passage_weights
        
        # 运行个性化 PageRank
        ppr_sorted_passage_indices, ppr_sorted_passage_scores = self.run_ppr(node_weights)
        
        return ppr_sorted_passage_indices, ppr_sorted_passage_scores

    def run_ppr(self, node_weights):
        """
        运行个性化 PageRank 算法
        
        Args:
            node_weights (ndarray): 节点权重数组
            
        Returns:
            tuple: (sorted_passage_hash_ids, sorted_passage_scores)
        """
        # 处理无效权重
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        
        # 计算 PageRank 分数
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        # 提取段落节点分数
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        
        # 按分数降序排序
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]
        
        # 获取段落 hash_id
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]] 
            for i in sorted_indices_in_doc_scores
        ]
        
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        """
        使用 BFS 迭代计算实体分数（原始版本）
        
        Args:
            question_embedding: 问题嵌入向量
            seed_entity_indices: 种子实体索引
            seed_entities: 种子实体文本
            seed_entity_hash_ids: 种子实体 hash_id
            seed_entity_scores: 种子实体分数
            
        Returns:
            tuple: (entity_weights, actived_entities)
        """
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        
        # 初始化种子实体
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score    
        
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        
        # BFS 迭代扩展
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            
            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                # 低于阈值的实体不再扩展
                if entity_score < self.config.iteration_threshold:
                    continue
                
                # 获取实体相关的句子
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]) if sid not in used_sentence_hash_ids]
                if not sentence_hash_ids:
                    continue
                
                # 计算句子与问题的相似度
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                
                # 选择 top-k 句子
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence]
                
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    
                    # 获取句子中的实体
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids[top_sentence_hash_id]
                    
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        # 传播分数：实体分数 * 句子分数
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        
                        next_enitity_node_idx = self.node_name_to_vertex_idx[next_entity_hash_id]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration + 1)
            
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        
        return entity_weights, actived_entities

    def calculate_entity_scores_vectorized(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        """
        使用 GPU 加速的向量化版本计算实体分数
        
        使用 PyTorch 稀疏张量进行矩阵运算，支持动态剪枝
        
        Args:
            question_embedding: 问题嵌入向量
            seed_entity_indices: 种子实体索引
            seed_entities: 种子实体文本
            seed_entity_hash_ids: 种子实体 hash_id
            seed_entity_scores: 种子实体分数
            
        Returns:
            tuple: (entity_weights, actived_entities)
        """
        # 初始化实体权重
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        num_entities = len(self.entity_hash_ids)
        num_sentences = len(self.sentence_hash_ids)
        
        # 一次性计算所有句子与问题的相似度
        question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
        sentence_similarities_np = np.dot(self.sentence_embeddings, question_emb).flatten()
        
        # 转换为 torch 张量并移到 GPU
        sentence_similarities = torch.from_numpy(sentence_similarities_np).float().to(self.device)
        
        # 跟踪已使用的句子（去重）
        used_sentence_mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
        
        # 初始化种子实体分数为稀疏张量
        seed_indices = torch.tensor([[idx] for idx in seed_entity_indices], dtype=torch.long).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.device
        ).coalesce()
        
        # 维护稠密累加器用于总分
        entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
        entity_scores_dense.scatter_(0, torch.tensor(seed_entity_indices, device=self.device), 
                                     torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.device))
        
        # 初始化 actived_entities
        actived_entities = {}
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 0)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        
        current_entity_scores_sparse = entity_scores_sparse
        
        # 迭代矩阵传播
        for iteration in range(1, self.config.max_iterations):
            # 转换为稠密用于阈值操作
            current_entity_scores_dense = current_entity_scores_sparse.to_dense()
            
            # 应用阈值
            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.config.iteration_threshold, 
                current_entity_scores_dense, 
                torch.zeros_like(current_entity_scores_dense)
            )
            
            # 获取非零索引
            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(nonzero_indices) == 0:
                break
            
            # 提取非零值并创建稀疏张量
            nonzero_values = current_entity_scores_dense[nonzero_indices]
            current_entity_scores_sparse = torch.sparse_coo_tensor(
                nonzero_indices.unsqueeze(0), nonzero_values, (num_entities,), device=self.device
            ).coalesce()
            
            # 步骤 1: 稀疏实体分数 @ 稀疏 E2S 矩阵
            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.device
            ).coalesce()
            
            # E @ E2S -> 句子激活分数
            sentence_activation = torch.sparse.mm(
                self.entity_to_sentence_sparse.t(),
                current_scores_2d
            )
            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()
            
            # 应用句子去重：掩码已使用的句子
            sentence_activation = torch.where(
                used_sentence_mask,
                torch.zeros_like(sentence_activation),
                sentence_activation
            )
            
            # 步骤 2: 每个实体独立选择 top-k 句子
            selected_sentence_indices_list = []
            
            if len(nonzero_indices) > 0 and self.config.top_k_sentence > 0:
                for i, entity_idx in enumerate(nonzero_indices):
                    entity_score = nonzero_values[i]
                    
                    # 获取连接到该实体的句子
                    entity_row = self.entity_to_sentence_sparse[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]
                    
                    if len(entity_sentence_indices) == 0:
                        continue
                    
                    # 过滤已使用的句子
                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]
                    
                    if len(available_sentence_indices) == 0:
                        continue
                    
                    # 获取句子相似度
                    sentence_sims = sentence_similarities[available_sentence_indices]
                    
                    # 选择 top-k 句子
                    k = min(self.config.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[top_k_local_indices]
                        selected_sentence_indices_list.append(top_k_sentence_indices)
                
                # 合并所有选中的句子
                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)
                    
                    # 标记为已使用
                    used_sentence_mask[unique_selected_sentences] = True
                    
                    # 计算加权句子分数
                    weighted_sentence_scores = sentence_activation * sentence_similarities
                    
                    # 清零未选中的句子
                    mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.device)
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask,
                        weighted_sentence_scores,
                        torch.zeros_like(weighted_sentence_scores)
                    )
                else:
                    weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            else:
                weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.device)
            
            # 步骤 3: 加权句子 @ S2E -> 传播到下一个实体
            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(weighted_nonzero_mask, as_tuple=False).squeeze(-1)
            
            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[weighted_nonzero_indices]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack([weighted_nonzero_indices, torch.zeros_like(weighted_nonzero_indices)]),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.device
                ).coalesce()
                
                next_entity_scores_result = torch.sparse.mm(
                    self.sentence_to_entity_sparse.t(),
                    weighted_scores_2d
                )
                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.device)
            
            # 更新实体分数
            entity_scores_dense += next_entity_scores_dense
            
            # 更新 actived_entities 字典
            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(next_entity_scores_np >= self.config.iteration_threshold)[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.entity_hash_ids[entity_idx]
                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)
            
            # 准备下一次迭代的稀疏张量
            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(next_nonzero_mask, as_tuple=False).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices.unsqueeze(0), next_nonzero_values, 
                    (num_entities,), device=self.device
                ).coalesce()
            else:
                break
        
        # 转换回 numpy 进行最终处理
        entity_scores_final = entity_scores_dense.cpu().numpy()
        
        # 映射实体分数到图节点权重
        nonzero_indices = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.entity_hash_ids[entity_idx]
            entity_node_idx = self.node_name_to_vertex_idx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)
        
        return entity_weights, actived_entities

    def calculate_passage_scores(self, question_embedding, actived_entities):
        """
        计算段落分数
        
        结合 DPR 分数和实体匹配分数
        
        Args:
            question_embedding: 问题嵌入向量
            actived_entities (Dict): 激活的实体字典
            
        Returns:
            ndarray: 段落权重数组
        """
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        
        # 获取 DPR 分数
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        
        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
            
            # 计算实体匹配奖励
            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus
            
            # 组合分数：DPR 分数 + 实体奖励
            passage_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)
            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.config.passage_node_weight
        
        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        """
        稠密段落检索（DPR）
        
        Args:
            question_embedding: 问题嵌入向量
            
        Returns:
            tuple: (sorted_passage_indices, sorted_passage_scores)
        """
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores
    
    def get_seed_entities(self, question):
        """
        从问题中提取种子实体
        
        Args:
            question (str): 问题文本
            
        Returns:
            tuple: (seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores)
        """
        # 使用 NER 提取问题中的实体
        question_entities = list(self.spacy_ner.question_ner(question))
        
        if len(question_entities) == 0:
            return [], [], [], []
        
        # 编码问题实体
        question_entity_embeddings = self.config.embedding_model.encode(
            question_entities, normalize_embeddings=True, show_progress_bar=False, batch_size=self.config.batch_size
        )
        
        # 计算与实体库的相似度
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []       
        
        # 为每个问题实体找到最相似的库实体
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages):
        """
        构建索引：段落嵌入、实体识别、图构建
        
        Args:
            passages (List[str]): 段落列表
        """
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        
        # 插入段落并计算嵌入
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        
        # 加载或执行 NER
        existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(hash_id_to_passage.keys())
        
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k: hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities, new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
        
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        
        # 提取节点和边
        entity_nodes, sentence_nodes, passage_hash_id_to_entities, self.entity_to_sentence, self.sentence_to_entity = self.extract_nodes_and_edges(
            existing_passage_hash_id_to_entities, existing_sentence_to_entities
        )
        
        # 插入句子和实体嵌入
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))
        
        # 构建 hash_id 映射
        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [self.sentence_embedding_store.text_to_hash_id[s] for s in sentence]
        
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [self.entity_embedding_store.text_to_hash_id[e] for e in entities]
        
        # 添加边
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        
        # 增强图
        self.augment_graph()
        
        # 保存图
        output_graphml_path = os.path.join(self.config.working_dir, self.dataset_name, "LinearRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)   
        self.graph.write_graphml(output_graphml_path)

    def add_adjacent_passage_edges(self):
        """
        添加相邻段落之间的边
        
        基于段落索引的连续性构建边
        """
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        
        # 提取带索引的段落
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        
        # 添加相邻边
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        """
        增强图：添加节点和边
        """
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        """
        添加节点到图中
        
        包括实体节点和段落节点
        """
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()} 
        
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}
        
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        
        # 添加新节点
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)
        
        # 构建索引
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}   
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id] 
            for passage_id in passage_hash_ids 
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        """
        添加边到图中
        
        基于 node_to_node_stats 添加带权重的边
        """
        edges = []
        weights = []
        
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        """
        添加实体到段落的边
        
        权重基于实体在段落中的出现频率
        
        Args:
            passage_hash_id_to_entities (Dict): 段落 hash_id 到实体列表的映射
        """
        passage_to_entity_count = {} 
        passage_to_all_score = defaultdict(int)
        
        # 计算实体出现次数
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        
        # 计算归一化权重
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        """
        从 NER 结果中提取节点和边
        
        Args:
            existing_passage_hash_id_to_entities (Dict): 段落 hash_id 到实体列表的映射
            existing_sentence_to_entities (Dict): 句子到实体列表的映射
            
        Returns:
            tuple: (entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity)
        """
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence = defaultdict(set)
        sentence_to_entity = defaultdict(set)
        
        # 提取段落实体
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        
        # 提取句子实体关系
        for sentence, entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        """
        合并新旧 NER 结果
        
        Args:
            existing_passage_hash_id_to_entities: 已有段落 NER 结果
            existing_sentence_to_entities: 已有句子 NER 结果
            new_passage_hash_id_to_entities: 新增段落 NER 结果
            new_sentence_to_entities: 新增句子 NER 结果
            
        Returns:
            tuple: 合并后的结果
        """
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        """
        保存 NER 结果到 JSON 文件
        
        Args:
            existing_passage_hash_id_to_entities (Dict): 段落 hash_id 到实体列表的映射
            existing_sentence_to_entities (Dict): 句子到实体列表的映射
        """
        with open(self.ner_results_path, "w") as f:
            json.dump({
                "passage_hash_id_to_entities": existing_passage_hash_id_to_entities, 
                "sentence_to_entities": existing_sentence_to_entities
            }, f)
