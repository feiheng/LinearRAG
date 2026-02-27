# LinearRAG 命名实体识别模块
# 使用 Spacy 进行命名实体识别，提取段落和问题中的实体

import spacy
from collections import defaultdict
import pdb


class SpacyNER:
    """
    Spacy 命名实体识别类
    
    封装 Spacy 模型，提供批量和单条文本的实体识别功能
    """
    def __init__(self, spacy_model):
        """
        初始化 Spacy NER 模型
        
        Args:
            spacy_model (str): Spacy 模型名称，如 "en_core_web_trf"
        """
        self.spacy_model = spacy.load(spacy_model)

    def batch_ner(self, hash_id_to_passage, max_workers):
        """
        批量命名实体识别
        
        Args:
            hash_id_to_passage (Dict[str, str]): 段落 hash_id 到内容的映射
            max_workers (int): 最大并行工作线程数
            
        Returns:
            tuple: (passage_hash_id_to_entities, sentence_to_entities)
                - passage_hash_id_to_entities: 段落 hash_id 到实体列表的映射
                - sentence_to_entities: 句子到实体列表的映射
        """
        # 获取段落列表
        passage_list = list(hash_id_to_passage.values())
        
        # 计算批处理大小
        batch_size = len(passage_list) // max_workers
        
        # 使用 Spacy 的 pipe 进行批量处理
        docs_list = self.spacy_model.pipe(passage_list, batch_size=batch_size)
        
        passage_hash_id_to_entities = {}
        sentence_to_entities = defaultdict(list)
        
        # 处理每个段落的 NER 结果
        for idx, doc in enumerate(docs_list):
            passage_hash_id = list(hash_id_to_passage.keys())[idx]
            
            # 提取实体和句子
            single_passage_hash_id_to_entities, single_sentence_to_entities = self.extract_entities_sentences(doc, passage_hash_id)
            
            # 合并结果
            passage_hash_id_to_entities.update(single_passage_hash_id_to_entities)
            for sent, ents in single_sentence_to_entities.items():
                for e in ents:
                    if e not in sentence_to_entities[sent]:
                        sentence_to_entities[sent].append(e)
        
        return passage_hash_id_to_entities, sentence_to_entities
            
    def extract_entities_sentences(self, doc, passage_hash_id):
        """
        从 Spacy 文档中提取实体和句子
        
        Args:
            doc: Spacy 文档对象
            passage_hash_id (str): 段落 hash_id
            
        Returns:
            tuple: (passage_hash_id_to_entities, sentence_to_entities)
        """
        sentence_to_entities = defaultdict(list)
        unique_entities = set()
        passage_hash_id_to_entities = {}
        
        # 遍历所有识别出的实体
        for ent in doc.ents:
            # 跳过序数词和基数词
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            
            # 获取实体所在的句子
            sent_text = ent.sent.text
            ent_text = ent.text
            
            # 添加到句子 - 实体映射
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            
            # 添加到段落实体集合
            unique_entities.add(ent_text)
        
        # 保存段落的唯一实体列表
        passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)
        return passage_hash_id_to_entities, sentence_to_entities

    def question_ner(self, question: str):
        """
        对问题进行命名实体识别
        
        Args:
            question (str): 问题文本
            
        Returns:
            Set[str]: 识别出的实体集合（小写）
        """
        doc = self.spacy_model(question)
        question_entities = set()
        
        for ent in doc.ents:
            # 跳过序数词和基数词
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            # 转为小写并添加到集合
            question_entities.add(ent.text.lower())
        
        return question_entities
