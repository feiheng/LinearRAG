# LinearRAG 配置模块
# 定义 LinearRAG 系统的所有配置参数

from dataclasses import dataclass
from src.utils import LLM_Model

@dataclass
class LinearRAGConfig:
    """
    LinearRAG 配置类
    
    包含系统运行所需的所有超参数和配置选项
    
    Attributes:
        dataset_name (str): 数据集名称，用于确定数据路径和工作目录
        embedding_model (str): 嵌入模型名称，默认为 "all-mpnet-base-v2"
        llm_model (LLM_Model): 大语言模型实例，用于生成答案和评估
        chunk_token_size (int): 文本分块的 token 大小，默认 1000
        chunk_overlap_token_size (int): 分块重叠的 token 大小，默认 100
        spacy_model (str): Spacy 命名实体识别模型，默认 "en_core_web_trf"
        working_dir (str): 工作目录，用于保存索引和中间结果，默认 "./import"
        batch_size (int): 批处理大小，用于嵌入编码，默认 128
        max_workers (int): 最大并行工作线程数，默认 16
        retrieval_top_k (int): 检索时返回的顶级段落数量，默认 5
        max_iterations (int): 图搜索时实体扩展的最大迭代次数，默认 3
        top_k_sentence (int): 每个实体选择的顶级相关句子数，默认 1
        passage_ratio (float): 段落原始得分的权重系数，默认 1.5
        passage_node_weight (float): 段落在图中的节点权重，默认 0.05
        damping (float): PageRank 阻尼系数，默认 0.5
        iteration_threshold (float): 实体扩展的分数阈值，低于此值不再扩展，默认 0.5
        use_vectorized_retrieval (bool): 是否使用向量化矩阵计算（True）还是 BFS 迭代（False），默认 False
    """
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: LLM_Model = None
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    use_vectorized_retrieval: bool = False  # True 表示使用向量化矩阵计算，False 表示使用 BFS 迭代
