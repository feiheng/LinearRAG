# LinearRAG 主入口文件
# 负责解析命令行参数、加载模型和数据集、执行问答任务并评估结果

import argparse
import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
import os
import warnings
from src.evaluate import Evaluator
from src.utils import LLM_Model
from src.utils import setup_logging
from datetime import datetime

# 设置 CUDA 可见设备为 4 号 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# 忽略警告信息
warnings.filterwarnings('ignore')

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有配置参数的对象
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf", help="使用的 Spacy 命名实体识别模型")
    parser.add_argument("--embedding_model", type=str, default="model/all-mpnet-base-v2", help="使用的嵌入模型路径")
    parser.add_argument("--dataset_name", type=str, default="novel", help="使用的数据集名称")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="使用的大语言模型")
    parser.add_argument("--max_workers", type=int, default=16, help="最大并行工作线程数")
    parser.add_argument("--max_iterations", type=int, default=3, help="图搜索的最大迭代次数")
    parser.add_argument("--iteration_threshold", type=float, default=0.4, help="迭代阈值，低于此分数的实体不再扩展")
    parser.add_argument("--passage_ratio", type=float, default=2, help="段落得分的权重比例")
    parser.add_argument("--top_k_sentence", type=int, default=3, help="每个实体选择的顶级句子数量")
    parser.add_argument("--use_vectorized_retrieval", action="store_true", help="使用向量化矩阵检索而非 BFS 迭代")
    return parser.parse_args()


def load_dataset(dataset_name): 
    """
    加载数据集，包括问题和文本块
    
    Args:
        dataset_name (str): 数据集名称
        
    Returns:
        tuple: (questions, passages) 问题列表和段落列表
    """
    # 加载问题文件
    questions_path = f"dataset/{dataset_name}/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    # 加载文本块文件
    chunks_path = f"dataset/{dataset_name}/chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # 为每个文本块添加索引前缀
    passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    return questions, passages

def load_embedding_model(embedding_model):
    """
    加载嵌入模型
    
    Args:
        embedding_model (str): 嵌入模型路径
        
    Returns:
        SentenceTransformer: 加载的嵌入模型
    """
    embedding_model = SentenceTransformer(embedding_model, device="cuda")
    return embedding_model

def main():
    """
    主函数：执行完整的 LinearRAG 流程
    1. 解析参数
    2. 加载模型和数据集
    3. 构建索引
    4. 执行问答
    5. 评估结果
    """
    # 生成时间戳用于结果保存
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载嵌入模型
    embedding_model = load_embedding_model(args.embedding_model)
    
    # 加载数据集
    questions, passages = load_dataset(args.dataset_name)
    
    # 设置日志记录
    setup_logging(f"results/{args.dataset_name}/{time_str}/log.txt")
    
    # 初始化大语言模型
    llm_model = LLM_Model(args.llm_model)
    
    # 创建 LinearRAG 配置对象
    config = LinearRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        use_vectorized_retrieval=args.use_vectorized_retrieval
    )
    
    # 初始化 LinearRAG 模型
    rag_model = LinearRAG(global_config=config)
    
    # 构建索引（包括段落嵌入、实体识别、图构建）
    rag_model.index(passages)
    
    # 执行问答任务
    questions = rag_model.qa(questions)
    
    # 保存预测结果
    os.makedirs(f"results/{args.dataset_name}/{time_str}", exist_ok=True)
    with open(f"results/{args.dataset_name}/{time_str}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    
    # 评估结果
    evaluator = Evaluator(llm_model=llm_model, predictions_path=f"results/{args.dataset_name}/{time_str}/predictions.json")
    evaluator.evaluate(max_workers=args.max_workers)

if __name__ == "__main__":
    main()
