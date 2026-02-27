# LinearRAG 评估模块
# 负责评估模型预测结果的准确性，支持 LLM 评估和包含匹配两种方式

import json
import os
from src.utils import normalize_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

class Evaluator:
    """
    评估器类
    
    提供两种评估方式：
    1. LLM 评估：使用大语言模型判断答案是否正确
    2. 包含匹配：检查预测答案是否包含标准答案
    """
    def __init__(self, llm_model, predictions_path):
        """
        初始化评估器
        
        Args:
            llm_model (LLM_Model): 用于评估的大语言模型实例
            predictions_path (str): 预测结果文件路径
        """
        self.llm_model = llm_model
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        """
        加载预测结果
        
        Returns:
            List[Dict]: 预测结果列表，每个元素包含 question、pred_answer、gold_answer 等
        """
        prediction_results = json.load(open(self.predictions_path))
        return prediction_results
    
    def calculate_llm_accuracy(self, pre_answer, gold_ans):
        """
        使用 LLM 评估答案是否正确
        
        Args:
            pre_answer (str): 预测答案
            gold_ans (str): 标准答案
            
        Returns:
            float: 1.0 表示正确，0.0 表示错误
        """
        system_prompt = """You are an expert evaluator. 
        """
        
        user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
        Generated answer: {pre_answer}
        Gold answer: {gold_ans}

        The generated answer should be considered correct if it:
        1. Contains the key information from the gold answer
        2. Is factually accurate and consistent with the gold answer
        3. Does not contain any contradicting information

        Respond with ONLY 'correct' or 'incorrect'.
        Response:
        """
        
        # 调用 LLM 进行评估
        response = self.llm_model.infer([
            {"role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ])
        
        # 判断结果
        if response.strip().lower() == "correct":
            return 1.0
        else:
            return 0.0

    def calculate_contain(self, pre_answers, gold_ans):
        """
        检查预测答案是否包含标准答案（精确匹配）
        
        Args:
            pre_answers (str): 预测答案
            gold_ans (str): 标准答案
            
        Returns:
            int: 1 表示包含，0 表示不包含
        """
        # 处理空答案
        if pre_answers is None or pre_answers == "" or (isinstance(pre_answers, str) and pre_answers.strip() == ""):
            return 0            
        if gold_ans is None or gold_ans == "" or (isinstance(gold_ans, str) and gold_ans.strip() == ""):
            return 0
        
        # 标准化后检查包含关系
        s1 = normalize_answer(pre_answers)
        s2 = normalize_answer(gold_ans)
        
        if s2 in s1:
            return 1
        else:
            return 0
            
    def evaluate_sig_sample(self, idx, prediction):
        """
        评估单个样本
        
        Args:
            idx (int): 样本索引
            prediction (Dict): 预测结果字典
            
        Returns:
            tuple: (idx, llm_acc, contain_acc)
        """
        pre_answer = prediction["pred_answer"]
        gold_ans = prediction["gold_answer"]
        
        # 计算 LLM 准确率
        llm_acc = self.calculate_llm_accuracy(pre_answer, gold_ans)
        
        # 计算包含匹配准确率
        contain_acc = self.calculate_contain(pre_answer, gold_ans)
        
        return idx, llm_acc, contain_acc

    def evaluate(self, max_workers):
        """
        并行评估所有样本
        
        Args:
            max_workers (int): 最大并行线程数
            
        Returns:
            tuple: (llm_accuracy, contain_accuracy) 两种评估方式的准确率
        """
        # 初始化分数数组
        llm_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)
        
        # 使用线程池并行评估
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有评估任务
            futures = {
                executor.submit(self.evaluate_sig_sample, idx, pred): idx 
                for idx, pred in enumerate(self.prediction_results)
            }

            completed = 0
            total_llm_score = 0.0
            total_contain_score = 0.0
            
            # 进度条
            pbar = tqdm(total=len(futures), desc="评估样本", unit="sample")
            
            # 处理完成的任务
            for future in as_completed(futures):
                idx, llm_acc, contain_acc = future.result()
                
                # 保存分数
                llm_scores[idx] = llm_acc
                contain_scores[idx] = contain_acc
                self.prediction_results[idx]["llm_accuracy"] = llm_acc
                self.prediction_results[idx]["contain_accuracy"] = contain_acc
                
                # 累计分数
                total_llm_score += llm_acc
                total_contain_score += contain_acc
                completed += 1
                
                # 更新进度条
                current_llm_acc = total_llm_score / completed
                current_contain_acc = total_contain_score / completed
                pbar.set_postfix({
                    'LLM_Acc': f'{current_llm_acc:.3f}',
                    'Contain_Acc': f'{current_contain_acc:.3f}'
                })
                pbar.update(1)
            
            pbar.close()

        # 计算平均准确率
        llm_accuracy = sum(llm_scores) / len(llm_scores)
        contain_accuracy = sum(contain_scores) / len(contain_scores)

        # 记录日志
        logger.info(f"评估结果:")
        logger.info(f"  LLM 准确率：{llm_accuracy:.4f} ({sum(llm_scores)}/{len(llm_scores)})")
        logger.info(f"  包含匹配准确率：{contain_accuracy:.4f} ({sum(contain_scores)}/{len(contain_scores)})")
        
        # 保存详细的预测结果（包含每个样本的评估分数）
        with open(self.predictions_path, "w", encoding="utf-8") as f:
            json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)
        
        # 保存总体评估结果
        with open(os.path.join(os.path.dirname(self.predictions_path), "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump({"llm_accuracy": llm_accuracy, "contain_accuracy": contain_accuracy}, f, ensure_ascii=False, indent=4)
        
        return llm_accuracy, contain_accuracy
