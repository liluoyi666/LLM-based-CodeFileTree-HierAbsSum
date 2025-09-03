import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os

warnings.filterwarnings("ignore")


class ProjectSummaryRAG:
    """项目总结RAG检索系统"""

    def __init__(self, summaries_data: List[Dict[str, Any]], model_path: str = './local_model'):
        """
        初始化RAG系统

        参数:
            summaries_data: 从JSON文件加载的总结数据
            model_path: 本地模型路径
        """
        self.summaries = summaries_data

        # 检查模型是否存在，如果不存在则下载
        if not os.path.exists(model_path):
            print(f"本地模型不存在，正在下载模型到 {model_path}...")
            model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            model.save(model_path)
            print("模型下载完成!")

        # 从本地加载模型
        self.model = SentenceTransformer(model_path)

        # 准备文本和元数据
        self.texts = []
        self.metadata = []

        for item in summaries_data:
            # 组合名称和总结作为检索文本
            text = f"{item['name']}: {item['summary']}"
            self.texts.append(text)
            self.metadata.append({
                "name": item["name"],
                "path": item["path"],
                "file_type": item["file_type"],
                "full_summary": item["summary"]
            })

        # 生成嵌入向量
        print("正在生成文本嵌入...")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)
        print(f"已为 {len(self.texts)} 个文档生成嵌入向量")

    def search(self, query: str, top_k: int = 5, threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        搜索与查询最相关的项目总结

        参数:
            query: 搜索查询
            top_k: 返回最相关结果的数量
            threshold: 相似度阈值，低于此值的结果将被过滤

        返回:
            包含相关总结和元数据的列表
        """
        # 生成查询嵌入
        query_embedding = self.model.encode([query])

        # 计算相似度
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # 获取最相似的结果
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                results.append({
                    "similarity": similarity,
                    "metadata": self.metadata[i],
                    "text": self.texts[i]
                })

        # 按相似度排序并返回前k个结果
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def print_results(self, results: List[Dict[str, Any]]):
        """格式化打印搜索结果"""
        if not results:
            print("未找到相关结果")
            return

        print(f"\n找到 {len(results)} 个相关结果:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            print(f"{i}. 相似度: {result['similarity']:.4f}")
            print(f"   名称: {result['metadata']['name']}")
            print(f"   路径: {result['metadata']['path']}")
            print(f"   类型: {result['metadata']['file_type']}")
            print(f"   简介: {result['metadata']['full_summary'][:200]}...")  # 只显示前200个字符
            print("-" * 80)


def load_summaries_from_json(json_file_path: str) -> List[Dict[str, Any]]:
    """从JSON文件加载总结数据"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功从 {json_file_path} 加载 {len(data)} 条总结记录")
        return data
    except Exception as e:
        print(f"加载总结数据失败: {str(e)}")
        return []


# 使用示例
if __name__ == "__main__":
    # 加载总结数据
    summaries_file = "datas/AI-win11_summaries.json"  # 替换为您的总结JSON文件路径
    summaries_data = load_summaries_from_json(summaries_file)

    if not summaries_data:
        print("无法加载总结数据，请检查文件路径")
        exit(1)

    # 指定本地模型路径
    local_model_path = "./local_model"  # 可以修改为您想要的路径

    # 初始化RAG系统
    rag_system = ProjectSummaryRAG(summaries_data, model_path=local_model_path)

    # 交互式搜索
    print("\n项目总结检索系统已就绪!")
    print("输入您的查询或输入'quit'退出")

    while True:
        query = input("\n请输入搜索查询: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("程序结束!")
            break

        if not query:
            continue

        # 执行搜索
        results = rag_system.search(query, top_k=5)

        # 显示结果
        rag_system.print_results(results)
