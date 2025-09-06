import os
import fnmatch
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from api import create_client, get_response_from_llm  # 导入API函数
import prompts  # 导入提示词


class FileTreeNode:
    """文件树节点类"""

    def __init__(self, name: str, path: str, file_type: str, depth: int):
        self.name = name
        self.path = path
        self.file_type = file_type  # 改为文件类型字符串
        self.depth = depth
        self.children: List[FileTreeNode] = []
        self.summary: Optional[str] = None
        self.content: Optional[str] = None
        self.is_empty = False  # 新增属性：标记是否为空

    def add_child(self, child_node):
        """添加子节点"""
        self.children.append(child_node)

    def to_dict(self) -> Dict[str, Any]:
        """将节点转换为字典形式"""
        return {
            "name": self.name,
            "path": self.path,
            "file_type": self.file_type,  # 修改为文件类型
            "depth": self.depth,
            "summary": self.summary,
            "is_empty": self.is_empty,  # 新增字段
            "children": [child.to_dict() for child in self.children]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileTreeNode':
        """从字典数据创建FileTreeNode实例"""
        node = cls(
            name=data['name'],
            path=data['path'],
            file_type=data['file_type'],
            depth=data['depth']
        )
        node.summary = data.get('summary')
        node.is_empty = data.get('is_empty', False)  # 新增字段

        # 递归创建子节点
        for child_data in data.get('children', []):
            child_node = cls.from_dict(child_data)
            node.add_child(child_node)

        return node

    def print_tree_visual(self, show_summary: bool = True, max_depth: Optional[int] = None,
                          indent: str = "", current_depth: int = 0,max_len:int=None,num_lines=None) -> str:
        """
        以简洁的树状结构可视化返回文件树的字符串表示，只使用缩进
        """
        # 检查是否达到最大深度限制
        if max_depth is not None and current_depth > max_depth:
            return ""

        # 构建当前节点的字符串
        result = indent + self.name

        if self.file_type == "directory":
            result = result + "\\"

        # 标记空文件/目录
        if self.is_empty:
            result += ""

        # 如果有摘要且需要显示，添加摘要信息
        if show_summary and self.summary and not self.is_empty:
            # 只取摘要的第一行（如果有多行）
            if self.file_type=="directory":
                summary_line = " ".join(self.summary.split('\n'))
            else:
                summary_line = f"\n{indent}    ".join([line for line in self.summary.split('\n') if line != ''][:num_lines])
            if max_len is not None:
                summary_line=summary_line[:max_len]+"..."
            result += f": {summary_line}"

        result += "\n"

        # 为子节点增加缩进
        new_indent = indent + "    "

        # 递归处理所有子节点
        for child in self.children:
            result += child.print_tree_visual(
                show_summary,
                max_depth,
                new_indent,
                current_depth + 1,
                max_len=max_len,
                num_lines=num_lines
            )

        return result

    def get_all_summaries(self) -> List[Dict[str, str]]:
        """
        获取所有节点的总结信息列表，用于RAG索引
        """
        summaries = []

        # 添加当前节点的总结（跳过空文件/目录）
        if self.summary and not self.is_empty:
            summaries.append({
                "name": self.name,
                "path": self.path,
                "file_type": self.file_type,
                "summary": self.summary
            })

        # 递归添加子节点的总结
        for child in self.children:
            summaries.extend(child.get_all_summaries())

        return summaries

    def save(self, filename_prefix: str = "project_tree"):
        """
        将节点的树状图和JSON表示保存到文件中
        """
        # 生成树状图
        tree_visual = self.print_tree_visual(show_summary=True)

        # 生成JSON表示
        tree_json = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

        # 获取所有总结信息
        summaries = self.get_all_summaries()
        summaries_json = json.dumps(summaries, indent=2, ensure_ascii=False)

        # 保存树状图到文本文件（用于作为LLM提示词与人类理解）
        tree_filename = f"datas/{filename_prefix}_tree.txt"
        with open(tree_filename, 'w', encoding='utf-8') as f:
            f.write(tree_visual)

        # 保存JSON到文件（用于重新加载）
        json_filename = f"datas/{filename_prefix}_structure.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            f.write(tree_json)

        # 保存总结信息到文件（用于RAG）
        summaries_filename = f"datas/{filename_prefix}_summaries.json"
        with open(summaries_filename, 'w', encoding='utf-8') as f:
            f.write(summaries_json)

        print(f"树状图已保存到: {tree_filename}")
        print(f"JSON结构已保存到: {json_filename}")
        print(f"总结信息已保存到: {summaries_filename}")

        return tree_filename, json_filename, summaries_filename


class ProjectSummarizer:
    """项目总结工具"""

    def __init__(self, max_depth: int = 5, model: str = "deepseek-chat"):
        self.max_depth = max_depth
        self.text_extensions = {'.txt', '.py', '.js', '.java', '.c', '.cpp', '.h', '.html', '.css', '.json', '.xml', '.md', '.rst'}
        self.model = model
        self.client, self.model_name = create_client(model)

    def is_text_file(self, file_path: str) -> bool:
        """检查文件是否为文本文件"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.text_extensions

    def get_file_type(self, path: str) -> str:
        """获取文件类型"""
        if os.path.isfile(path):
            return os.path.splitext(path)[1].lower()  # 返回文件扩展名
        else:
            return "directory"  # 目录返回"directory"

    def read_file_content(self, file_path: str) -> Tuple[str, bool]:
        """读取文件内容，返回内容和是否为空文件的标志"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                is_empty = len(content.strip()) == 0
                return content, is_empty
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    is_empty = len(content.strip()) == 0
                    return content, is_empty
            except:
                return f"无法读取文件: {file_path}", True
        except Exception as e:
            return f"读取文件时出错: {str(e)}", True

    def is_directory_empty(self, dir_path: str) -> bool:
        """检查目录是否为空（忽略隐藏文件）"""
        try:
            for item in os.listdir(dir_path):
                if not item.startswith('.'):  # 忽略隐藏文件和目录
                    return False
            return True
        except:
            return True  # 如果无法访问，视为空

    def summarize_file(self, file_path: str, content: str) -> str:
        """使用AI总结文件内容"""
        # 截断过长的内容以避免token限制
        max_content_length = 8000  # 根据模型token限制调整
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [内容已截断]"

        # 准备提示词
        prompt = prompts.FILE_SUMMARY_PROMPT.format(
            file_name=os.path.basename(file_path),
            file_path=file_path,
            file_content=content
        )

        try:
            # 调用AI模型
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model_name,
                system_message="""你是一个资深程序员，了解面向对象的思想，擅长分析和总结代码项目结构。
如果文件为代码，按以下格式总结，控制在256字以内，不重要或被嵌套的类或函数可不单独总结:
```
该代码大体实现了...，调用了...
class a: 这是一个类，实现了...
class b: 这是一个类，实现了...
def c: 这是一个函数，实现了...
X=1: 这是一个宏。
```
非代码文件无需使用该格式，直接使用自然语言概括其内容
""",
                print_debug=False,
                temperature=0.5
            )
            print("已总结", file_path, response)
            return response
        except Exception as e:
            return f"AI总结失败: {str(e)}"

    def summarize_directory(self, dir_path: str, node: FileTreeNode,visibility = 2) -> str:
        """使用AI总结目录内容"""
        # 获取子节点的简要信息
        children_info = []
        for child in node.children:
            if child.is_empty:
                continue  # 跳过空文件/目录

            child_info = f"{child.name} ({child.file_type})"
            if child.summary:
                # 只取摘要的第一行
                summary_line = child.summary.split('\n')[0]
                child_info += f": {summary_line}"
            children_info.append(child_info)

        # 如果没有非空子节点，返回空目录提示
        if not children_info:
            return ""

        # children_summary = "\n".join(children_info)
        children_summary = "\n"+node.print_tree_visual(show_summary=True,max_depth=visibility)

        prompt = prompts.DIRECTORY_SUMMARY_PROMPT.format(
            dir_name=os.path.basename(dir_path),
            dir_path=dir_path,
            children_summaries=children_summary
        )

        try:
            # 调用AI模型
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.client,
                model=self.model_name,
                system_message="你是一个资深程序员，了解面向对象的思想，擅长分析和总结代码项目结构。",
                print_debug=False,
                temperature=0.5
            )
            print("*已总结", children_summary, response)
            return response
        except Exception as e:
            # 失败时回退到简单统计
            file_count = sum(1 for child in node.children if child.file_type != "directory" and not child.is_empty)
            dir_count = sum(1 for child in node.children if child.file_type == "directory" and not child.is_empty)
            return f"AI总结失败，包含 {file_count} 个文件, {dir_count} 个子目录。错误: {str(e)}"

    def build_tree(self, root_path: str, current_depth: int = 0,visibility = 2) -> Optional[FileTreeNode]:
        """构建文件树"""
        if current_depth > self.max_depth:
            return None

        root_name = os.path.basename(root_path)
        file_type = self.get_file_type(root_path)
        node = FileTreeNode(root_name, root_path, file_type, current_depth)

        if file_type != "directory":
            # 处理文件
            if self.is_text_file(root_path):
                content, is_empty = self.read_file_content(root_path)
                node.content = content
                node.is_empty = is_empty

                # 只有非空文本文件才进行总结
                if not is_empty:
                    node.summary = self.summarize_file(root_path, content)
                else:
                    node.summary = ""
            else:
                node.summary = f"非文本文件: {root_name}"
                node.is_empty = True  # 非文本文件视为空（不进行内容分析）
        else:
            # 处理目录
            try:
                # 检查目录是否为空
                is_empty_dir = self.is_directory_empty(root_path)
                node.is_empty = is_empty_dir

                if is_empty_dir:
                    node.summary = ""
                else:
                    for item in os.listdir(root_path):
                        # 跳过以点开头的文件和目录（如.git、.idea等）
                        if item.startswith('.'):
                            continue

                        item_path = os.path.join(root_path, item)
                        child_node = self.build_tree(item_path, current_depth + 1,visibility = 2)
                        if child_node:
                            node.add_child(child_node)

                    # 只有非空目录才进行总结
                    if not is_empty_dir:
                        node.summary = self.summarize_directory(root_path, node,visibility)
            except PermissionError:
                node.summary = "无权限"
                node.is_empty = True
            except Exception as e:
                node.summary = f"遍历目录时出错: {str(e)}"
                node.is_empty = True

        return node

    def load_tree_from_json(self, json_file_path: str) -> Optional[FileTreeNode]:
        # 从JSON文件加载树结构

        try:
            with open(f"datas/{json_file_path}_structure.json", 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 使用FileTreeNode的类方法从字典重建树
            return FileTreeNode.from_dict(data)
        except Exception as e:
            print(f"从JSON加载树结构失败: {str(e)}")
            return None

    def generate_project_summary(self, root_path: str) -> Dict[str, Any]:
        """生成项目总结"""
        if not os.path.exists(root_path):
            return {"error": "路径不存在"}

        tree = self.build_tree(root_path)
        if not tree:
            return {"error": "无法构建文件树"}

        return tree.to_dict()


# 使用示例
if __name__ == "__main__":
    # 创建总结器实例，设置最大深度为3，指定模型
    summarizer = ProjectSummarizer(max_depth=3, model="deepseek-chat")

    # 项目位置与项目名称
    project_path = r"D:\py_project\AI-win11-Administrator"
    name = "AI-win11"

    # 开始构建树，信息视野2
    tree = summarizer.build_tree(project_path,visibility=2)

    # 可视化展示
    print(tree.print_tree_visual())

    # 保存结果至datas/my_project
    tree.save(name)

    # 指定名称重新加载树
    tree = summarizer.load_tree_from_json(name)

    # 展示深度2，每个摘要只最多展示一行
    print(tree.print_tree_visual(max_depth=2,num_lines=1))