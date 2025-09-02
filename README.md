# LLM-based CodeFileTree HierAbsSum

## 项目简介

LLM-based CodeFileTree HierAbsSum 是一个创新的项目分析工具，专门设计用于帮助开发者和AI系统理解和分析代码项目结构。它通过智能分层总结技术，将面向对象编程思想与AI能力深度融合，解决了大型代码项目管理中的核心痛点。

## 核心特性

### 🎯 智能分层总结
- **自底向上的分析架构**：每个文件/目录只关注自身功能和接口
- **AI驱动的语义理解**：使用大模型生成精准的项目总结
- **层级信息封装**：上级无需了解下级具体实现，只需知道功能接口

### 🌳 可视化项目结构
- **树状结构展示**：清晰直观的项目层级关系可视化
- **可配置深度限制**：支持按需显示不同层级的项目信息
- **智能摘要显示**：每个节点附带AI生成的功能摘要

### 🔍 高级检索功能
- **RAG集成**：基于语义的代码检索系统
- **多模型支持**：兼容多种大语言模型(Anthropic、OpenAI、Google Gemini等)
- **本地向量化**：使用SentenceTransformer进行本地文本嵌入

### 💾 持久化与复用
- **JSON序列化**：完整保存项目结构和AI总结
- **快速重加载**：避免重复分析，节省计算资源
- **多格式输出**：同时生成可视化树状图和结构化数据

## 技术原理

### 分层总结架构
工具采用创新的"分层总结"方法，模仿了面向对象设计中的封装原则：

1. **文件级别分析**：AI分析单个代码文件的功能和外部接口
2. **目录级别聚合**：上级目录基于下级摘要生成更高层次的总结
3. **项目全局视图**：最终形成既保持细节又具备全局观的项目认知

### 解决的核心问题

1. **大项目处理**：避免传统方法中让AI直接阅读所有代码的低效做法
2. **位置编码稀释**：分层方法防止远程依赖关系的语义稀释
3. **精准定位**：结合树状结构和RAG，快速定位功能相关代码

## 安装与使用

### 快速开始
```bash
# 克隆项目
git clone <repository-url>
cd Project_Summary_Tool

# 安装依赖
pip install -r requirements.txt

# 运行示例
python tree.py
```

### 基本用法
```python
# 初始化总结器
summarizer = ProjectSummarizer(max_depth=3, model="deepseek-chat")

# 生成项目总结
project_path = "your/project/path"
tree = summarizer.build_tree(project_path)

# 可视化展示
print(tree.print_tree_visual(show_summary=True, max_depth=2))

# 保存结果
tree.save("my_project")
```

### RAG检索使用
```python
# 加载总结数据
summaries_data = load_summaries_from_json("my_project_summaries.json")

# 初始化RAG系统
rag_system = ProjectSummaryRAG(summaries_data)

# 语义检索
results = rag_system.search("文件上传功能", top_k=5)
rag_system.print_results(results)
```

## 应用场景

### 对于AI编程助手
- **项目上下文提供**：为编码AI提供结构化项目理解
- **精准代码定位**：快速找到相关功能模块
- **智能代码生成**：基于项目架构生成符合规范的代码

### 对于人类开发者
- **项目入职加速**：快速理解大型项目结构
- **代码审查辅助**：识别模块功能和依赖关系
- **文档自动生成**：基于AI总结生成项目文档

### 对于团队管理
- **架构可视化**：直观展示项目组织架构
- **知识传承**：保存项目结构和设计思路
- **质量评估**：通过摘要分析模块职责划分清晰度

## 支持的模型

- DeepSeek Chat
- OpenAI GPT系列
- Anthropic Claude
- Google Gemini
- 其他兼容OpenAPI的模型

## 项目结构

```
LLM-based CodeFileTree HierAbsSum/
├── tree.py              # 核心文件树构建与总结功能
├── rag.py               # RAG检索系统
├── api.py               # 多模型API统一接口
├── prompts.py           # AI提示词模板
├── datas/               # 数据保存目录
├── local_model/         # 本地模型缓存
└── README.md           # 项目说明文档
```

## 性能优势

1. **高效处理**：万行代码项目可在分钟内完成分析
2. **资源优化**：避免不必要的内容重复分析
3. **灵活配置**：可根据项目规模调整分析深度
4. **多模型支持**：充分利用各种AI模型的优势

## 未来规划

- [ ] IDE插件开发
- [ ] 实时项目监控
- [ ] 团队协作功能
- [ ] 更多代码语言支持
- [ ] 架构坏味道检测

## 贡献指南

我们欢迎各种形式的贡献！请参阅CONTRIBUTING.md了解详情。

## 许可证

本项目采用Apache 2.0开源许可证，详见LICENSE文件。

---

**Project Summary Tool** 不仅是一个技术工具，更是连接传统软件开发与AI智能编程的桥梁。它让AI真正具备了"理解"大型项目的能力，为下一代智能编程助手奠定了坚实基础。