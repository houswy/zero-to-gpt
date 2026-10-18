# zero-to-gpt：从标量求导到可微调的 GPT

一个"学中干"的学习仓库。六个阶段，每个阶段一个带自动测试的起始笔记本：先填空实现，测试全部通过再看参考答案。终点是一个自己写出的、能加载 GPT-2 权重并微调的 Transformer。

主线跟随 Andrej Karpathy 的 *Neural Networks: Zero to Hero* 视频系列，配 Sebastian Raschka 的《从零构建大模型》（*Build a Large Language Model (From Scratch)*）做教材。全部内容可在 CPU 或 Colab 免费 GPU 上完成。

## 学习路线

| 阶段 | 目录 | 要实现的东西 | 资料 | 完成标准 | 状态 |
|---|---|---|---|---|---|
| 0 | `00_skeleton` | 跑通并标注 bigram.py 与 gpt.py 的六段训练骨架；拧三个旋钮看损失曲线 | 视频第 7 集前半；ng-video-lecture 仓库 | 能说出六段各自作用；三组对照实验的曲线与记录 | ☐ |
| 1 | `01_micrograd` | 标量自动求导引擎 `Value`，两层 MLP | 视频第 1 集；micrograd 仓库 | 与 PyTorch 梯度对齐；4 样本训练到全对 | ☐ |
| 2 | `02_makemore` | 字符级语言模型：二元统计 → MLP → BatchNorm → WaveNet | 视频第 2～6 集；makemore 仓库 | 手推一遍反向传播；生成像样的英文名字 | ☐ |
| 3 | `03_attention` | 自注意力、多头、因果掩码、Transformer 块 | 书第 3～4 章 | 不看答案写出 GPT 结构，形状全部能解释 | ☐ |
| 4 | `04_nanogpt` | 字符级 GPT 在莎士比亚文本上训练 | 视频第 7 集；ng-video-lecture、nanoGPT 仓库 | 能生成文本；损失曲线判断过拟合 | ☐ |
| 5 | `05_tokenizer` | BPE 分词器 | 视频第 8 集；minbpe 仓库的 exercise.md | 通过 minbpe 的测试；与 tiktoken 对比 | ☐ |
| 6 | `06_finetune` | 加载 GPT-2 权重，分类微调与 LoRA 指令微调 | 书第 5～7 章、附录 E | 一个能跟随简单指令的小模型 | ☐ |

每个阶段结束要回答起始笔记本末尾的思考题，答案写进该目录的 `NOTES.md`。

## 目录约定

```
00_skeleton/
  00_skeleton.ipynb            阶段 0：读与改参数，不写代码
01_micrograd/
  01_micrograd_starter.ipynb   起始笔记本：任务说明 + TODO 骨架 + 自动测试 + 思考题
  my_solution.ipynb            自己的实现（完成后提交）
  NOTES.md                     思考题答案、踩坑记录
02_makemore/ … 06_finetune/    同上
reference/                     上游参考实现（git 子模块，见下表）
tools/build_nb.py              把 percent 格式的 .py 拼成 .ipynb（维护起始笔记本用）
requirements.txt
```

上游参考实现以 git 子模块挂在 `reference/` 下，克隆时加 `--recurse-submodules` 即可离线阅读；它们只用于对照，不要直接改。

## 上游资料

| 资料 | 用途 | 许可 |
|---|---|---|
| [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | 视频配套讲义笔记本 | MIT |
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | 阶段 1 参考实现 | MIT |
| [karpathy/makemore](https://github.com/karpathy/makemore) | 阶段 2 参考实现与数据 | MIT |
| [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) | 阶段 4 的 bigram.py、gpt.py 与莎士比亚文本 | MIT |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 阶段 4 之后的进阶阅读 | MIT |
| [karpathy/minbpe](https://github.com/karpathy/minbpe) | 阶段 5 参考实现与练习 | MIT |
| [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 阶段 3、6 教材代码 | Apache-2.0 |

视频：YouTube 搜索 *Neural Networks: Zero to Hero*；B 站有中文字幕搬运。

## 使用方法

获取：`git clone --recurse-submodules https://github.com/houswy/zero-to-gpt.git`。环境：Python 3.10 以上，`pip install -r requirements.txt`。阶段 1～4 在 CPU 上即可；阶段 4 全量训练和阶段 6 建议 Colab 免费 GPU。

三条规则：
1. 先填 TODO，再运行测试格；测试用 PyTorch 或上游测试集作标准答案，全部通过才算完成。
2. 卡住超过 30 分钟再看上游参考实现，看完合上重写。
3. 每阶段结束回答思考题，写进 `NOTES.md`，不会的题照实写"不会"。

## 许可

本仓库自己写的代码与笔记采用 MIT 许可。起始笔记本中的测试用例改编自上游仓库，版权归原作者，许可见上表。
