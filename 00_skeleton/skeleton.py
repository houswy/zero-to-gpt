# %% [markdown]
# # 阶段 0：训练一个语言模型的六段骨架
#
# 目标：不写代码，只读和改参数。先跑 122 行的二元模型，给六段骨架做标注；再跑 Transformer 版，看懂两者只有"模型"一段不同；最后拧三个旋钮，用损失曲线建立手感。
#
# 代码改编自 Karpathy 的 ng-video-lecture 仓库（bigram.py、gpt.py，MIT 许可），结构保持一致，只做了分段和参数化。
#
# 用时：二元模型 1 分钟；小 GPT 在 CPU 上约 3～5 分钟，Colab T4 不到 1 分钟；三个旋钮实验 CPU 约 15～20 分钟。

# %%
# ===== 准备：读取数据 =====
import os, time, math, urllib.request
import torch, torch.nn as nn
from torch.nn import functional as F
import sys
SMOKE = '--smoke' in sys.argv                  # 自测开关：极小规模跑通流程（命令行加 --smoke）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
CANDIDATES = ['../reference/ng-video-lecture/input.txt', 'reference/ng-video-lecture/input.txt', 'input.txt']
path = next((p for p in CANDIDATES if os.path.exists(p)), None)
if path is None:
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt', 'input.txt'); path = 'input.txt'
text = open(path, encoding='utf-8').read()
print('设备', device, '| 语料字符数', len(text)); print(text[:200])

# %% [markdown]
# ## 六段骨架总览
#
# | 环节 | 做什么 | 本笔记本里的位置 | 到了大模型里变成什么 |
# |---|---|---|---|
# | 1 数据 | 把语料读进来 | 上一格 | 数万亿 token 的网页语料 |
# | 2 分词 | 文本与整数互相转换 | 第 2 格 | BPE 分词器（阶段 5 自己写） |
# | 3 切批 | 随机截一段作输入，右移一位作目标 | 第 3 格 | 完全一样，外加分布式切分 |
# | 4 模型与损失 | 预测下一个符号，交叉熵 | 第 4 格 | 模型换成 Transformer，损失不变 |
# | 5 优化循环 | 前向、反传、更新，循环到底 | 第 5 格 | 加调度、混合精度、梯度裁剪，循环本身不变 |
# | 6 采样 | 一次生成一个符号再喂回去 | 第 6 格 | 加 KV 缓存与采样策略，原理不变 |

# %%
# ===== 环节 2：分词（字符级，词表就是出现过的字符）=====
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}; itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)
print('词表大小', vocab_size); print(''.join(chars))
print(encode('hii there'), '->', decode(encode('hii there')))
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]
print('训练集', len(train_data), '验证集', len(val_data))

# %%
# ===== 环节 3：切批（输入 x 与目标 y 的关系是本阶段最重要的一件事）=====
def get_batch(split, block_size, batch_size):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
xb, yb = get_batch('train', block_size=8, batch_size=1)
print('x:', xb[0].tolist(), decode(xb[0].tolist()))
print('y:', yb[0].tolist(), decode(yb[0].tolist()))
for t in range(8):
    print(f'看到 {xb[0, :t + 1].tolist()} 时，要预测 {yb[0, t].item()}')
# 思考：一个长度为 8 的样本里实际上藏着 8 个训练例子，这就是 y 右移一位的原因。

# %%
# ===== 环节 4：模型与损失（二元模型：只看前一个字符）=====
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)   # 第 i 行就是"i 后面接谁"的打分
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)                            # (B, T, C)
        if targets is None: return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))  # 交叉熵：预测下一个符号
        return logits, loss
    def generate(self, idx, max_new_tokens):                                # 环节 6 的实现也在模型里
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)                     # 只看最后一个位置
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)      # 采样一个，接到后面
        return idx

# %%
# ===== 环节 5：优化循环（所有模型都是这一个循环）=====
@torch.no_grad()
def estimate_loss(model, block_size, batch_size, eval_iters):
    model.eval(); out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split, block_size, batch_size)); losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(); return out

def train_loop(model, block_size, batch_size, max_iters, lr, eval_interval, eval_iters, tag=''):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist, t0 = [], time.time()
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            l = estimate_loss(model, block_size, batch_size, eval_iters); hist.append((it, l['train'], l['val']))
            print(f'{tag} step {it:5d}: train {l["train"]:.4f}  val {l["val"]:.4f}', flush=True)
        xb, yb = get_batch('train', block_size, batch_size)
        _, loss = model(xb, yb)                 # 前向
        opt.zero_grad(set_to_none=True)
        loss.backward()                         # 反传（阶段 1 要手写的就是这一行背后的事）
        opt.step()                              # 更新
    print(f'{tag} 用时 {time.time() - t0:.0f} 秒', flush=True); return hist

bigram = BigramLanguageModel(vocab_size).to(device)
print('参数量', sum(p.numel() for p in bigram.parameters()))
hist_bigram = train_loop(bigram, block_size=8, batch_size=32, max_iters=30 if SMOKE else 3000, lr=1e-2,
                         eval_interval=10 if SMOKE else 300, eval_iters=5 if SMOKE else 200, tag='bigram')

# %%
# ===== 环节 6：采样 =====
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)     # 从换行符开始
print(decode(bigram.generate(ctx, max_new_tokens=300)[0].tolist()))
# 观察：有英文味但不成词。二元模型只看前一个字符，这是它的上限，不是训练不够。

# %% [markdown]
# ## 换成 Transformer：只有"环节 4"变了
#
# 下面这组类来自 gpt.py。先当黑箱：`Head`、`MultiHeadAttention` 接收一段序列、输出同样长度的序列，让每个位置能看到自己之前的位置；`FeedForward` 对每个位置独立做一次非线性变换；`Block` 把两者用残差连接串起来；`GPTLanguageModel` 叠若干个 Block，再加位置编码。内部原理留到阶段 3 拆解。
#
# 请对照上面的 bigram：数据、分词、切批、优化循环、采样五段原封不动，只有模型类换了。

# %%
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False); self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))     # 因果掩码：看不到未来
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // n_head, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd); self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout); self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))          # 残差连接：深层网络能训练的关键
        return x + self.ffwd(self.ln2(x))

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd); self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        if targets is None: return logits, None
        B, T, C = logits.shape
        return logits, F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])                  # 只保留最近 block_size 个符号
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

# %%
# ===== 小 GPT 基线：CPU 约 3～5 分钟 =====
BASE = dict(batch_size=32, block_size=64, max_iters=1000, lr=3e-4, eval_interval=100, eval_iters=50,
            n_embd=128, n_head=4, n_layer=4, dropout=0.0)
if SMOKE: BASE.update(max_iters=20, eval_interval=10, eval_iters=3)
RUNS = {}
def run(tag, **over):
    cfg = {**BASE, **over}; torch.manual_seed(1337)
    if SMOKE: cfg.update(max_iters=3, eval_interval=2, eval_iters=1, batch_size=4, block_size=min(cfg['block_size'], 16), n_embd=min(cfg['n_embd'], 32), n_head=2, n_layer=min(cfg['n_layer'], 1))
    model = GPTLanguageModel(vocab_size, cfg['n_embd'], cfg['n_head'], cfg['n_layer'], cfg['block_size'], cfg['dropout']).to(device)
    print(f'[{tag}] 参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M，配置 {over or "基线"}')
    hist = train_loop(model, cfg['block_size'], cfg['batch_size'], cfg['max_iters'], cfg['lr'], cfg['eval_interval'], cfg['eval_iters'], tag)
    RUNS[tag] = (hist, model); return model
gpt = run('base')
print(decode(gpt.generate(ctx, max_new_tokens=300)[0].tolist()))
# 对照 bigram 的输出：开始出现完整单词和类似剧本的格式。损失大约从 4.2 降到 2 以下。

# %% [markdown]
# ## 拧三个旋钮
#
# 每个实验只改一个参数，其余与基线相同。看三样东西：最终验证损失、用时、生成文本。
# CPU 上三组共约 15～20 分钟；嫌慢可以把 BASE 里的 max_iters 改成 500。

# %%
# 旋钮 1：上下文长度。block_size 从 64 改成 8，模型每次只能看 8 个字符。
m = run('block_size=8', block_size=8)
print(decode(m.generate(ctx, max_new_tokens=200)[0].tolist()))

# %%
# 旋钮 2：模型规模。层数与宽度各翻倍，参数量约 4 倍，注意用时。
m = run('2x size', n_layer=8, n_embd=256)
print(decode(m.generate(ctx, max_new_tokens=200)[0].tolist()))

# %%
# 旋钮 3：学习率。乘 10 与除 10 各跑一次。
run('lr x10', lr=3e-3)
run('lr /10', lr=3e-5)

# %%
# ===== 汇总：验证损失曲线 =====
summary = {k: (v[0][-1][2], v[0][-1][1]) for k, v in RUNS.items()}
print('最终验证损失 / 训练损失：')
for k, (vl, tl) in summary.items(): print(f'  {k:14s} val {vl:.3f}  train {tl:.3f}')
try:
    import matplotlib
    if SMOKE: matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for k, (hist, _) in RUNS.items():
        plt.plot([h[0] for h in hist], [h[2] for h in hist], marker='o', ms=3, label=k)
    plt.xlabel('iteration'); plt.ylabel('val loss'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig('val_loss.png', dpi=150); print('曲线已保存为 val_loss.png')
    if not SMOKE: plt.show()        # 脚本模式下 plt.show() 会阻塞等待关窗，自测时跳过
except ImportError:
    print('未安装 matplotlib，跳过画图')

# %% [markdown]
# ## 完成后写进 NOTES.md 的四件事
# 1. 用自己的话写六段骨架各做什么，每段一句。
# 2. 为什么 y 是 x 右移一位？一个长度为 T 的样本里有多少个训练例子？
# 3. 三个旋钮各自的现象：上下文变短后文本哪里变差；模型翻倍后损失降了多少、时间长了多少；学习率乘 10 和除 10 分别发生了什么。
# 4. 对照 bigram 与 GPT 的代码，列出"完全没改"的行有哪些。这些就是训练的骨架。
