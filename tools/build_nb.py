# 把 percent 格式的 .py 单元文件拼成 Colab 可打开的 .ipynb，并对代码单元做语法检查
import json, sys, ast
out, srcs = sys.argv[1], sys.argv[2:]
text = ''.join(open(f, encoding='utf-8').read() + '\n' for f in srcs)
cells, cur = [], None
for line in text.splitlines(keepends=True):
    if line.startswith('# %% [markdown]'):
        cur = {'cell_type': 'markdown', 'metadata': {}, 'source': []}; cells.append(cur); continue
    if line.startswith('# %%'):
        cur = {'cell_type': 'code', 'metadata': {}, 'source': [], 'outputs': [], 'execution_count': None}; cells.append(cur); continue
    if cur is None: continue
    if cur['cell_type'] == 'markdown':
        line = line[2:] if line.startswith('# ') else (line[1:] if line.startswith('#') else line)
    cur['source'].append(line)
for c in cells:
    while c['source'] and not c['source'][-1].strip(): c['source'].pop()
    while c['source'] and not c['source'][0].strip(): c['source'].pop(0)
errors = 0
for i, c in enumerate(cells):
    if c['cell_type'] != 'code': continue
    code = ''.join(l for l in c['source'] if not l.lstrip().startswith('!'))
    try: ast.parse(code)
    except SyntaxError as e: errors += 1; print(f'语法错误：第 {i} 格 行 {e.lineno}: {e.msg}')
nb = {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
      'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'}},
      'nbformat': 4, 'nbformat_minor': 5}
json.dump(nb, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('cells', len(cells), 'code cells', sum(c['cell_type'] == 'code' for c in cells), 'syntax errors', errors)
