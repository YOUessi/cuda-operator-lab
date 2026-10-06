# P01 原始证据

本目录三个 part*.b64 按编号拼接后是 gzip 压缩 JSON 的 Base64 表示。解压后的 JSON 包含全部 CPU 源码片段执行结果和五份测试/执行日志，不是截取的有利样本。没有 GPU 时延数据。

## 完整性

| 对象 | 字节数 | SHA256 |
|---|---:|---|
| part01.b64 | 2020 | ad0c277f9d9831495a7e181d10b8d9c0d3d7a6f2108dc883a4bf6645ff9eb183 |
| part02.b64 | 2020 | fb25c5440355fbe786c1365ed2127192180eaa8399c905c18a382b1486460e4e |
| part03.b64 | 2093 | 0882529088569ef628ee521d240d5dd0336be5f7482447749f2e09804b9d92e4 |
| 拼接 Base64 | 6133 | c7b0971e96e67ab4a1c37a516f1fe5f6433d02014b939cb1d61ac0ea4f07938d |
| 解压 JSON 包 | 72255 | 77b419d62e53f136ee8e84b4c96f493fe5d8f5e7f15c7101f5720607481a4aa5 |

解压包中的原文件：

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| cpu_source_probe01.json | 43127 | 224086b5ec8ffbd83f034ac8e9bbb91f8c75c82a52f281a70bee3afaa444e0c7 |
| contract_red.log | 12669 | 3667b333f1069f1d3001b1774ce7609b956d7183054a2fc76250edc2fa4a6ec7 |
| contract_green.log | 1022 | ae2c54bb6f7e7b2dca8838a0cbbfd50b876ff375803f94988e96ee2d7098c436 |
| probe_red.log | 7560 | 932c602acdf7388726ebca5f38d3b5a51111a8fa4a0b5cc8817680ddb65e97cc |
| unit_green.log | 1565 | 52a997bcb79c634c3ce143ac3a5dbeaf248a0bb043ec786cd32df2d7bf5b1eec |
| probe01.log | 1373 | 330b11986a1551650f9315503d501fc04bcabdf5cec93941663a88f7332749d4 |

上述分段已从 GitHub 重新 fetch，在 Tang 拼接、解压并与全部六个原文件逐字节比较通过。正式探针的源提交为 7ebac09c084070b90eb56ad683f6e7fba61636f1。

## 查看和还原

从仓库根目录运行下面的标准库代码；只写到一个新目录，拒绝覆盖已有文件：

```python
import base64
import gzip
import hashlib
import json
from pathlib import Path

root = Path('reports/research/rl-output-head-audit/P01/evidence')
encoded = b''.join((root / f'part{i:02}.b64').read_bytes() for i in (1, 2, 3))
assert hashlib.sha256(encoded).hexdigest() == 'c7b0971e96e67ab4a1c37a516f1fe5f6433d02014b939cb1d61ac0ea4f07938d'
raw = gzip.decompress(base64.b64decode(encoded))
assert hashlib.sha256(raw).hexdigest() == '77b419d62e53f136ee8e84b4c96f493fe5d8f5e7f15c7101f5720607481a4aa5'
bundle = json.loads(raw)
expected = {'cpu_source_probe01.json', 'contract_red.log', 'contract_green.log',
            'probe_red.log', 'unit_green.log', 'probe01.log'}
assert set(bundle['files']) == expected
out = Path('/tmp/rl-output-head-p01-evidence')
out.mkdir(exist_ok=False)
for name, text in bundle['files'].items():
    with (out / name).open('xb') as f:
        f.write(text.encode('utf-8'))
print(out)
```

## 发布问题记录

第一次整段 Base64 文本传输在 24fecf9 提交中发生人工转录/截断，验证发现 3501 字节与原 6133 字节不符且无法解码。原始数据没有改变。随后改为三个较短分段传输，在 96de583 上重新读取并核验全部六份原文件一致，f098e4f 删除了错误单文件；Git 历史保留失败记录。不得使用失败版本作证据。

运行日志保留一个把 requires_grad 张量转换为 Python scalar 的警告，它来自报告最大误差的代码，不在被测试的上游定义中。本轮没有计时，不以该警告推导性能或CUDA结果。
