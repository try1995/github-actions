# Kronos 股票预测 · GitHub Actions 定时任务

用 [Kronos](https://github.com/NeoQuasar-AI/Kronos) 时间序列模型预测 A 股未来价格，生成 K 线预测图，
通过 GitHub Actions 定时自动运行，并把 K 线图和预测摘要发送到你的邮箱。

- 每天 **08:00** 和 **11:30**（北京时间）自动运行，仅工作日（周一至周五）
- 股票代码、邮箱配置全部在 GitHub 网页界面里改，不用改代码
- 支持手动触发，随时跑一次

## 目录结构

```
github-actions/
├── .github/workflows/kronos-predict.yml   # GitHub Actions 工作流（定时调度）
├── kronos_predict.py                      # 主脚本：拉数据 -> 预测 -> 出图 -> 发邮件
├── model/                                 # Kronos 模型包（本地引用，勿删）
├── requirements.txt                       # Python 依赖
└── readme.md
```

## 在 GitHub 界面配置

### 1. 股票代码（Variables）

进入仓库 **Settings → Secrets and variables → Actions → Variables**，点 **New repository variable**：

| 名称 | 值 | 说明 |
|---|---|---|
| `STOCK_CODE` | `601601\|002185` | 股票代码，多只用 `\|` 分隔，自动识别沪/深/北交所 |
| `HF_ENDPOINT` | `https://huggingface.co` | 模型下载地址（可选）。默认官方站，GitHub 官方 runner 直连。国内自建 runner 可改成 `https://hf-mirror.com` |

不配置的话脚本会用默认值 `601601|002185`。

### 2. 邮箱（Secrets）

同样位置，切到 **Secrets** 页签，点 **New repository secret**，逐个添加：

| 名称 | 示例值 | 说明 |
|---|---|---|
| `SMTP_HOST` | `smtp.qq.com` | SMTP 服务器地址 |
| `SMTP_PORT` | `465` | SSL 用 465；STARTTLS 用 587 |
| `SMTP_PROTOCOL` | `ssl` | `ssl` 或 `starttls` |
| `SMTP_USER` | `12345678@qq.com` | 发件邮箱账号 |
| `SMTP_PASS` | （QQ 邮箱授权码） | QQ/163 用授权码，**不是登录密码** |
| `SMTP_RECIPIENT` | `you@example.com` | 收件邮箱，多个用逗号分隔 |

#### QQ 邮箱授权码怎么拿

1. 电脑登录 QQ 邮箱 → 设置 → **账户**
2. 找到 **POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV 服务**
3. 打开 **SMTP 服务**（勾选后要短信验证）
4. 验证通过后页面会显示一串 **授权码**，把它填到 `SMTP_PASS`（授权码只看得到一次，注意保存）

## 使用

- **自动运行**：工作日北京时间 08:00、11:30 自动执行（GitHub cron 为 UTC，实际运行可能延迟几分钟）
- **手动触发**：仓库 **Actions → Kronos 股票预测 → Run workflow → Run workflow**
- **本地跑**：`pip install -r requirements.txt && python kronos_predict.py`（不配置 SMTP 环境变量时只预测不发邮件，不报错）

## 常见问题

- **邮箱收到空白/报错**：确认 `SMTP_PASS` 填的是 QQ 邮箱**授权码**而非登录密码；`SMTP_PROTOCOL` 与端口匹配（465→ssl，587→starttls）。
- **收不到邮件但日志显示发送成功**：检查垃圾邮件文件夹；确认 `SMTP_RECIPIENT` 拼写。
- **GitHub Actions 不运行**：仓库超过 60 天无活动会自动禁用定时工作流，到 Actions 页面点 **Enable workflow** 重新启用。
- **报错 `KronosTokenizer.__init__() missing 16 required positional arguments`**：模型配置下载失败。脚本本地默认走国内镜像 `hf-mirror.com`，而 GitHub 官方 runner 在国外连不上镜像。工作流里已默认用官方 `https://huggingface.co`（见 `HF_ENDPOINT`），无需处理；若手动改了记得改回。
- **每次都要重新下载模型**：首次运行会下载 Kronos 模型（约几百 MB），之后走缓存，会快很多。

## 邮件内容

每封邮件包含：
- **正文**：每只股票每个周期的当前价、预测价、预测涨跌幅
- **附件**：所有股票的 K 线预测图（`pred_<代码>_<周期>_chart.png`）
