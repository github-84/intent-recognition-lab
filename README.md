# 意图识别技术与实战

这是一个轻量级中文意图识别项目，覆盖从语料构建、模型训练、评估报告、API 服务到网页演示的完整流程。

第一版使用字符级 `TF-IDF + LogisticRegression`，不依赖大模型或在线 API，适合在普通 CPU 环境中快速跑通。

## 项目结构

```text
intent-recognition-lab/
  data/intents.csv              # 中文意图识别样例数据
  src/train.py                  # 训练与评估脚本
  src/predict.py                # 本地推理模块
  src/api.py                    # FastAPI 服务
  web/index.html                # 网页测试台
  models/                       # 训练后生成模型
  reports/                      # 训练后生成评估报告
  requirements.txt              # Python 依赖
```

## 1. 创建并进入虚拟环境

```powershell
cd D:\86157\desktop\myweb\intent-recognition-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. 安装依赖

```powershell
pip install -r requirements.txt
```

## 3. 训练模型

```powershell
python src/train.py
```

训练完成后会生成：

- `models/intent_model.joblib`
- `reports/classification_report.txt`
- `reports/confusion_matrix.csv`

## 4. 命令行测试预测

```powershell
python src/predict.py
```

输入示例：

```text
我想退货
帮我查一下订单
转人工客服
```

## 5. 启动 API 服务

```powershell
uvicorn src.api:app --reload
```

健康检查：

```text
http://127.0.0.1:8000/health
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

## 6. 打开网页测试台

直接用浏览器打开：

```text
D:\86157\desktop\myweb\intent-recognition-lab\web\index.html
```

网页会请求本地 API：

```text
http://127.0.0.1:8000/predict
```

因此需要先保持 `uvicorn src.api:app --reload` 正在运行。

## API 示例

请求：

```json
{
  "text": "我想退货"
}
```

响应：

```json
{
  "text": "我想退货",
  "intent": "refund",
  "intent_name": "退款售后",
  "confidence": 0.91,
  "probabilities": {
    "refund": 0.91,
    "query_order": 0.03
  },
  "probability_names": {
    "refund": "退款售后",
    "query_order": "查询订单"
  }
}
```

## 可扩展方向

- 增加更多业务意图和真实客服语料
- 加入数据分析脚本，统计类别分布与文本长度
- 对比 `LinearSVC`、`FastText`、`TextCNN`、`BERT` 等模型
- 将模型导出为 ONNX，进一步优化部署性能
- 增加低置信度兜底逻辑，例如转人工客服
