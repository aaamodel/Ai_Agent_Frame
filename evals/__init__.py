# -*- coding: utf-8 -*-
"""evals 包入口。

刻意**不在此处 import 任何重型依赖**（llama_index / pymilvus / fastapi 等），
以保证 ``pytest evals/test_metrics.py`` 在只装了 pytest 的干净环境里也能跑起来
（指标函数是纯标准库实现，这是可复现性的基础）。
"""
