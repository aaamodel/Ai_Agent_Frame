# -*- coding: utf-8 -*-
"""evals 包入口。

刻意**不在此处 import 任何重型依赖**（llama_index / pymilvus / fastapi 等），
以保证 ``pytest evals/test_metrics.py`` 在只装了 pytest 的干净环境里也能跑起来
（指标函数是纯标准库实现，这是可复现性的基础）。
"""

# Windows：强制 SelectorEventLoop。默认的 ProactorEventLoop 在网络抖动（socket 于
# 重叠读取未完成时被中止，WinError 995）时，IOCP 完成回调可能对一个已结束的 future
# 二次 set_exception → InvalidStateError 直接打死事件循环（asyncio.run 整体崩溃，
# 实测于 eval T07：整条 --run-all 在出报告前进程退出）。SelectorEventLoop 无重叠 I/O，
# 从根上消除该竞态；全仓库仅有同步 subprocess.run（不依赖 asyncio 子进程），可安全切换。
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

