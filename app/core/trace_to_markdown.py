from __future__ import annotations
from datetime import datetime
import functools
import inspect
import json
from pathlib import Path
import sys


def deep_serialize(obj, visited=None):
    """递归展开嵌套类实例、Pydantic 模型、字典与列表，兼容复杂的 SDK 对象"""
    if visited is None:
        visited = set()

    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj

    obj_id = id(obj)
    if obj_id in visited:
        return "<Circular Reference>"
    visited.add(obj_id)

    try:
        # 兼容 Pydantic v2 / v1 模型展开
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            try:
                return deep_serialize(obj.model_dump(), visited.copy())
            except Exception:
                pass
        elif hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            try:
                return deep_serialize(obj.dict(), visited.copy())
            except Exception:
                pass

        # 基础容器
        if isinstance(obj, (list, tuple, set)):
            return [deep_serialize(item, visited.copy()) for item in obj]

        if isinstance(obj, dict):
            return {
                str(k): deep_serialize(v, visited.copy()) for k, v in obj.items()
            }

        # 带有 __dict__ 的对象 (自定义类/LangChain/LangGraph 组件)
        if hasattr(obj, "__dict__"):
            data = {"__class__": obj.__class__.__name__}
            for k, v in obj.__dict__.items():
                if not k.startswith("_"):  # 屏蔽内部变量以防触发属性计算异常
                    try:
                        data[k] = deep_serialize(v, visited.copy())
                    except Exception:
                        data[k] = f"<Unserializable: {type(v).__name__}>"
            return data

        # 带有 __slots__ 的对象
        if hasattr(obj, "__slots__"):
            data = {"__class__": obj.__class__.__name__}
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    try:
                        data[slot] = deep_serialize(
                            getattr(obj, slot), visited.copy()
                        )
                    except Exception:
                        data[slot] = f"<Unserializable: {slot}>"
            return data

        return str(obj)
    except Exception as e:
        return f"<Serialize Error: {str(e)}>"
    finally:
        visited.remove(obj_id)

def trace_to_markdown(output_file="trace_result.md"):
    """支持同步 (def) 与异步 (async def) 的低侵入变量追踪装饰器（增量去重版）"""

    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)
        target_code = func.__code__

        def create_tracer():
            line_counts = {}
            snapshots = []
            input_args = {}
            return_val_holder = {"val": None}

            last_locals = {}


            def trace_func(frame, event, arg):
                if frame.f_code != target_code:
                    return None

                lineno = frame.f_lineno

                if event == "call":
                    # 捕获函数入参（包括 self, context_schema, cache 等）
                    for k, v in frame.f_locals.items():
                        serialized_v = deep_serialize(v)
                        input_args[k] = serialized_v
                        # 初始化 known 状态，防止后续重复打印入参
                        last_locals[k] = serialized_v
                    return trace_func

                elif event == "line":
                    line_counts[lineno] = line_counts.get(lineno, 0) + 1

                    # 限制每行代码仅记录 1 次（自动屏蔽 for 循环后续重复迭代）
                    if line_counts[lineno] == 1:
                        current_locals = {}
                        diff = {}

                        for k, v in frame.f_locals.items():
                            serialized_v = deep_serialize(v)
                            current_locals[k] = serialized_v

                            # 【修复点 2】增量比对核心：只有新出现的变量，或值发生改变的变量，才会被记录
                            if k not in last_locals or last_locals[k] != serialized_v:
                                diff[k] = serialized_v

                        if diff:
                            snapshots.append({
                                "line": lineno,
                                "changes": diff
                            })

                        # 更新上一帧的状态
                        last_locals.clear()
                        last_locals.update(current_locals)
                        #last_locals = current_locals.copy()

                elif event == "return":
                    # 捕获 return 前最后时刻的局部变量变动
                    diff = {}
                    for k, v in frame.f_locals.items():
                        serialized_v = deep_serialize(v)
                        if k not in last_locals or last_locals[k] != serialized_v:
                            diff[k] = serialized_v
                    if diff:
                        snapshots.append({
                            "line": lineno,
                            "changes": diff,
                            "note": "Return 触发前"
                        })

                    return_val_holder["val"] = deep_serialize(arg)

                return trace_func

            return trace_func, input_args, snapshots, return_val_holder

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                trace_func, input_args, snapshots, return_val_holder = create_tracer()
                old_trace = sys.gettrace()
                sys.settrace(trace_func)
                try:
                    res = await func(*args, **kwargs)
                finally:
                    sys.settrace(old_trace)

                _export_markdown(
                    output_file,
                    func.__name__,
                    input_args,
                    snapshots,
                    return_val_holder["val"],
                )
                return res

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                trace_func, input_args, snapshots, return_val_holder = create_tracer()
                old_trace = sys.gettrace()
                sys.settrace(trace_func)
                try:
                    res = func(*args, **kwargs)
                finally:
                    sys.settrace(old_trace)

                _export_markdown(
                    output_file,
                    func.__name__,
                    input_args,
                    snapshots,
                    return_val_holder["val"],
                )
                return res

            return sync_wrapper

    return decorator


def _export_markdown(filepath, func_name, inputs, steps, return_val):
    """【修复点 3】修改渲染逻辑，专注于展示增量的变量变动"""
    md = [
        f"# 🔍 函数变量追踪报告: `{func_name}`",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## 📥 1. 入参 (Input Arguments)",
        "```json",
        json.dumps(inputs, indent=2, ensure_ascii=False),
        "```\n",
        "## 🔄 2. 变量演变过程 (增量变动)",
        "> 💡 **说明**：为避免信息冗余，此处**仅展示每一步中新增或发生修改的变量**。\n",
    ]

    for idx, step in enumerate(steps, 1):
        note = step.get("note", "产生新变量或发生修改")
        # Python 的 event='line' 是在执行这行代码前触发，因此此刻捕获的差异，实际上是上一行代码执行的结果
        md.append(f"### 📍 游标到达行号: {step['line']} ({note})")
        md.append("```json")
        md.append(json.dumps(step["changes"], indent=2, ensure_ascii=False))
        md.append("```\n")

    md.extend(
        [
            "## 📤 3. 函数返回值 (Return Value)",
            "```json",
            json.dumps(return_val, indent=2, ensure_ascii=False),
            "```\n",
        ]
    )

    Path(filepath).write_text("\n".join(md), encoding="utf-8")
