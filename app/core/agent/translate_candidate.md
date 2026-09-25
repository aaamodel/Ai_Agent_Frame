# 🔍 函数变量追踪报告: `translate_candidate`
**生成时间**: 2026-09-25 22:03:52

## 📥 1. 入参 (Input Arguments)
```json
{
  "candidate_id": "tool:alt_tool",
  "candidates": [
    {
      "__class__": "Candidate",
      "id": "tool:alt_tool",
      "description": "尚未尝试的工具"
    },
    {
      "__class__": "Candidate",
      "id": "tool:t2_tool",
      "description": "尚未尝试的工具"
    }
  ],
  "allowed_tools": [
    "t1_tool",
    "t2_tool",
    "alt_tool"
  ],
  "facts": []
}
```

## 🔄 2. 变量演变过程 (增量变动)
> 💡 **说明**：为避免信息冗余，此处**仅展示每一步中新增或发生修改的变量**。

### 📍 游标到达行号: 459 (产生新变量或发生修改)
```json
{
  "parsed": "tool:alt_tool"
}
```

### 📍 游标到达行号: 463 (产生新变量或发生修改)
```json
{
  "valid_ids": [
    "tool:t2_tool",
    "tool:alt_tool"
  ]
}
```

### 📍 游标到达行号: 467 (产生新变量或发生修改)
```json
{
  "kind": "tool",
  "_": ":",
  "name": "alt_tool"
}
```

### 📍 游标到达行号: 469 (产生新变量或发生修改)
```json
{
  "allowed": [
    "alt_tool",
    "t2_tool",
    "t1_tool"
  ]
}
```

## 📤 3. 函数返回值 (Return Value)
```json
{
  "tool_name": "alt_tool",
  "action_input": {},
  "title": "调用 alt_tool"
}
```
