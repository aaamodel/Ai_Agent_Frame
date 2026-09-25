# 🔍 函数变量追踪报告: `build_candidates`
**生成时间**: 2026-09-25 22:03:52

## 📥 1. 入参 (Input Arguments)
```json
{
  "state": {
    "run_id": "s-evidence-restore-off:6962bb44c785451d8a83f0498bab5884",
    "session_id": "s-evidence-restore-off",
    "trace_id": "trace-test",
    "user_input": "查退款政策",
    "intent": {
      "intent": "general",
      "confidence": 1.0,
      "slots": {},
      "preferred_mode": null,
      "allowed_tools": null
    },
    "should_plan": true,
    "mode_source": "strategy",
    "memory_context": {
      "short_term": [],
      "long_term": [],
      "short term memory": [],
      "long term memory": []
    },
    "skills_prompt": "",
    "skills_index": "",
    "extracted_facts": [],
    "step_corrections": [],
    "active_tool_names": [
      "rag_knowledge_search"
    ],
    "tool_schemas": {
      "rag_knowledge_search": "功能描述: rag_knowledge_search 测试工具\n输入JSON格式规约Schema: {\"type\": \"object\", \"properties\": {\"q\": {\"type\": \"string\"}}}\n【此工具专属调用守则】：\n"
    },
    "fc_tool_definitions": [
      {
        "type": "function",
        "function": {
          "name": "rag_knowledge_search",
          "description": "rag_knowledge_search 测试工具",
          "parameters": {
            "type": "object",
            "properties": {
              "q": {
                "type": "string"
              }
            }
          }
        }
      }
    ],
    "extra_system_hints": [],
    "budget": {
      "per_tool_limits": {
        "rag_knowledge_search": 3
      },
      "default_per_tool": 10,
      "total_budget": 20,
      "invalid_limit": 3,
      "invalid_consecutive_limit": 2,
      "relevance_check_call": 5,
      "used": {},
      "invalid": {},
      "consecutive_invalid": {},
      "locked": {},
      "total_used": 0
    },
    "plan": [
      {
        "id": "t1",
        "title": "取数1",
        "description": "调用 rag_knowledge_search 取数",
        "action_type": "tool",
        "tool_name": "rag_knowledge_search",
        "tool_args_hint": "{\"q\": \"x\"}",
        "covers_sub_questions": null
      }
    ],
    "cursor": 0,
    "subtask_results": [],
    "skipped_task_ids": [],
    "early_finish": false,
    "react_messages": [],
    "react_history_lines": [],
    "react_protocol": "",
    "react_step": 0,
    "react_empty_turns": 0,
    "evidence_units": [],
    "evidence_meta": {
      "next_seq": 1,
      "rounds": []
    },
    "steps": [],
    "retry_counts": {},
    "replan_attempts": 0,
    "max_replan": 1,
    "max_steps": 5,
    "last_error": null,
    "empty_data_signal": null,
    "insufficiency_signal": null,
    "insufficiency_kind": null,
    "draft_answer": null,
    "pending_tool": null,
    "reflect_failed": false,
    "final_answer": "",
    "success": false,
    "degraded": false,
    "mode_used": "plan_execute",
    "route": ""
  },
  "current_result": {
    "subtask_id": "t1",
    "title": "取数1",
    "action_type": "tool",
    "plan_attempt": 0,
    "tool_name": "rag_knowledge_search",
    "kind": "plan_tool_call",
    "action": "rag_knowledge_search",
    "action_input": {
      "q": "x"
    },
    "observation": "rag_knowledge_search <- {\"q\": \"x\"} => --- 知识库检索结果 (查询: 退款政策) ---\n[1] 来源文献: policy_1.txt\n内容片段: 退款审批须上传签收凭证与发票照片，财务在材料齐全后三个工作日内完成审核，审核通过的款项原路退回付款账户，遇法定节假日顺延，跨月提交的单据并入下一结算周期统一处理。\n[2] 来源文献: policy_2.txt\n内容片段: 退货商品入库验收由仓储岗负责，外包装破损或附件缺失的包裹需现场拍照登记，验收不通过的退货单退回客服跟进，客户补充材料后重新发起流程，验收通过才释放退款额度。\n[3] 来源文献: policy_3.txt\n内容片段: 运费险理赔在退款完成后自动触发，理赔金额按收货与退货两段实际运费计算，三个工作日内发放至客户下单时使用的支付账户，客户可在订单详情页查看理赔进度与到账记录。\n[4] 来源文献: policy_4.txt\n内容片段: 大额退款（单笔超过一千元）须财务主管二次复核，复核内容包括订单真实性与发票状态，每月五日与二十日为大额退款集中打款日，紧急情形可申请单独走款但需分管总监邮件审批。\n[5] 来源文献: policy_5.txt\n内容片段: 优惠券与积分抵扣部分按原渠道分别退回：平台券退回卡券包且有效期不延长，积分退回会员账户并恢复成长值，第三方支付的差额部分按原路退回，组合支付订单逐笔算清。\n[6] 来源文献: policy_6.txt\n内容片段: 跨境订单退款涉及汇率波动，按下单时锁定的结算汇率折算外币，关税与清关服务费不在退款范围内，银行端国际汇款一般需要五到七个工作日，到账短信可能延迟。\n[7] 来源文献: policy_7.txt\n内容片段: 质量问题导致的退货运费由商家承担，客户先行垫付后凭快递底单报销，七天无理由退货的往返运费由客户自行承担，拒收包裹产生的退回运费同样从退款金额中扣减。\n[8] 来源文献: policy_8.txt\n内容片段: 退款纠纷统一由售后专员建单跟进，协商记录全程留痕，超过十五天未达成一致的工单升级至平台介入，平台依据聊天记录与物流凭证在七个工作日内作出裁决。\n"
  }
}
```

## 🔄 2. 变量演变过程 (增量变动)
> 💡 **说明**：为避免信息冗余，此处**仅展示每一步中新增或发生修改的变量**。

### 📍 游标到达行号: 306 (产生新变量或发生修改)
```json
{
  "name": "rag_knowledge_search"
}
```

### 📍 游标到达行号: 308 (产生新变量或发生修改)
```json
{
  "allowed": [
    "rag_knowledge_search"
  ]
}
```

### 📍 游标到达行号: 309 (产生新变量或发生修改)
```json
{
  "results": []
}
```

### 📍 游标到达行号: 312 (产生新变量或发生修改)
```json
{
  "results": [
    {
      "subtask_id": "t1",
      "title": "取数1",
      "action_type": "tool",
      "plan_attempt": 0,
      "tool_name": "rag_knowledge_search",
      "kind": "plan_tool_call",
      "action": "rag_knowledge_search",
      "action_input": {
        "q": "x"
      },
      "observation": "rag_knowledge_search <- {\"q\": \"x\"} => --- 知识库检索结果 (查询: 退款政策) ---\n[1] 来源文献: policy_1.txt\n内容片段: 退款审批须上传签收凭证与发票照片，财务在材料齐全后三个工作日内完成审核，审核通过的款项原路退回付款账户，遇法定节假日顺延，跨月提交的单据并入下一结算周期统一处理。\n[2] 来源文献: policy_2.txt\n内容片段: 退货商品入库验收由仓储岗负责，外包装破损或附件缺失的包裹需现场拍照登记，验收不通过的退货单退回客服跟进，客户补充材料后重新发起流程，验收通过才释放退款额度。\n[3] 来源文献: policy_3.txt\n内容片段: 运费险理赔在退款完成后自动触发，理赔金额按收货与退货两段实际运费计算，三个工作日内发放至客户下单时使用的支付账户，客户可在订单详情页查看理赔进度与到账记录。\n[4] 来源文献: policy_4.txt\n内容片段: 大额退款（单笔超过一千元）须财务主管二次复核，复核内容包括订单真实性与发票状态，每月五日与二十日为大额退款集中打款日，紧急情形可申请单独走款但需分管总监邮件审批。\n[5] 来源文献: policy_5.txt\n内容片段: 优惠券与积分抵扣部分按原渠道分别退回：平台券退回卡券包且有效期不延长，积分退回会员账户并恢复成长值，第三方支付的差额部分按原路退回，组合支付订单逐笔算清。\n[6] 来源文献: policy_6.txt\n内容片段: 跨境订单退款涉及汇率波动，按下单时锁定的结算汇率折算外币，关税与清关服务费不在退款范围内，银行端国际汇款一般需要五到七个工作日，到账短信可能延迟。\n[7] 来源文献: policy_7.txt\n内容片段: 质量问题导致的退货运费由商家承担，客户先行垫付后凭快递底单报销，七天无理由退货的往返运费由客户自行承担，拒收包裹产生的退回运费同样从退款金额中扣减。\n[8] 来源文献: policy_8.txt\n内容片段: 退款纠纷统一由售后专员建单跟进，协商记录全程留痕，超过十五天未达成一致的工单升级至平台介入，平台依据聊天记录与物流凭证在七个工作日内作出裁决。\n"
    }
  ]
}
```

### 📍 游标到达行号: 315 (产生新变量或发生修改)
```json
{
  "facts": []
}
```

### 📍 游标到达行号: 316 (产生新变量或发生修改)
```json
{
  "candidates": []
}
```

### 📍 游标到达行号: 319 (产生新变量或发生修改)
```json
{
  "excluded": []
}
```

### 📍 游标到达行号: 320 (产生新变量或发生修改)
```json
{
  "attempted": []
}
```

### 📍 游标到达行号: 344 (产生新变量或发生修改)
```json
{
  "record": {
    "subtask_id": "t1",
    "title": "取数1",
    "action_type": "tool",
    "plan_attempt": 0,
    "tool_name": "rag_knowledge_search",
    "kind": "plan_tool_call",
    "action": "rag_knowledge_search",
    "action_input": {
      "q": "x"
    },
    "observation": "rag_knowledge_search <- {\"q\": \"x\"} => --- 知识库检索结果 (查询: 退款政策) ---\n[1] 来源文献: policy_1.txt\n内容片段: 退款审批须上传签收凭证与发票照片，财务在材料齐全后三个工作日内完成审核，审核通过的款项原路退回付款账户，遇法定节假日顺延，跨月提交的单据并入下一结算周期统一处理。\n[2] 来源文献: policy_2.txt\n内容片段: 退货商品入库验收由仓储岗负责，外包装破损或附件缺失的包裹需现场拍照登记，验收不通过的退货单退回客服跟进，客户补充材料后重新发起流程，验收通过才释放退款额度。\n[3] 来源文献: policy_3.txt\n内容片段: 运费险理赔在退款完成后自动触发，理赔金额按收货与退货两段实际运费计算，三个工作日内发放至客户下单时使用的支付账户，客户可在订单详情页查看理赔进度与到账记录。\n[4] 来源文献: policy_4.txt\n内容片段: 大额退款（单笔超过一千元）须财务主管二次复核，复核内容包括订单真实性与发票状态，每月五日与二十日为大额退款集中打款日，紧急情形可申请单独走款但需分管总监邮件审批。\n[5] 来源文献: policy_5.txt\n内容片段: 优惠券与积分抵扣部分按原渠道分别退回：平台券退回卡券包且有效期不延长，积分退回会员账户并恢复成长值，第三方支付的差额部分按原路退回，组合支付订单逐笔算清。\n[6] 来源文献: policy_6.txt\n内容片段: 跨境订单退款涉及汇率波动，按下单时锁定的结算汇率折算外币，关税与清关服务费不在退款范围内，银行端国际汇款一般需要五到七个工作日，到账短信可能延迟。\n[7] 来源文献: policy_7.txt\n内容片段: 质量问题导致的退货运费由商家承担，客户先行垫付后凭快递底单报销，七天无理由退货的往返运费由客户自行承担，拒收包裹产生的退回运费同样从退款金额中扣减。\n[8] 来源文献: policy_8.txt\n内容片段: 退款纠纷统一由售后专员建单跟进，协商记录全程留痕，超过十五天未达成一致的工单升级至平台介入，平台依据聊天记录与物流凭证在七个工作日内作出裁决。\n"
  }
}
```

### 📍 游标到达行号: 346 (产生新变量或发生修改)
```json
{
  "used_tools": [
    "rag_knowledge_search"
  ]
}
```

## 📤 3. 函数返回值 (Return Value)
```json
{
  "__class__": "CandidateSet",
  "candidates": [],
  "excluded": []
}
```
