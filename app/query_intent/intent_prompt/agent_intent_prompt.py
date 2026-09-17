"""
本文件聚合了 core/prompt 目录下 3 个原有模块缺失引用的 7 个类：
  1. OrchestrationMode  —— 编排模式枚举 (原 config)
  2. AgentProfileDO     —— 智能体配置实体 (原 dao.entity)
  3. AgentPromptDO      —— 智能体提示词实体 (原 dao.entity)
  4. AgentProfileMapper —— 智能体配置 Mapper 接口 (原 dao.mapper)
  5. AgentPromptMapper  —— 智能体提示词 Mapper 接口 (原 dao.mapper)
  6. AgentPromptCacheManager —— 智能体提示词缓存管理器 (原 core.prompt)
  7. PromptTemplateUtils     —— 提示模板工具类 (原 core.prompt)

"""

from __future__ import annotations

import logging
import re
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from app.query_intent.intent_mapper import BaseMapper


logger = logging.getLogger(__name__)

# =====================================================================
# 1. OrchestrationMode  编排模式枚举 (对应 com.nageoffer.ai.ragent.rag.config.OrchestrationMode)
# =====================================================================
class OrchestrationMode(Enum):
    """执行架构档位，由 ragent.engine.type 指定（部署级决策，切换需重启）"""

    WORKFLOW = "WORKFLOW"   # v1 编排管线：意图分类 → 检索 → 合成，链路确定、延迟低
    AGENT    = "AGENT"      # v2 ReAct 架构：主 Agent 决策，RAG 管线降级为其中一个 Tool

    @staticmethod
    def of(value: Optional[str]) -> "OrchestrationMode":
        """解析配置值，大小写不敏感；空白或无法识别时回落 WORKFLOW"""
        if value is None or str(value).strip() == "":
            return OrchestrationMode.WORKFLOW
        normalized = str(value).strip().upper()
        for mode in OrchestrationMode:
            if mode.value == normalized:
                return mode
        return OrchestrationMode.WORKFLOW

    def __str__(self) -> str:  # noqa: D401
        return self.value


# =====================================================================
# 2. AgentProfileDO  智能体配置实体 (对应 t_agent_profile 表)
# =====================================================================
@dataclass
class AgentProfileDO:
    id: Optional[str] = None
    name: Optional[str] = None                       # 展示名称，全局唯一
    description: Optional[str] = None
    avatar: Optional[str] = None                     # 头像预设标识
    builtin: Optional[int] = None                    # 是否内置：内置智能体不可编辑/删除，是回落终点
    active: Optional[int] = None                     # 是否激活，全局仅允许一条为 1
    create_by: Optional[str] = None
    update_by: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    deleted: Optional[int] = None                    # 逻辑删除

    # ---------- 与原 Java getter 语义对齐的便捷方法 ----------
    def get_id(self) -> Optional[str]:
        return self.id

    def get_name(self) -> Optional[str]:
        return self.name

    def get_builtin(self) -> Optional[int]:
        return self.builtin

    def get_active(self) -> Optional[int]:
        return self.active

    def get_create_time(self) -> Optional[datetime]:
        return self.create_time


# =====================================================================
# 3. AgentPromptDO  智能体提示词实体 (对应 t_agent_prompt 表)
# =====================================================================
@dataclass
class AgentPromptDO:
    id: Optional[str] = None
    agent_id: Optional[str] = None                    # 归属智能体
    slot_key: Optional[str] = None                    # 槽位标识，取值对应 AgentPromptSlot.name
    content: Optional[str] = None                     # 提示词全文（空白视为未配置并回落内置智能体）
    create_by: Optional[str] = None
    update_by: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    deleted: Optional[int] = None                     # 逻辑删除

    # ---------- 与原 Java getter 语义对齐的便捷方法 ----------
    def get_agent_id(self) -> Optional[str]:
        return self.agent_id

    def get_slot_key(self) -> Optional[str]:
        return self.slot_key

    def get_content(self) -> Optional[str]:
        return self.content


# =====================================================================
# 4. AgentProfileMapper  智能体配置 Mapper 接口
# =====================================================================
class AgentProfileMapper(BaseMapper[AgentProfileDO], ABC):
    """继承 BaseMapper[T] 即可获得标准 CRUD：
       insert / delete_by_id / update_by_id / select_by_id / select_list 等
    """
    pass


# =====================================================================
# 5. AgentPromptMapper  智能体提示词 Mapper 接口
# =====================================================================
class AgentPromptMapper(BaseMapper[AgentPromptDO], ABC):
    """继承 BaseMapper[T] 即可获得标准 CRUD"""
    pass


# =====================================================================
# 6. AgentPromptCacheManager  智能体提示词缓存管理器
# =====================================================================
@dataclass
class AgentPromptCacheManager:
    """
    缓存"激活智能体叠加自定义提示词之后"的结果（slot_key → content 映射）。
    v2 版本号随槽位集合变化递增，避免旧缓存缺少新增槽位。
    """

    # 注入依赖；缺失时自动退化为本地内存缓存，方便测试/离线使用
    string_redis_template: Optional[Any] = None
    object_mapper: Optional[Any] = None

    CACHE_KEY: str = "ragent:agent:resolved-prompts:v2"
    CACHE_EXPIRE_HOURS: int = 1

    # ---------- 本地兜底缓存（当 string_redis_template 为 None 时启用） ----------
    _local_cache: Dict[str, str] = field(default_factory=dict, repr=False)
    _local_expire_at: Optional[float] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def get_from_cache(self) -> Optional[Dict[str, str]]:
        """读取缓存，不存在则返回 None；异常被记录后返回 None"""
        try:
            if self.string_redis_template is None:
                import time as _time
                if (self._local_expire_at is None) or (_time.time() > self._local_expire_at):
                    self._local_cache.clear()
                    return None
                return dict(self._local_cache) if self._local_cache else None

            cache_json = self.string_redis_template.opsForValue().get(self.CACHE_KEY)
            if cache_json is None:
                return None
            if self.object_mapper is not None:
                return self.object_mapper.readValue(cache_json, dict)
            import json as _json
            return _json.loads(cache_json)
        except Exception as e:
            logger.error("从 Redis 读取智能体提示词缓存失败", exc_info=e)
            return None

    # ------------------------------------------------------------------
    def save_to_cache(self, prompts: Dict[str, str]) -> None:
        """写入缓存；异常只记录不抛出"""
        try:
            if self.string_redis_template is None:
                import time as _time
                self._local_cache = dict(prompts) if prompts else {}
                self._local_expire_at = _time.time() + self.CACHE_EXPIRE_HOURS * 3600
                return

            if self.object_mapper is not None:
                cache_json = self.object_mapper.writeValueAsString(prompts)
            else:
                import json as _json
                cache_json = _json.dumps(prompts, ensure_ascii=False)
            import time as _time
            self.string_redis_template.opsForValue().set(
                self.CACHE_KEY, cache_json,
                self.CACHE_EXPIRE_HOURS * 3600, _time_unit_hours_sentinel()
            )
        except Exception as e:
            logger.error("保存智能体提示词到 Redis 缓存失败", exc_info=e)

    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        """任何智能体/槽位写操作后必须调用，否则改动直到过期才生效"""
        try:
            if self.string_redis_template is None:
                self._local_cache.clear()
                self._local_expire_at = None
                logger.info("智能体提示词本地缓存已清除")
                return
            self.string_redis_template.delete(self.CACHE_KEY)
            logger.info("智能体提示词缓存已清除")
        except Exception as e:
            logger.error("清除智能体提示词缓存失败", exc_info=e)


def _time_unit_hours_sentinel() -> str:
    """小占位，方便底层 Redis 客户端识别时间单位为小时；若接口不接收字符串参数可忽略"""
    return "HOURS"


# =====================================================================
# 7. PromptTemplateUtils  提示模板工具类
# =====================================================================
class PromptTemplateUtils:
    """`final` 工具类：模板清洗、占位符填充、section 解析"""

    _MULTI_BLANK_LINES = re.compile(r"\n{3,}")
    _SECTION_HEADER = re.compile(
        r"^---\s*section:\s*(\S+)\s*---$",
        re.MULTILINE,
    )

    @staticmethod
    def cleanup_prompt(prompt: Optional[str]) -> str:
        """折叠 3+ 连续空行为 2，并去掉首尾空白"""
        if prompt is None:
            return ""
        return PromptTemplateUtils._MULTI_BLANK_LINES.sub("\n\n", prompt).strip()

    @staticmethod
    def fill_slots(template: Optional[str], slots: Optional[Dict[str, str]]) -> str:
        """按 `{key}` 占位符替换；缺失值用空串补齐；未命中的占位符保留原样"""
        if template is None:
            return ""
        if slots is None or len(slots) == 0:
            return template
        result = template
        for key, raw_value in slots.items():
            value = "" if raw_value is None else str(raw_value)
            result = result.replace("{" + key + "}", value)
        return result

    @staticmethod
    def parse_sections(content: Optional[str]) -> Dict[str, str]:
        """
        将包含 `--- section: <name> ---` 分隔符的模板内容解析为 name → content 的有序映射。
        分隔符必须出现在行首；section 内容的首尾空行会被清理，但内部结构保留。
        """
        sections: "OrderedDict[str, str]" = OrderedDict()
        if content is None or content.strip() == "":
            return sections
        matcher = PromptTemplateUtils._SECTION_HEADER.finditer(content)
        last_start = -1
        last_name: Optional[str] = None
        for match in matcher:
            if last_name is not None:
                sections[last_name] = PromptTemplateUtils._trim_section(content[last_start:match.start()])
            last_name = match.group(1)
            last_start = match.end()
        if last_name is not None:
            sections[last_name] = PromptTemplateUtils._trim_section(content[last_start:])
        return sections

    @staticmethod
    def _trim_section(section: str) -> str:
        """去掉 section 开头一个换行 + 结尾空白"""
        if section.startswith("\n"):
            section = section[1:]
        return section.rstrip()


__all__ = [
    "OrchestrationMode",
    "AgentProfileDO",
    "AgentPromptDO",
    "AgentProfileMapper",
    "AgentPromptMapper",
    "AgentPromptCacheManager",
    "PromptTemplateUtils",
]
