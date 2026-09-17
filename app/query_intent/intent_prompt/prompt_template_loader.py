import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.query_intent.intent_prompt.agent_intent_prompt import PromptTemplateUtils


logger = logging.getLogger(__name__)


def _is_blank(s: Optional[str]) -> bool:
    return s is None or s.strip() == ""


@dataclass
class PromptTemplateLoader:
    resource_loader: object = None
    cache: Dict[str, str] = field(default_factory=dict)
    section_cache: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def load(self, path: str) -> str:
        if _is_blank(path):
            raise ValueError("提示模板路径为空")
        if path not in self.cache:
            self.cache[path] = self._read_resource(path)
        return self.cache[path]

    def render(self, path: str, slots: Dict[str, str]) -> str:
        template = self.load(path)
        filled = PromptTemplateUtils.fill_slots(template, slots)
        return PromptTemplateUtils.cleanup_prompt(filled)

    def load_section(self, path: str, section: str) -> str:
        if path not in self.section_cache:
            content = self.load(path)
            self.section_cache[path] = PromptTemplateUtils.parse_sections(content)
        sections = self.section_cache[path]
        template = sections.get(section)
        if template is None:
            raise ValueError("模板 section 不存在：{} -> {}".format(path, section))
        return template

    def render_section(self, path: str, section: str, slots: Dict[str, str]) -> str:
        template = self.load_section(path, section)
        filled = PromptTemplateUtils.fill_slots(template, slots)
        return PromptTemplateUtils.cleanup_prompt(filled)

    def _read_resource(self, path: str) -> str:
        location = path if path.startswith("classpath:") else "classpath:" + path
        if self.resource_loader is not None:
            resource = self.resource_loader.get_resource(location)
            if not resource.exists():
                raise ValueError("提示词模板路径不存在：" + path)
            try:
                with resource.get_input_stream() as in_stream:
                    return in_stream.read().decode("utf-8")
            except Exception as e:
                logger.error("读取提示模板失败，路径：{}".format(path), e)
                raise ValueError("读取提示模板失败，路径：{}".format(path), e)
        else:
            file_path = location.replace("classpath:", "")
            if not os.path.exists(file_path):
                raise ValueError("提示词模板路径不存在：" + path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error("读取提示模板失败，路径：{}".format(path), e)
                raise ValueError("读取提示模板失败，路径：{}".format(path), e)
