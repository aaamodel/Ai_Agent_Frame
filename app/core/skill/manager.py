### File: ./core/skills/manager.py
from typing import Dict, Any, List, TypedDict, Optional
import re
import yaml
from pathlib import PurePosixPath

from trace_to_markdown import trace_to_markdown
from app.core.backends.filesystem import FilesystemBackend

from loguru import logger

# 严格对齐约束
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024


class SkillMetadata(TypedDict):
    """规范级技能元数据定义"""
    name: str
    description: str
    file_path: str  # SKILL.md 的绝对/虚拟文件路径
    allowed_tools: List[str]
    license: Optional[str]
    compatibility: Optional[str]


class SkillsState:
    """系统当前的技能状态快照存储"""

    def __init__(self):
        self.available_skills: Dict[str, SkillMetadata] = {}


class SkillManager:
    """技能核心服务：联动物理/虚拟文件系统，完成动态扫描、元数据解析与 System Prompt 提炼"""

    def __init__(self, fs_backend: FilesystemBackend, skills_dir: str = "./skills"):
        self.fs_backend = fs_backend
        self.skills_dir = skills_dir
        self.state = SkillsState()

    def _validate_skill_name(self, name: str, directory_name: str) -> tuple[bool, str]:
        """校验技能名称是否符合 Agent Skills 规范约束"""
        if not name:
            return False, "name is required"
        if len(name) > MAX_SKILL_NAME_LENGTH:
            return False, f"name exceeds {MAX_SKILL_NAME_LENGTH} characters"
        if name.startswith("-") or name.endswith("-") or "--" in name:
            return False, "name must be lowercase alphanumeric with single hyphens only"

        for c in name:
            if c != "-" and not (c.isalpha() and c.islower()) and not c.isdigit():
                return False, "name must be lowercase alphanumeric with single hyphens only"

        if name != directory_name:
            return False, f"name '{name}' must match its parent directory name '{directory_name}'"
        return True, ""

    def _parse_skill_metadata(self, content: str, file_path: str, directory_name: str) -> Optional[SkillMetadata]:
        """解析并校验 SKILL.md 的 YAML Frontmatter 内容"""
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)
        if not match:
            logger.warning(f"Skipping {file_path}: no valid YAML frontmatter delimiters (---) found.")
            return None

        try:
            frontmatter_data = yaml.safe_load(match.group(1))
        except yaml.YAMLError as e:
            logger.warning(f"Invalid YAML syntax in {file_path}: {e}")
            return None

        if not isinstance(frontmatter_data, dict):
            return None

        name = str(frontmatter_data.get("name", "")).strip()
        description = str(frontmatter_data.get("description", "")).strip()
        if not name or not description:
            logger.warning(f"Skipping {file_path}: missing required fields 'name' or 'description'.")
            return None

        # 校验名称合规性
        is_valid, err_msg = self._validate_skill_name(name, directory_name)
        if not is_valid:
            logger.warning(f"Skill '{name}' in {file_path} standard verification failed: {err_msg}")
            return None

        # 描述截断保护
        if len(description) > MAX_SKILL_DESCRIPTION_LENGTH:
            description = description[:MAX_SKILL_DESCRIPTION_LENGTH]

        # 解析推荐工具集
        raw_tools = frontmatter_data.get("allowed-tools", "")
        allowed_tools = []
        if isinstance(raw_tools, str):
            allowed_tools = [t.strip(",").strip() for t in raw_tools.split() if t.strip()]

        return SkillMetadata(
            name=name,
            description=description,
            file_path=file_path,
            allowed_tools=allowed_tools,
            license=str(frontmatter_data.get("license", "")).strip() or None,
            compatibility=str(frontmatter_data.get("compatibility", "")).strip() or None
        )

    @trace_to_markdown(output_file="scan_and_refresh_skills")
    async def scan_and_refresh_skills(self) -> SkillsState:
        """【完全异步化】扫描指定技能根目录，完成全量冷启动/热刷新"""
        new_skills: Dict[str, SkillMetadata] = {}
        try:
            # 1. 列出技能根目录下的所有子文件夹
            ls_results = await self.fs_backend.ls(self.skills_dir)
            for entry in ls_results.entries:
                if not entry.get("is_dir"):
                    continue

                dir_path = entry["path"]
                dir_name = PurePosixPath(dir_path).name
                skill_md_path = str(PurePosixPath(dir_path) / "SKILL.md")

                # 2. 异步读取该技能夹下的 SKILL.md 文件
                try:
                    # 默认读取前 2000 行，足以覆盖任何标准 Frontmatter 元数据块
                    read_result = await self.fs_backend.read(skill_md_path, offset=0, limit=2000)
                    if not read_result or read_result.error:
                        continue  # 有错误直接跳过

                    file_data = read_result.file_data
                    if not file_data:
                        continue

                    content_str = file_data.get("content", "")
                    if not content_str:
                        logger.warning(f"文件为空: {skill_md_path}")
                        continue

                    # 传入提取后的纯字符串
                    metadata = self._parse_skill_metadata(content_str, skill_md_path, dir_name)



                    if metadata:
                        new_skills[metadata["name"]] = metadata
                except Exception as file_err:
                    # 某个技能缺失 SKILL.md 或读取失败不崩溃系统，仅降级报错
                    logger.exception(f"Path {skill_md_path} is not a valid skill folder: {file_err}")
                    continue

        except Exception as e:
            logger.error(f"Failed to scan skills directory {self.skills_dir}: {e}")

        self.state.available_skills = new_skills
        return self.state

    @trace_to_markdown(output_file="get_skills_summary_for_prompt")
    def get_skills_summary_for_prompt(self) -> str:
        """生成注入到 System Prompt 中的精简导流文本块（实现渐进式披露的基石）"""
        if not self.state.available_skills:
            return ""

        prompt_lines = [
            "## Available Advanced Skills",
            "You have access to the following long-form specialized behaviors.",
            # 👇 这里已经完全对齐为你的真实工具名 file_read_tool
            "NOTE: Only the name and brief abstract are shown below. If you decide a skill is highly relevant but you lack detailed instructions, execution parameters, or precise context rules to safely proceed, you MUST use the file-reading tool `file_read_tool` to read the respective 'Source File' before making calls.",
            ""
        ]

        for name, meta in self.state.available_skills.items():
            line = f"- **{name}**: {meta['description']} (Source File: `{meta['file_path']}`)"
            prompt_lines.append(line)

        return "\n".join(prompt_lines)

    # ------------------------------------------------------------------
    # 方案A：意图-技能命中（让 Skill 真正“号令”可用工具）
    # ------------------------------------------------------------------
    @staticmethod
    def _chinese_chars(text: str) -> str:
        """提取字符串中的中文字符，忽略英文、数字、标点。"""
        return "".join(ch for ch in (text or "") if "\u4e00" <= ch <= "\u9fff")

    def resolve_relevant_skill(self, query: str) -> Optional[SkillMetadata]:
        """基于轻量中文字符双重合匹配，返回与用户问题最相关的技能；无命中返回 None。

        设计动机：Pipeline 按意图类型粗暴注入工具集（例如“任务规划 → feishu”），
        完全无视 SKILL.md 声明的 allowed-tools。本方法在编排层按 query 命中的
        技能元数据（name / description / allowed-tools）做相关性打分，供上层用
        技能立场的工具白名单覆盖 Pipeline 注入。

        打分策略（保守防误杀）：
          - 只统计中文字符（含 unigram 与 bigram），排除英文/数字噪声；
          - 超用词/助词已被 _chinese_chars 之后的操作区分，不参与 bigram；
          - 命中得分 = 与技能文本共享的中文 bigram 数量；得分 >= 阈值(默认 1)
            视为命中，取得分最高者。

        Args:
            query: 用户原始问题。

        Returns:
            命中且得分达标的 SkillMetadata；否则 None。
        """
        query_chinese: str = self._chinese_chars(query or "").strip().lower()
        if not query_chinese or not self.state.available_skills:
            return None

        def bigrams(text: str) -> set:
            chars: str = self._chinese_chars(text).lower()
            return {chars[i:i + 2] for i in range(len(chars) - 1)}

        query_ngrams: set = bigrams(query_chinese)

        best_name: Optional[str] = None
        best_score: int = 0
        for name, meta in self.state.available_skills.items():
            candidate_text: str = (
                f"{name} {meta.get('description', '')} {' '.join(meta.get('allowed_tools') or [])}"
            )
            score: int = len(query_ngrams & bigrams(candidate_text))
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None or best_score < 1:
            return None
        return self.state.available_skills.get(best_name)