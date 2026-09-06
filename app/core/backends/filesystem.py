import errno
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Literal, Final
from dataclasses import dataclass, field
from typing_extensions import TypedDict, NotRequired

from loguru import logger

# ==========================================
# 1. 基础类型定义与全局常量
# ==========================================
DEFAULT_GREP_TIMEOUT: Final = 30

FileOperationError = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
]

FILE_NOT_FOUND: Final = "file_not_found"
PERMISSION_DENIED: Final = "permission_denied"
IS_DIRECTORY: Final = "is_directory"
INVALID_PATH: Final = "invalid_path"


class FileInfo(TypedDict):
    path: str
    is_dir: NotRequired[bool]
    size: NotRequired[int]
    modified_at: NotRequired[str]


class GrepMatch(TypedDict):
    path: str
    line: int
    text: str


class FileData(TypedDict):
    content: str
    encoding: str
    created_at: NotRequired[str]
    modified_at: NotRequired[str]


# ==========================================
# 2. 标准化返回值包装类 (DataClasses)
# ==========================================
@dataclass
class ReadResult:
    error: Optional[str] = None
    file_data: Optional[FileData] = None


@dataclass
class WriteResult:
    error: Optional[str] = None
    path: Optional[str] = None
    files_update: Optional[Dict[str, Any]] = None


@dataclass
class EditResult:
    error: Optional[str] = None
    path: Optional[str] = None
    files_update: Optional[Dict[str, Any]] = None
    occurrences: Optional[int] = None


@dataclass
class LsResult:
    error: Optional[str] = None
    # 🔥 关键修复：默认 entries 为空列表，避免 None 导致迭代崩溃
    entries: List[FileInfo] = field(default_factory=list)


@dataclass
class GrepResult:
    error: Optional[str] = None
    matches: Optional[List[GrepMatch]] = None


# ==========================================
# 3. 底层系统辅助函数
# ==========================================
@lru_cache(maxsize=1)
def _resolve_ripgrep_path() -> Optional[str]:
    """定位系统环境中的 ripgrep 可执行文件"""
    return shutil.which("rg")


def _is_eloop_oserror(exc: OSError) -> bool:
    """兼容多平台的循环软链接异常识别"""
    if exc.errno == errno.ELOOP:
        return True
    if sys.platform == "win32":
        return getattr(exc, "winerror", None) == 1921
    return False


def _raise_if_symlink_loop(path: Path) -> None:
    """修复 Python 3.13+ 对循环软链接不抛出异常的破坏性变更"""
    if not path.is_symlink():
        return
    try:
        path.stat()
    except OSError as exc:
        if _is_eloop_oserror(exc):
            raise


# ==========================================
# 4. 核心功能实现类
# ==========================================
class FilesystemBackend:
    """高安全、全功能、集成了 ripgrep 且接口标准化的本地文件系统后端"""

    def __init__(
            self,
            root_dir: Optional[str | Path] = None,
            virtual_mode: bool = True,
            max_file_size_mb: int = 10,
    ) -> None:
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _resolve_path(self, key: str) -> Path:
        """路径边界安全审查与安全沙箱防护"""
        if self.virtual_mode:
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                raise ValueError(INVALID_PATH)

            full = (self.cwd / vpath.lstrip("/")).resolve()
            try:
                full.relative_to(self.cwd)
            except ValueError:
                raise ValueError(INVALID_PATH)

            _raise_if_symlink_loop(full)


            return full

        path = Path(key)
        full = path if path.is_absolute() else (self.cwd / path).resolve()
        _raise_if_symlink_loop(full)

        # ========== 🎯 调试日志黄金修正版 ==========
        logger.warning(f"[DEBUG] 物理模式 - skill的全路径是：{full}")

        # ==========================================

        return full

    def _to_virtual_path(self, path: Path) -> str:
        """将底层物理物理路径反向映射成 Agent 视角下的虚拟路径"""
        return "/" + path.resolve().relative_to(self.cwd).as_posix()

    async def ls(self, path: Optional[str] = None, **kwargs: Any) -> LsResult:
        """非递归获取目录快照（增强了对大模型传参的兼容性，且保证 entries 永不为 None）"""

        # 🎯 兜底逻辑：如果大模型没传 path 而是传了 directory，自动纠正
        if not path:
            path = kwargs.get("directory") or kwargs.get("dir")
        if not path:
            # 参数缺失时返回错误，但 entries 保持空列表（已由 dataclass 默认值保证）
            return LsResult(error="Missing required parameter 'path' or 'directory'")

        try:
            dir_path = self._resolve_path(path)

            if not dir_path.exists():
                return LsResult(error=FILE_NOT_FOUND)
            if not dir_path.is_dir():
                return LsResult(error=IS_DIRECTORY)
        except ValueError:
            return LsResult(error=INVALID_PATH)
        except OSError as e:
            return LsResult(error=str(e))

        entries: List[FileInfo] = []
        try:
            for child_path in dir_path.iterdir():
                try:
                    is_file = child_path.is_file()
                    is_dir = child_path.is_dir()
                except OSError:
                    continue

                if not is_file and not is_dir:
                    continue

                display_p = str(child_path) if not self.virtual_mode else self._to_virtual_path(child_path)
                info: FileInfo = {"path": display_p, "is_dir": is_dir}
                try:
                    st = child_path.stat()
                    info["size"] = 0 if is_dir else int(st.st_size)
                    info["modified_at"] = datetime.fromtimestamp(st.st_mtime).isoformat()
                except OSError:
                    pass
                entries.append(info)
        except OSError as e:
            # 扫描过程中出错，返回部分结果并附带错误信息
            return LsResult(entries=entries, error=f"Partial scan error: {e}")

        entries.sort(key=lambda x: x["path"])
        return LsResult(entries=entries)


    async def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """按指定行跨度范围读取文件"""
        print("开始执行filesystem.read方法",flush=True)
        try:
            resolved_path = self._resolve_path(file_path)
            # 🎯【诊断】打印 file_read_tool 收到的原始 key 与解析后的物理路径，定位畸形路径来源
            if not resolved_path.exists():
                logger.warning(
                    "[路径诊断] file_read 未命中：原始key={!r} | cwd(根)={} | 解析后={}",
                    file_path,
                    self.cwd,
                    resolved_path,
                )
            if not resolved_path.exists() or resolved_path.is_dir():
                if resolved_path.is_dir():
                    return ReadResult(error=IS_DIRECTORY)
                return ReadResult(error=FILE_NOT_FOUND)

            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            st = resolved_path.stat()
            lines = content.splitlines(keepends=True)
            sliced_content = "".join(lines[offset:offset + limit]) if offset < len(lines) else ""
            file_data: FileData = {
                "content": sliced_content,
                "encoding": "utf-8",
                "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat()
            }
            return ReadResult(file_data=file_data)

        except PermissionError:
            print("权限错误", flush=True)
            return ReadResult(error=PERMISSION_DENIED)

        except (OSError, UnicodeDecodeError, ValueError) as e:
            print("其他读取文件错误", flush=True)
            return ReadResult(error=str(e))

    async def write(self, file_path: str, content: str) -> WriteResult:
        """安全的覆盖式或增量式文件写入文件"""
        try:
            resolved_path = self._resolve_path(file_path)
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW

            fd = os.open(resolved_path, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(content)

            display_path = str(resolved_path) if not self.virtual_mode else self._to_virtual_path(resolved_path)
            return WriteResult(path=display_path)
        except PermissionError:
            return WriteResult(error=PERMISSION_DENIED)
        except (OSError, UnicodeEncodeError, ValueError) as e:
            return WriteResult(error=str(e))

    async def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """通过旧代码块精确锚定并进行差分代码编辑"""
        try:
            resolved_path = self._resolve_path(file_path)
            if not resolved_path.exists():
                return EditResult(error=FILE_NOT_FOUND)

            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            # 跨平台换行符洗白规范化
            old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
            new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")

            # 兼容外部框架可能注入的工具依赖
            try:
                from deepagents.backends.utils import perform_string_replacement
                result = perform_string_replacement(content, old_string, new_string, replace_all)
                if isinstance(result, str):
                    return EditResult(error=result)
                new_content, count = result
            except ImportError:
                # Fallback: 原生 Python 字符串简单替换逻辑
                count = content.count(old_string)
                if count == 0:
                    return EditResult(error="Old string chunk not found in target file.")
                if count > 1 and not replace_all:
                    return EditResult(error="Multiple occurrences found; edit is ambiguous. Set replace_all=True.")
                new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string,
                                                                                                          new_string, 1)

            flags = os.O_WRONLY | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd_w = os.open(resolved_path, flags)
            with os.fdopen(fd_w, "w", encoding="utf-8", newline="") as f:
                f.write(new_content)

            display_path = str(resolved_path) if not self.virtual_mode else self._to_virtual_path(resolved_path)
            return EditResult(path=display_path, occurrences=count)
        except PermissionError:
            return EditResult(error=PERMISSION_DENIED)
        except (OSError, UnicodeError, ValueError) as e:
            return EditResult(error=str(e))

    async def grep(self, pattern: str, path: Optional[str] = None, glob: Optional[str] = None) -> GrepResult:
        """基于 ripgrep/Python 双引擎的高性能全文本搜索工具"""
        try:
            base_full = self._resolve_path(path or ".")
            if not base_full.exists():
                return GrepResult(error=FILE_NOT_FOUND)
        except (ValueError, OSError, RuntimeError) as e:
            return GrepResult(error=str(e))

        # 1. 尝试使用高吞吐量的 ripgrep 引擎进行秒级分析
        results = self._ripgrep_search(pattern, base_full, glob)

        # 2. ripgrep 出错、超时或缺失时，启动原生流式方案兜底
        if results is None:
            logger.info("Falling back to absolute pure Python grep implementation.")
            results, err = await self._python_search(pattern, base_full, glob)
            if err:
                return GrepResult(error=err)

        matches: List[GrepMatch] = []
        for fpath, items in results.items():
            for line_num, line_text in items:
                matches.append({
                    "path": fpath,
                    "line": int(line_num),
                    "text": line_text
                })
        return GrepResult(matches=matches)

    def _ripgrep_search(self, pattern: str, base_full: Path, include_glob: Optional[str]) -> Optional[
        Dict[str, List[Tuple[int, str]]]]:
        """基于底层 `rg --json -F` 的跨进程高并发高速查询实现"""
        rg_path = _resolve_ripgrep_path()
        if rg_path is None:
            return None

        cmd = [rg_path, "--json", "-F"]
        if include_glob:
            cmd.extend(["--glob", include_glob])

        rg_cwd: Optional[str] = None
        if base_full.is_dir():
            cmd.extend(["--", pattern, "."])
            rg_cwd = str(base_full)
        else:
            cmd.extend(["--", pattern, str(base_full)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DEFAULT_GREP_TIMEOUT,
                check=False,
                cwd=rg_cwd,
            )
        except subprocess.TimeoutExpired:
            logger.warning("ripgrep timed out; using Python grep fallback")
            return None
        except (FileNotFoundError, PermissionError, NotADirectoryError) as e:
            logger.warning("ripgrep subprocess run failed (%s); clearing path cache", type(e).__name__)
            _resolve_ripgrep_path.cache_clear()
            return None

        if proc.returncode not in (0, 1):
            return None

        results: Dict[str, List[Tuple[int, str]]] = {}
        base_resolved = base_full.resolve()

        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("type") != "match":
                continue

            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue

            raw = Path(ftext)
            p = raw if raw.is_absolute() else (base_full / raw)

            try:
                p.resolve().relative_to(base_resolved)
            except (ValueError, OSError):
                continue

            if self.virtual_mode:
                try:
                    virt = self._to_virtual_path(p)
                except (ValueError, OSError, RuntimeError):
                    continue
            else:
                virt = str(p)

            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(virt, []).append((int(ln), lt))

        return results

    async def _python_search(
            self,
            pattern: str,
            base_full: Path,
            include_glob: Optional[str],
            timeout: int = 30,
    ) -> Tuple[Dict[str, List[Tuple[int, str]]], Optional[str]]:
        """当 ripgrep 失效时的安全原生内存流式兜底搜索"""
        deadline = time.monotonic() + timeout
        results: Dict[str, List[Tuple[int, str]]] = {}
        root = base_full if base_full.is_dir() else base_full.parent

        try:
            for fp in root.rglob("*"):
                if time.monotonic() > deadline:
                    break
                try:
                    if not fp.is_file():
                        continue
                    if fp.stat().st_size > self.max_file_size_bytes:
                        continue
                    # 匹配用户传入的文件过滤规则
                    if include_glob and not fp.match(include_glob):
                        continue
                except OSError:
                    continue

                virt_path = str(fp) if not self.virtual_mode else self._to_virtual_path(fp)

                try:
                    with fp.open(encoding="utf-8", errors="ignore") as handle:
                        for line_num, raw_line in enumerate(handle, 1):
                            if line_num % 2048 == 0 and time.monotonic() > deadline:
                                break
                            if pattern in raw_line:
                                results.setdefault(virt_path, []).append((line_num, raw_line.rstrip("\n")))
                except OSError:
                    continue
        except OSError as e:
            return results, str(e)

        return results, None