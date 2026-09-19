# -*- coding: utf-8 -*-
"""后端托管前端构建产物的行为约定。

要点（见设计文档 4.1 / 第 9 节）：
  - 非 /api/ 前缀的未知路径必须回退到 index.html，否则前端路由一刷新就 404
  - /api/ 前缀一律交给 router，不得被静态挂载拦截
  - dist 不存在时后端必须照常启动（不能因为前端没构建就起不来）
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import _mount_frontend


def _make_dist(tmp_path: Path, html: str = "<html><body>SPA</body></html>") -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(html, encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    return dist


def test_mount_serves_index_at_root(tmp_path: Path) -> None:
    app = FastAPI()
    _mount_frontend(app, _make_dist(tmp_path))
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "SPA" in res.text


def test_deep_frontend_route_falls_back_to_index(tmp_path: Path) -> None:
    """⚠️ Review Focus：/documents 这类前端路由刷新必须不 404。"""
    app = FastAPI()
    _mount_frontend(app, _make_dist(tmp_path))
    client = TestClient(app)
    res = client.get("/documents/sales_kb")
    assert res.status_code == 200
    assert "SPA" in res.text


def test_static_asset_is_served(tmp_path: Path) -> None:
    app = FastAPI()
    dist = _make_dist(tmp_path)
    _mount_frontend(app, dist)
    client = TestClient(app)
    res = client.get("/assets/app.js")
    assert res.status_code == 200
    assert "console.log" in res.text


def test_api_paths_are_not_intercepted(tmp_path: Path) -> None:
    """⚠️ /api/ 前缀必须原样交给 router——被静态挂载吃掉就是灾难。"""
    app = FastAPI()

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    _mount_frontend(app, _make_dist(tmp_path))
    client = TestClient(app)
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_unknown_api_path_returns_404_not_index(tmp_path: Path) -> None:
    """不存在的 /api/ 路径要 404，不能悄悄返回 HTML——那会让前端解析 JSON 时炸。"""
    app = FastAPI()
    _mount_frontend(app, _make_dist(tmp_path))
    client = TestClient(app)
    res = client.get("/api/v1/definitely-not-here")
    assert res.status_code == 404


def test_missing_dist_does_not_break_startup(tmp_path: Path) -> None:
    app = FastAPI()
    _mount_frontend(app, tmp_path / "does-not-exist")  # 不应抛错
    client = TestClient(app)
    assert client.get("/").status_code == 404
