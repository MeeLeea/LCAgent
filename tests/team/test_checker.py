"""测试 team/checker/checker.py 的共享校验工具函数。

覆盖:
- parse_filelist: 解析 .f,跳过注释/指令行,返回 .v/.sv/.vhd/.vhdl token;文件缺失返回空列表
- coverage_inclusion_ok: syn_filelist ⊆ sim_filelist 集合包含判定;syn 缺失/为空时返回 False
- write_text_file: 同步写文本文件(供 asyncio.to_thread 调用)
"""
from __future__ import annotations

from team.checker.checker import coverage_inclusion_ok, parse_filelist, write_text_file


def test_parse_filelist(tmp_path):
    """parse_filelist 解析 .f,跳过注释/指令行,返回 .v/.sv token。"""
    fl = tmp_path / "syn.f"
    fl.write_text(
        "# comment\n"
        "src/top.sv\n"
        "// line comment\n"
        "+incdir+./inc\n"
        "-f other.f\n"
        "src/util.v\n"
        "src/pkg.vhd  # trailing\n"
    )
    out = parse_filelist(str(tmp_path), "syn.f")
    assert out == ["src/top.sv", "src/util.v", "src/pkg.vhd"]


def test_parse_filelist_missing(tmp_path):
    """filelist 不存在时返回空列表。"""
    assert parse_filelist(str(tmp_path), "nope.f") == []


def test_coverage_inclusion_ok(tmp_path):
    """coverage_inclusion_ok:syn ⊆ sim 时为 True,否则 False。"""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "syn_filelist.f").write_text("src/a.sv\n")
    (tmp_path / "scripts" / "sim_filelist.f").write_text("src/a.sv\nsrc/b.sv\n")
    assert coverage_inclusion_ok(str(tmp_path)) is True
    (tmp_path / "scripts" / "sim_filelist.f").write_text("src/b.sv\n")
    assert coverage_inclusion_ok(str(tmp_path)) is False


def test_coverage_inclusion_ok_no_syn(tmp_path):
    """syn_filelist.f 为空/不存在时返回 False。"""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "sim_filelist.f").write_text("src/a.sv\n")
    assert coverage_inclusion_ok(str(tmp_path)) is False


def test_write_text_file(tmp_path):
    """write_text_file 同步写入文本内容。"""
    target = tmp_path / "out.log"
    write_text_file(str(target), "hello\nworld")
    assert target.read_text(encoding="utf-8") == "hello\nworld"
