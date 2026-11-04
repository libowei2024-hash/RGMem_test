#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PersonaMem-v1 数据整理：按原始 persona_id 关联共享历史，输出 20 个工作表。

用法：
  python personamem_20_personas_clean.py \
    --questions questions_32k.csv \
    --contexts shared_contexts_32k.jsonl \
    --out PersonaMem_20_Personas.xlsx

不调用任何 LLM API，不依赖 pandas/openpyxl，不修改原始数据。
不推断 Session、不根据 system 消息切分；只保留原始消息顺序、索引、角色、内容、共享历史 ID。
兼容 all_options JSON/单引号字面量混合格式；四选项强校验。
"""

import argparse
import ast
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

# 以下为对 32K 原始 system persona 文本的简短中文概括（初始设定，并非轨迹分析）。
# 需要核查完整初始描述，可查看每位用户工作表中的 system 消息。
PROFILE = {
    0: ("Kanoa Manu", "软件工程师；爱好数字音乐创作，尝试融合传统太平洋音乐与电子乐。"),
    1: ("Alex Martinez", "资深内容创作与分析从业者；重视信息核验、内容质量和行业指导。"),
    2: ("Alex", "Python 编程爱好者；也喜欢徒步与摄影，持续学习新技术。"),
    3: ("Arjun Patel", "水下曲棍球业余运动员兼博主；热衷比赛、分享训练经验。"),
    4: ("Lisa Johnson", "移动应用创业者；关注安卓应用国际化、本地化与用户增长。"),
    5: ("Dr. Isabella Martinez", "进化生物学研究者；关注遗传适应、物种多样化与科学教育。"),
    6: ("Priya", "市场营销从业者；长期关注大学橄榄球队，喜欢组织观赛活动。"),
    7: ("Alex Martinez", "平面设计学生；偏爱游戏和数字视觉创作，喜欢色彩与造型设计。"),
    8: ("Alex 'Eagle' Rivers", "大学橄榄球运动员；重视训练与团队合作，希望进入职业赛场。"),
    9: ("Marcus Green", "可持续发展咨询顾问；推动游戏赛事降低环境影响，同时热爱游戏。"),
    10: ("Dr. Alex Tan", "大学历史教授；研究中国古代东吴历史，并参与历史公共传播。"),
    11: ("Mei-Ling Tan", "理财顾问；擅长为家庭设计节约预算的活动和生活规划。"),
    12: ("Jordan Ellis", "药物化学研究者；熟悉精神活性药物领域，重视药物安全和青年科研培养。"),
    13: ("Alex Rivera", "资深保险行业管理者；关注预测建模的价值及其准确性与风险。"),
    14: ("Leilani Hayes", "泰拳运动员；擅长探索非常规搏击技术，也指导年轻选手。"),
    15: ("Malia", "说唱音乐人；作品风格鲜明，常表达社会议题并融合传统音乐元素。"),
    16: ("Alex Rosenberg", "科学记者；关注极端环境微生物及其应用，擅长科普传播。"),
    17: ("Alex Rivera", "植物学家；热爱野花，长期关注生物多样性和植物保护。"),
    18: ("Kai", "喜剧爱好者；尤其喜欢 Louie Anderson 的表演和家庭主题幽默。"),
    19: ("Maria Gonzalez", "航空与军事科技杂志编辑；热爱航空新闻、航空展和博物馆。"),
}


def read_sources(qpath: Path, cpath: Path):
    with qpath.open("r", encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f))
    contexts = {}
    with cpath.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"JSONL 第 {line_num} 行不是 context_id -> messages 字典")
            for context_id, messages in data.items():
                if context_id in contexts:
                    raise ValueError(f"重复 shared_context_id: {context_id}")
                if not isinstance(messages, list):
                    raise ValueError(f"{context_id} 不是消息列表")
                contexts[context_id] = messages
    persona_contexts = defaultdict(list)
    assigned = {}
    for q in questions:
        pid = int(q["persona_id"])
        cid = q["shared_context_id"]
        if cid not in contexts:
            raise ValueError(f"问题 {q['question_id']} 找不到对应 shared_context_id={cid}")
        if cid in assigned and assigned[cid] != pid:
            raise ValueError(f"同一 context 关联了多个 persona: {cid}")
        assigned[cid] = pid
        if cid not in persona_contexts[pid]:
            persona_contexts[pid].append(cid)
    unused = sorted(set(contexts) - set(assigned))
    if unused:
        raise ValueError(f"有 {len(unused)} 个 context 未能从 CSV 映射到 persona；已停止，避免错误归属")
    return questions, contexts, persona_contexts


def safe_xml_text(value):
    text = str(value if value is not None else "")
    # Excel 单元格上限 32767 字符；保持完整内容优先，遇超限应报错而非悄悄裁剪。
    if len(text) > 32767:
        raise ValueError(f"有一格超出 Excel 最大长度（{len(text)} 字符），请调整输出方式")
    # OOXML 禁止多数 C0 控制字符。
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return escape(text, {'"': '&quot;'})


def excel_col(n):
    output = ""
    while n:
        n, rest = divmod(n - 1, 26)
        output = chr(65 + rest) + output
    return output


def sheet_xml(rows, widths, filter_header=True):
    ncols = max(map(len, rows), default=1)
    chunks = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
        '<sheetFormatPr defaultRowHeight="18"/>',
        '<cols>' + ''.join(f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>' for i, width in enumerate(widths, 1)) + '</cols>',
        '<sheetData>',
    ]
    for ridx, row in enumerate(rows, 1):
        height = ' ht="27" customHeight="1"' if ridx == 1 else ''
        chunks.append(f'<row r="{ridx}"{height}>')
        for colidx, value in enumerate(row, 1):
            col = excel_col(colidx)
            # 所有值作为文本存储，避免 ID 精度丢失与公式注入。Excel 筛选依然可用。
            chunks.append(
                f'<c r="{col}{ridx}" s="{1 if ridx == 1 else 2}" t="inlineStr">'
                f'<is><t xml:space="preserve">{safe_xml_text(value)}</t></is></c>'
            )
        chunks.append('</row>')
    chunks.append('</sheetData>')
    if filter_header and len(rows) > 1:
        chunks.append(f'<autoFilter ref="A1:{excel_col(ncols)}{len(rows)}"/>')
    chunks.append('</worksheet>')
    return ''.join(chunks)


def save_xlsx(out_path: Path, sheets):
    """零依赖生成标准 .xlsx；每张表：sheet name, rows, column widths。"""
    mainns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    relns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    pkg_relns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    contentns = 'http://schemas.openxmlformats.org/package/2006/content-types'
    workbook = ''.join([
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<workbook xmlns="{mainns}" xmlns:r="{relns}"><sheets>',
        *[f'<sheet name="{safe_xml_text(name)}" sheetId="{i}" r:id="rId{i}"/>'
          for i,(name,_,_) in enumerate(sheets,1)],
        '</sheets></workbook>',
    ])
    rels = f'<Relationships xmlns="{pkg_relns}">' + ''.join(
        f'<Relationship Id="rId{i}" Type="{relns}/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1,len(sheets)+1)
    ) + f'<Relationship Id="rId{len(sheets)+1}" Type="{relns}/styles" Target="styles.xml"/></Relationships>'
    content = f'<Types xmlns="{contentns}">' \
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' \
              '<Default Extension="xml" ContentType="application/xml"/>' \
              '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' \
              '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' \
              + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(sheets)+1)) + '</Types>'
    styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="{mainns}">
<fonts count="2"><font><sz val="10"/><name val="Microsoft YaHei"/></font><font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Microsoft YaHei"/></font></fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF19466C"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFill="1" applyFont="1" applyAlignment="1"><alignment vertical="center"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="0"/></xf></cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('[Content_Types].xml', content)
        archive.writestr('_rels/.rels', f'<Relationships xmlns="{pkg_relns}"><Relationship Id="rId1" Type="{relns}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', workbook)
        archive.writestr('xl/_rels/workbook.xml.rels', rels)
        archive.writestr('xl/styles.xml', styles)
        for i,(_,rows,widths) in enumerate(sheets,1):
            archive.writestr(f'xl/worksheets/sheet{i}.xml', sheet_xml(rows, widths))


def build(qpath, cpath, out):
    questions, contexts, persona_contexts = read_sources(qpath, cpath)
    by_persona = defaultdict(list)
    for q in questions:
        by_persona[int(q['persona_id'])].append(q)
    persona_ids = sorted(persona_contexts)
    overview = [["persona_id", "姓名/代称", "初始画像概述（取自 system persona）", "共享历史数", "消息数", "评测题数", "说明"]]
    q_rows = [["persona_id", "姓名", "shared_context_id", "题号(question_id)", "题型", "主题", "历史截止index", "问题", "选项A", "选项B", "选项C", "选项D", "正确选项"]]
    sheets = []
    total_msg = 0
    for pid in persona_ids:
        cids = persona_contexts[pid]
        qlist = by_persona[pid]
        name, desc = PROFILE.get(pid, (f"Persona {pid}", "请查看初始 system 消息"))
        nmessages = 0
        msg_rows = [["shared_context_id", "原始消息索引(从0起)", "role", "原始消息内容(保留全文)"]]
        for cid in cids:
            messages = contexts[cid]
            nmessages += len(messages)
            for idx,m in enumerate(messages):
                msg_rows.append([cid, str(idx), m.get('role',''), m.get('content','')])
        total_msg += nmessages
        overview.append([str(pid), name, desc, str(len(cids)), str(nmessages), str(len(qlist)), "仅保留原始共享历史标识；未推断 Session"])
        sheets.append((f"persona_{pid:02d}", msg_rows, [40,22,12,100]))
        for q in qlist:
            # 同时兼容 JSON 列表与 Python 字面量列表（部分题目使用单引号）。
            raw_options = q['all_options']
            try:
                opts = json.loads(raw_options)
            except json.JSONDecodeError:
                try:
                    opts = ast.literal_eval(raw_options)  # 不使用不安全的 eval()
                except (SyntaxError, ValueError) as e:
                    raise ValueError(f"题目 {q['question_id']} 选项解析失败：{e}") from e
            if not isinstance(opts, (list, tuple)) or len(opts) != 4:
                raise ValueError(f"题目 {q['question_id']} 应有四个选项，实际为 {type(opts).__name__}，数量不正确")
            clean_opts = []
            for letter, option in zip('abcd', opts):
                if not isinstance(option, str):
                    raise ValueError(f"题目 {q['question_id']} 选项 {letter} 不是文本")
                m = re.match(r'^\s*\(([a-dA-D])\)\s*', option)
                if m is None or m.group(1).lower() != letter:
                    raise ValueError(f"题目 {q['question_id']} 选项 {letter} 标签错误")
                # 列标题已标记选项 A/B/C/D，去掉正文开头的 (a)/(b)/(c)/(d)。
                clean_opts.append(option[m.end():])
            opts = clean_opts
            q_rows.append([str(pid),name,q['shared_context_id'],q['question_id'],q['question_type'],q['topic'],q['end_index_in_shared_context'],q['user_question_or_message'],*opts,q['correct_answer']])
    notice = [
        ["主题", "说明"],
        ["数据范围", "PersonaMem-v1 / 32K，按 questions_32k.csv 中的 persona_id 关联 shared_contexts_32k.jsonl"],
        ["Persona", "20 个 persona_id，37 份 shared_context，共 589 道题"],
        ["Context", "shared_context_id 是原始上下文键；独立保留，不推测上下文之间的时间关系"],
        ["Session", "原始文件未提供显式 session_id；不做任何 Session 推断或划分"],
        ["role=system", "这是基准提供的用户画像/系统设定；和用户真实发言要区分；不能直接当作用户行为证据"],
        ["历史截止", "questions 表 end_index_in_shared_context 为评测历史边界；遵循原协议判断是否包含边界消息"],
        ["用户概述", "基于初始 system persona 作人工简述，只是静态初始画像，不代表随对话演化后的最终状态"],
        ["安全性", "脚本不调用 OpenAI API，也不修改任何 RGMem 数据库、容器或项目配置"],
    ]
    workbook_sheets = [
        ("阅读说明", notice, [24,104]),
        ("Persona总览", overview, [15,25,96,16,24,16,54]),
        ("评测题目", q_rows, [13,22,40,38,40,24,20,82,82,82,82,82,15]),
        *sheets,
    ]
    save_xlsx(out, workbook_sheets)
    print(f"已生成：{out.resolve()}")
    print(f"Persona: {len(persona_ids)}, Context: {len(contexts)}, Messages: {total_msg}, Questions: {len(questions)}")
    print(f"Excel 工作表: {len(workbook_sheets)}（阅读说明、总览、评测题 + 每个 persona 一张）")


def main():
    parser = argparse.ArgumentParser(description="严格按原始 persona_id / shared_context_id 整理 PersonaMem Excel（零依赖，不推断 Session）")
    parser.add_argument('--questions', type=Path, default=Path('questions_32k.csv'))
    parser.add_argument('--contexts', type=Path, default=Path('shared_contexts_32k.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('PersonaMem_32K_20_Personas.xlsx'))
    a = parser.parse_args()
    build(a.questions, a.contexts, a.out)

if __name__ == '__main__':
    main()
