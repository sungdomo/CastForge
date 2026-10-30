//! Computation graph visualization exporter (Mermaid, Graphviz DOT, JSON, Interactive HTML
//! Pass Timeline Debugger) and IR snapshot diffing translators (`castforge::viz`).

use crate::codegen::{MemoryPlan, build_memory_plan};
use crate::ir::{Module, OpId, ValueDef, ValueId};
use crate::mlir::print_mlir;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VizFormat {
    Mermaid,
    Dot,
    Json,
    Html,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphJson {
    pub module_name: Option<String>,
    pub nodes: Vec<GraphNodeJson>,
    pub edges: Vec<GraphEdgeJson>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphNodeJson {
    pub id: String,
    pub label: String,
    pub op_name: String,
    pub category: String,
    pub result_type: String,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphEdgeJson {
    pub from: String,
    pub to: String,
    pub ssa_name: String,
    pub tensor_type: String,
}

/// Frame passed to the interactive HTML Pass Timeline Debugger.
#[derive(Debug, Clone, Serialize)]
pub struct TimelineFrame {
    pub step_index: usize,
    pub pass_name: String,
    pub changed: bool,
    pub ops_removed: usize,
    pub ops_modified: usize,
    pub ops_created: usize,
    pub mlir_text: String,
    pub graph: GraphJson,
    pub memory_plan: MemoryPlan,
}

/// Render a single `Module` into Mermaid, Graphviz DOT, JSON, or standalone HTML.
pub fn render_module(module: &Module, format: VizFormat) -> String {
    match format {
        VizFormat::Mermaid => render_mermaid(module),
        VizFormat::Dot => render_dot(module),
        VizFormat::Json => render_json(module),
        VizFormat::Html => {
            let frame = TimelineFrame {
                step_index: 0,
                pass_name: "Module Snapshot".to_string(),
                changed: false,
                ops_removed: 0,
                ops_modified: 0,
                ops_created: 0,
                mlir_text: print_mlir(module),
                graph: build_graph_json(module),
                memory_plan: build_memory_plan(module),
            };
            render_timeline_html(&[frame])
        }
    }
}

/// A-to-B Visualization Translator (`Ir(Module) -> Text(Mermaid | DOT | JSON | HTML)`).
pub struct VizTranslator {
    pub format: VizFormat,
}

impl crate::ir::Translator for VizTranslator {
    fn name(&self) -> &str {
        "viz"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> anyhow::Result<crate::ir::Payload> {
        let module = input.into_ir()?;
        Ok(crate::ir::Payload::Text(render_module(
            &module,
            self.format,
        )))
    }
}

fn classify_op_category(op_name: &str) -> &'static str {
    if op_name.contains("constant") || op_name.contains("iota") {
        "constant"
    } else if op_name.contains("transpose")
        || op_name.contains("reshape")
        || op_name.contains("broadcast")
        || op_name.contains("slice")
        || op_name.contains("concatenate")
    {
        "shape"
    } else if op_name.contains("quant") {
        "quant"
    } else if op_name.contains("composite") {
        "composite"
    } else if op_name.contains("return") {
        "return"
    } else {
        "compute"
    }
}

pub fn build_graph_json(module: &Module) -> GraphJson {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut op_node_ids: HashMap<OpId, String> = HashMap::new();
    let mut arg_node_ids: HashMap<ValueId, String> = HashMap::new();
    let mut counter = 0usize;

    for func_op_id in module.functions() {
        let func_op = &module.ops[func_op_id];
        let region_id = func_op.regions[0];
        let block_id = module.regions[region_id].blocks[0];
        let block = &module.blocks[block_id];

        for &arg_id in &block.arguments {
            counter += 1;
            let nid = format!("n{counter}");
            let val = &module.values[arg_id];
            let ssa = val.ssa_name.as_deref().unwrap_or("arg");
            nodes.push(GraphNodeJson {
                id: nid.clone(),
                label: format!("%{ssa}"),
                op_name: "input_arg".to_string(),
                category: "arg".to_string(),
                result_type: val.ty.to_string(),
                attributes: HashMap::new(),
            });
            arg_node_ids.insert(arg_id, nid);
        }

        for &op_id in &block.ops {
            counter += 1;
            let nid = format!("n{counter}");
            let op = &module.ops[op_id];
            let full_name = op.name.to_string();
            let res_ty = op
                .results
                .first()
                .map(|&r| module.values[r].ty.to_string())
                .unwrap_or_else(|| "()".to_string());
            let mut attrs = HashMap::new();
            for (k, v) in &op.attributes {
                attrs.insert(k.clone(), v.to_string());
            }
            nodes.push(GraphNodeJson {
                id: nid.clone(),
                label: full_name.clone(),
                op_name: full_name.clone(),
                category: classify_op_category(&full_name).to_string(),
                result_type: res_ty,
                attributes: attrs,
            });
            op_node_ids.insert(op_id, nid.clone());

            for &operand_val in &op.operands {
                let val = &module.values[operand_val];
                let src_nid = match val.def {
                    ValueDef::BlockArgument { .. } => arg_node_ids.get(&operand_val).cloned(),
                    ValueDef::OpResult { op: prod_op, .. } => op_node_ids.get(&prod_op).cloned(),
                };
                if let Some(src) = src_nid {
                    edges.push(GraphEdgeJson {
                        from: src,
                        to: nid.clone(),
                        ssa_name: val.ssa_name.clone().unwrap_or_default(),
                        tensor_type: val.ty.to_string(),
                    });
                }
            }
        }
    }

    GraphJson {
        module_name: module.name.clone(),
        nodes,
        edges,
    }
}

pub fn render_mermaid(module: &Module) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "flowchart TD");

    let mut op_node_ids: HashMap<OpId, String> = HashMap::new();
    let mut arg_node_ids: HashMap<ValueId, String> = HashMap::new();
    let mut counter = 0usize;

    for func_op_id in module.functions() {
        let func_op = &module.ops[func_op_id];
        let fn_name = func_op
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or("main");

        let _ = writeln!(out, "  subgraph fn_{fn_name}[\"func.func @{fn_name}\"]");
        let region_id = func_op.regions[0];
        let block_id = module.regions[region_id].blocks[0];
        let block = &module.blocks[block_id];

        for &arg_id in &block.arguments {
            counter += 1;
            let nid = format!("N{counter}");
            let val = &module.values[arg_id];
            let ssa = val.ssa_name.as_deref().unwrap_or("arg");
            let _ = writeln!(out, "    {nid}([\"%{ssa} : {}\"])", val.ty);
            arg_node_ids.insert(arg_id, nid);
        }

        for &op_id in &block.ops {
            counter += 1;
            let nid = format!("N{counter}");
            let op = &module.ops[op_id];
            let _ = writeln!(out, "    {nid}[\"{}\"]", op.name);
            op_node_ids.insert(op_id, nid.clone());

            for &operand_val in &op.operands {
                let val = &module.values[operand_val];
                let src_nid = match val.def {
                    ValueDef::BlockArgument { .. } => arg_node_ids.get(&operand_val).cloned(),
                    ValueDef::OpResult { op: prod_op, .. } => op_node_ids.get(&prod_op).cloned(),
                };
                if let Some(src) = src_nid {
                    let _ = writeln!(out, "    {src} -->|\"{}\"| {nid}", val.ty);
                }
            }
        }

        let _ = writeln!(out, "  end");
    }

    out
}

pub fn render_dot(module: &Module) -> String {
    let mut out = String::new();
    let mod_name = module.name.as_deref().unwrap_or("castforge_module");
    let _ = writeln!(out, "digraph \"{mod_name}\" {{");
    let _ = writeln!(out, "  rankdir=TB;");
    let _ = writeln!(
        out,
        "  node [shape=box, style=\"rounded,filled\", \
         fillcolor=\"#f8fafc\", fontname=\"monospace\"];"
    );
    let _ = writeln!(out, "  edge [fontname=\"monospace\", fontsize=10];");

    let mut op_node_ids: HashMap<OpId, String> = HashMap::new();
    let mut arg_node_ids: HashMap<ValueId, String> = HashMap::new();
    let mut counter = 0usize;

    for func_op_id in module.functions() {
        let func_op = &module.ops[func_op_id];
        let fn_name = func_op
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or("main");

        let _ = writeln!(out, "  subgraph cluster_{fn_name} {{");
        let _ = writeln!(out, "    label=\"func.func @{fn_name}\";");
        let region_id = func_op.regions[0];
        let block_id = module.regions[region_id].blocks[0];
        let block = &module.blocks[block_id];

        for &arg_id in &block.arguments {
            counter += 1;
            let nid = format!("n{counter}");
            let val = &module.values[arg_id];
            let ssa = val.ssa_name.as_deref().unwrap_or("arg");
            let _ = writeln!(
                out,
                "    {nid} [label=\"%{ssa}\\n{}\", shape=oval, fillcolor=\"#e0f2fe\"];",
                val.ty
            );
            arg_node_ids.insert(arg_id, nid);
        }

        for &op_id in &block.ops {
            counter += 1;
            let nid = format!("n{counter}");
            let op = &module.ops[op_id];
            let _ = writeln!(out, "    {nid} [label=\"{}\"];", op.name);
            op_node_ids.insert(op_id, nid.clone());

            for &operand_val in &op.operands {
                let val = &module.values[operand_val];
                let src_nid = match val.def {
                    ValueDef::BlockArgument { .. } => arg_node_ids.get(&operand_val).cloned(),
                    ValueDef::OpResult { op: prod_op, .. } => op_node_ids.get(&prod_op).cloned(),
                };
                if let Some(src) = src_nid {
                    let _ = writeln!(out, "    {src} -> {nid} [label=\"{}\"];", val.ty);
                }
            }
        }

        let _ = writeln!(out, "  }}");
    }

    let _ = writeln!(out, "}}");
    out
}

pub fn render_json(module: &Module) -> String {
    let graph = build_graph_json(module);
    serde_json::to_string_pretty(&graph).unwrap_or_default()
}

/// Computes a unified line-by-line MLIR diff and operation-count delta between `before` and
/// `after`.
pub fn diff_modules(before: &Module, after: &Module) -> String {
    let before_txt = print_mlir(before);
    let after_txt = print_mlir(after);
    diff_mlir_strings(&before_txt, &after_txt)
}

/// Computes a unified diff between two MLIR text strings.
pub fn diff_mlir_strings(before_txt: &str, after_txt: &str) -> String {
    let before_lines: Vec<&str> = before_txt.lines().collect();
    let after_lines: Vec<&str> = after_txt.lines().collect();
    let mut out = String::new();
    let _ = writeln!(out, "--- a/module.mlir");
    let _ = writeln!(out, "+++ b/module.mlir");

    let max_len = before_lines.len().max(after_lines.len());
    let mut removed = 0usize;
    let mut added = 0usize;
    for i in 0..max_len {
        match (before_lines.get(i), after_lines.get(i)) {
            (Some(a), Some(b)) if a == b => {
                let _ = writeln!(out, "  {a}");
            }
            (Some(a), Some(b)) => {
                let _ = writeln!(out, "- {a}");
                let _ = writeln!(out, "+ {b}");
                removed += 1;
                added += 1;
            }
            (Some(a), None) => {
                let _ = writeln!(out, "- {a}");
                removed += 1;
            }
            (None, Some(b)) => {
                let _ = writeln!(out, "+ {b}");
                added += 1;
            }
            (None, None) => {}
        }
    }
    let _ = writeln!(out, "// Summary: -{removed} removed / +{added} added lines");
    out
}

/// Pipeable IR Snapshot Translator (`Ir(Module) -> Ir(Module)`).
///
/// Captures the current MLIR text into the root `builtin.module` attribute
/// `"castforge.snapshot_mlir"` so a downstream [`IrDiffTranslator`] (`ir-diff`) can emit a
/// unified before/after diff across any sequence of intervening passes.
#[derive(Default)]
pub struct SnapshotIrTranslator;

impl crate::ir::Translator for SnapshotIrTranslator {
    fn name(&self) -> &str {
        "snapshot-ir"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> anyhow::Result<crate::ir::Payload> {
        let mut module = input.into_ir()?;
        let snap = print_mlir(&module);
        if let Some(&fn_op) = module.functions().first() {
            module.ops[fn_op].attributes.insert(
                "castforge.snapshot_mlir".to_string(),
                crate::ir::Attribute::String(snap),
            );
        }
        Ok(crate::ir::Payload::from_ir(module))
    }
}

/// Pipeable IR Diff Translator (`Ir(Module) -> Text(Unified Diff)`).
///
/// Compares the incoming [`Module`] against an explicit `baseline` module or the snapshot
/// recorded by an upstream [`SnapshotIrTranslator`] (`snapshot-ir`).
#[derive(Default)]
pub struct IrDiffTranslator {
    pub baseline: Option<Module>,
}

impl IrDiffTranslator {
    /// Creates an [`IrDiffTranslator`] comparing against `baseline`.
    pub fn with_baseline(baseline: Module) -> Self {
        Self {
            baseline: Some(baseline),
        }
    }
}

impl crate::ir::Translator for IrDiffTranslator {
    fn name(&self) -> &str {
        "ir-diff"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> anyhow::Result<crate::ir::Payload> {
        let mut module = input.into_ir()?;
        let snap_attr = module.functions().first().and_then(|&fn_op| {
            module.ops[fn_op]
                .attributes
                .swap_remove("castforge.snapshot_mlir")
        });
        let after_txt = print_mlir(&module);
        let before_txt = if let Some(base) = &self.baseline {
            print_mlir(base)
        } else if let Some(crate::ir::Attribute::String(s)) = snap_attr {
            s
        } else {
            after_txt.clone()
        };
        Ok(crate::ir::Payload::Text(diff_mlir_strings(
            &before_txt,
            &after_txt,
        )))
    }
}

/// Renders a self-contained, zero-dependency interactive HTML Pass Debugger
/// allowing users to scrub through each pass step, inspect the DAG, and diff MLIR code.
pub fn render_timeline_html(frames: &[TimelineFrame]) -> String {
    let raw_json = serde_json::to_string(frames).unwrap_or_else(|_| "[]".to_string());
    // Escape backslashes and quotes so we can wrap the JSON across <=80-col JS string lines
    let escaped: String = raw_json
        .chars()
        .flat_map(|c| match c {
            '\\' => vec!['\\', '\\'],
            '"' => vec!['\\', '"'],
            '\n' => vec!['\\', 'n'],
            '\r' => vec![],
            other => vec![other],
        })
        .collect();
    let chars: Vec<char> = escaped.chars().collect();
    let mut chunk_lines = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let mut end = (i + 72).min(chars.len());
        while end > i && chars[end - 1] == '\\' {
            end -= 1;
        }
        let s: String = chars[i..end].iter().collect();
        chunk_lines.push(format!("  \"{s}\""));
        i = end;
    }
    let frames_expr = format!("JSON.parse([\n{}\n].join(''))", chunk_lines.join(",\n"));

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>CastForge — Interactive Pass Timeline & Graph Debugger</title>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}
  header {{
    background: #020617;
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  header h1 {{ font-size: 16px; font-weight: 700; letter-spacing: 0.4px; color: var(--accent); }}
  .controls {{ display: flex; align-items: center; gap: 12px; }}
  button {{
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
  }}
  button:hover {{ border-color: var(--accent); }}
  .workspace {{
    flex: 1;
    display: grid;
    grid-template-columns: 270px 1fr 460px;
    min-height: 0;
  }}
  .sidebar {{
    background: var(--panel);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    padding: 12px;
  }}
  .step-item {{
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid transparent;
    margin-bottom: 8px;
    cursor: pointer;
    transition: 0.15s;
  }}
  .step-item:hover {{ background: rgba(255,255,255,0.04); }}
  .step-item.active {{
    background: rgba(56, 189, 248, 0.12);
    border-color: var(--accent);
  }}
  .step-title {{
    font-size: 13px;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
  }}
  .badges {{ display: flex; gap: 6px; margin-top: 6px; font-size: 11px; color: var(--muted); }}
  .badge {{ padding: 2px 6px; border-radius: 4px; background: #0f172a; }}
  .badge.changed {{ color: #4ade80; }}
  .canvas-pane {{
    position: relative;
    overflow: auto;
    background: radial-gradient(#1e293b 1px, transparent 1px);
    background-size: 20px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px;
  }}
  .code-pane {{
    background: #0b1120;
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    min-height: 0;
  }}
  .pane-header {{
    padding: 10px 14px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }}
  pre {{
    flex: 1;
    padding: 14px;
    overflow: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12.5px;
    line-height: 1.55;
  }}
  .inspector {{
    border-top: 1px solid var(--border);
    padding: 12px 14px;
    font-size: 12.5px;
    background: var(--panel);
    min-height: 135px;
  }}
  .node-card {{
    width: 280px;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    border: 1.5px solid var(--border);
    background: #1e293b;
    cursor: pointer;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
  }}
  .node-card:hover {{ border-color: var(--accent); }}
  .node-card.cat-arg {{ border-color: #38bdf8; background: #0c2a40; }}
  .node-card.cat-constant {{ border-color: #a855f7; background: #2a1442; }}
  .node-card.cat-shape {{ border-color: #f59e0b; background: #382509; }}
  .node-card.cat-quant {{ border-color: #ec4899; background: #3d122b; }}
  .node-card.cat-composite {{ border-color: #10b981; background: #093527; }}
  .node-card.cat-return {{ border-color: #64748b; background: #1e293b; }}
  .node-op {{ font-family: monospace; font-weight: 700; font-size: 13px; }}
  .node-ty {{
    font-family: monospace;
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
    word-break: break-all;
  }}
  .edge-arrow {{ color: #64748b; font-size: 11px; font-family: monospace; margin: 2px 0; }}
</style>
</head>
<body>
<header>
  <h1>CastForge Pass Timeline & IR Debugger</h1>
  <div class="controls">
    <button id="prevBtn">&#8592; Prev Pass</button>
    <span id="stepIndicator" style="font-size:13px; color:var(--muted);">Step 0</span>
    <button id="nextBtn">Next Pass &#8594;</button>
  </div>
</header>
<div class="workspace">
  <div class="sidebar" id="sidebar"></div>
  <div class="canvas-pane" id="canvasPane"></div>
  <div class="code-pane">
    <div class="pane-header">MLIR IR Snapshot</div>
    <pre id="mlirCode"></pre>
    <div class="pane-header">Arena Memory & Liveness Summary</div>
    <div id="arenaPanel" style="padding:10px 14px; font-size:12px;"></div>
    <div class="pane-header">Selected Operation Inspector</div>
    <div class="inspector" id="inspector">
      Click any node in the graph to inspect its tensor types and attributes.
    </div>
  </div>
</div>
<script>
const FRAMES = {frames_expr};
let currentIdx = FRAMES.length - 1;

function selectStep(idx) {{
  currentIdx = Math.max(0, Math.min(FRAMES.length - 1, idx));
  renderSidebar();
  renderMain();
}}

function renderSidebar() {{
  const sb = document.getElementById('sidebar');
  sb.innerHTML = '';
  FRAMES.forEach((f, idx) => {{
    const div = document.createElement('div');
    div.className = 'step-item' + (idx === currentIdx ? ' active' : '');
    div.onclick = () => selectStep(idx);
    const statusCls = f.changed ? 'changed' : '';
    const statusTxt = f.changed ? 'CHANGED' : 'NO-OP';
    div.innerHTML = `
      <div class="step-title">
        <span>#${{f.step_index}} ${{f.pass_name}}</span>
        <span>${{f.graph.nodes.length}} nodes</span>
      </div>
      <div class="badges">
        <span class="badge ${{statusCls}}">${{statusTxt}}</span>
        <span class="badge">-${{f.ops_removed}}</span>
        <span class="badge">~${{f.ops_modified}}</span>
        <span class="badge">+${{f.ops_created}}</span>
      </div>
    `;
    sb.appendChild(div);
  }});
}}

function renderMain() {{
  const f = FRAMES[currentIdx];
  const total = FRAMES.length - 1;
  const ind = document.getElementById('stepIndicator');
  ind.textContent = `Step ${{f.step_index}} / ${{total}}: ${{f.pass_name}}`;
  document.getElementById('mlirCode').textContent = f.mlir_text;
  const mp = f.memory_plan || {{}};
  const saved = (mp.naive_activation_bytes || 0) - (mp.total_activation_bytes || 0);
  document.getElementById('arenaPanel').innerHTML = `
    <div><b>Weights:</b> ${{mp.total_constant_bytes || 0}} B |
    <b>Arena Workspace:</b> ${{mp.total_activation_bytes || 0}} B
    (Naive: ${{mp.naive_activation_bytes || 0}} B, Saved: ${{saved}} B,
    Slots: ${{mp.reusable_slot_count || 0}})</div>
  `;

  const canvas = document.getElementById('canvasPane');
  canvas.innerHTML = '';
  f.graph.nodes.forEach((n, i) => {{
    if (i > 0) {{
      const incoming = f.graph.edges.filter(e => e.to === n.id);
      const edgeLabel = incoming
        .map(e => (e.ssa_name ? '%' + e.ssa_name + ' : ' : '') + e.tensor_type)
        .join(' | ');
      const arrow = document.createElement('div');
      arrow.className = 'edge-arrow';
      arrow.textContent = '↓ ' + edgeLabel;
      canvas.appendChild(arrow);
    }}
    const card = document.createElement('div');
    card.className = `node-card cat-${{n.category}}`;
    card.innerHTML = `
      <div class="node-op">${{n.label}}</div>
      <div class="node-ty">${{n.result_type}}</div>
    `;
    card.onclick = () => {{
      const attrs = Object.entries(n.attributes)
        .map(([k, v]) => `<div><b>${{k}}</b>: <code>${{v}}</code></div>`)
        .join('');
      const fallback = '<div style="color:var(--muted)">No attributes</div>';
      document.getElementById('inspector').innerHTML = `
        <div style="margin-bottom:6px;">
          <b>Op:</b> <code>${{n.op_name}}</code> (${{n.category}})
        </div>
        <div style="margin-bottom:6px;"><b>Result Type:</b> <code>${{n.result_type}}</code></div>
        ${{attrs || fallback}}
      `;
    }};
    canvas.appendChild(card);
  }});
}}

document.getElementById('prevBtn').onclick = () => selectStep(currentIdx - 1);
document.getElementById('nextBtn').onclick = () => selectStep(currentIdx + 1);
window.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') selectStep(currentIdx - 1);
  if (e.key === 'ArrowRight') selectStep(currentIdx + 1);
}});

selectStep(FRAMES.length - 1);
</script>
</body>
</html>"#
    )
}
