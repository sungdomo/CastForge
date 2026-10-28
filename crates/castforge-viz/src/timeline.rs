use crate::TimelineFrame;

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
