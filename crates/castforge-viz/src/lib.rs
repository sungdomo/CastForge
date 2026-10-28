pub mod timeline;
pub use timeline::render_timeline_html;

use castforge_ir::{Module, OpId, ValueDef, ValueId};
use castforge_mlir::print_mlir;
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
            };
            render_timeline_html(&[frame])
        }
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
