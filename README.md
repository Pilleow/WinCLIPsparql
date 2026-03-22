# [SBK] WinCLIP + Knowledge Graph

**Student:** Igor Zamojski
**WinCLIP implementation by:** https://github.com/caoyunkang/WinClip

**WinCLIP implementation by:** https://github.com/caoyunkang/WinClip

## 🎯 Project Goal

The goal of this project is to build a simple demonstration pipeline that integrates **visual anomaly detection** with a **knowledge graph**.

A WinCLIP model (zero-/few-shot, based on CLIP) is used to detect and segment anomalies in selected classes from industrial datasets. The model output is then mapped directly to an **IRI node in an RDF knowledge graph**.  

From a user interface (UI), it is possible to execute **SPARQL queries** to retrieve recommended actions or mitigations for a given anomaly.

## 📝 Description

### 1. Data Selection (small subset)

A small subset of MVTec data is used for demonstration.

### 2. WinCLIP Inference (zero-/few-shot)

The project uses an existing implementation of WinCLIP:
- zero-shot anomaly detection (no training required)
- optional few-shot with 1-2 "good" images
- simple prompt ensemble for:
  - normal state
  - anomalous state + defect type

The focus is on **demonstration**, not model optimisation.

### 3. Mapping → Knowledge Graph Node

The model output (predicted defect type) is mapped to a specific IRI, e.g.: `ex:defect/mvtec/contamination`

A minimal RDF schema is defined with:

- `ex:DefectType` (e.g. scratch, contamination)
- `ex:Cause` (e.g. abrasive_contact)
- `ex:Action` (e.g. line_stop, replace_guide_rail)

### 4. Queries and Recommendations

Simple SPARQL queries are used to retrieve recommendations.

Example:
> "Get recommended actions for a given defect type"

Results are displayed in the UI together with:
- anomaly mask / heatmap
- predicted label
- associated actions and causes

### 5. Evaluation (lightweight)

The evaluation focuses on demonstrating functionality, not achieving SOTA performance.

For selected classes:
- image-level AUROC (i-AUROC)
- pixel-level AUROC (p-AUROC)
- qualitative visual results (screenshots)

WinCLIP reports strong performance on MVTec AD and VisA in zero-/few-shot settings - this project only verifies the pipeline on a small subset.

**Note:**  

The ontology and recommended actions do not need to be fully realistic - the goal is to demonstrate integration capabilities.

## ⚠️ Disclaimer

This project is a **proof-of-concept demo**.  
It is not intended for production use or real industrial deployment.
