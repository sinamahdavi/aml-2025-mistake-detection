# AML 2025 - Mistake Detection in Procedural Activities

**Course:** Advanced Machine Learning (AML) / Data Analysis and Artificial Intelligence (DAAI)  
**Project Title:** Learning to spot mistakes in procedural activities  
**Teaching Assistants:** Simone Alberto Peirone, Gaetano Salvatore Falco  
**Version:** 1.0 (5 Nov. 2025)

---

## 🎯 Project Overview

This project focuses on **procedure understanding** - teaching models to recognize, segment, and reason about the sequence of actions that compose complex human activities. Specifically, we explore the **mistake detection task**, which involves detecting and recognizing different kinds of errors in the steps of a procedural activity.

### Key Concepts

- **Task Graphs**: A compact representation of sequential steps required to complete a recipe
  - Special form of directed acyclic graphs (DAGs)
  - Can be manually annotated or learned from multiple human demonstrations
  - Used to predict recipe correctness by verifying steps satisfy possible paths

### Project Structure

1. **Part 1**: Train simple baseline models to predict correctness of procedure steps from video snippets
2. **Part 2**: Extension - Move to task verification setting (classify correctness of entire video recipe given task graph)

---

## 🚨 Important Notes

- Join the specific Discord channel for assistance during development
- Weekly meetings organized until end of course for Q&A
- Keep code on GitHub (fork project repository)
- Store features, videos, and annotations on Google Drive
- Use Google Colab for free GPU resources

---

## 📚 Step 1: Literature Review

Become familiar with relevant works in procedural activity understanding:

### Key Papers

**Procedure Learning (PL):**
1. **[PL1]** Peddi, Rohith, et al. "CaptainCook4D: A dataset for understanding errors in procedural activities." NeurIPS 2024.
2. **[PL2]** Seminara, Luigi, et al. "Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos." NeurIPS 2024.
3. **[PL3]** Flaborea, Alessandro, et al. "PREGO: online mistake detection in PRocedural EGOcentric videos." CVPR 2024.
4. **[PL4]** Peirone, Simone Alberto, et al. "HiERO: understanding the hierarchy of human behavior enhances reasoning on egocentric videos." ICCV 2025.
5. Bansal, Siddhant et al. "My view is the best view: Procedure learning from egocentric videos." ECCV 2022.
6. Chowdhury, Sayeed Shafayet et al. "Opel: Optimal transport guided procedure learning." NeurIPS 2024.

**Video Backbones (VB):**
1. Girdhar, Rohit, et al. "Omnivore: A single model for many visual modalities." CVPR 2022.
2. Feichtenhofer, Christoph, et al. "Slowfast networks for video recognition." ICCV 2019.
3. Girdhar, Rohit, et al. "Imagebind: One embedding space to bind them all." CVPR 2023.
4. **[VB4]** Lin, Kevin Qinghong, et al. "Egocentric video-language pretraining." NeurIPS 2022.
5. **[VB5]** Bolya, Daniel, et al. "Perception encoder: The best visual embeddings are not at the output of the network." arXiv preprint arXiv:2504.13181 (2025).

**Others (O):**
1. **[O1]** Thost, Veronika et al. "Directed Acyclic Graph Neural Networks." ICLR.

---

## 🔧 Step 2: Mistake Detection Baselines

### Task: Supervised Error Recognition (SupervisedER)

**Goal:** Binary classification of video segments as correct/incorrect execution

### Baselines from CaptainCook4D

- **V1**: Simple Multi-Layer Perceptron (MLP) head on pre-extracted sub-segment features
- **V2**: Transformer layer to combine cues from all sub-segments
- **V3**: Multi-modal (RGB + text + audio + depth) - *Not implemented in this project*

We focus on **V1 and V2** only.

### Substeps

#### 📌 Substep 2.1: Download Pre-extracted Features

- Use transfer learning with pre-trained backbones:
  - **Omnivore**
  - **SlowFast**
- Download pre-extracted features from CaptainCook4D dataset release
- **Key Question:** What are input/output of pre-trained models in feature extraction phase?

#### 📌 Substep 2.2: Reproduce V1 and V2 Baselines

- Replicate results from original paper
- **Metrics:** Accuracy, Precision, Recall, F1, AUC
- **Additional Analysis:** Performance on different error types
- **New Baseline:** Propose and compare (e.g., RNN/LSTM on step sequences)

#### 📌 Substep 2.3: Extend to New Features Extraction Backbone

- Adapt CaptainCook4D features extraction code
- Consider new backbones:
  - **EgoVLP** [VB4]
  - **PerceptionEncoder** [VB5]
- Download resized dataset version

---

## 🚀 Extension: Task Verification

**Goal:** Predict whether entire video recipe is correct execution (not just individual steps)

⚠️ **Must discuss with TA before starting!**

⚠️ **Use EgoVLP or PerceptionEncoder features** (aligned video-text embedding spaces)

### Three-Stage Pipeline

1. **Step Localization**: Detect individual steps in recipe video
2. **Step Matching**: Match detected steps to task graph nodes
3. **Classification**: Learn classifier to predict correct/incorrect recipe

### Substep 1: Recipe Step Localization

**Approaches:**
- Pre-trained ActionFormer (from CaptainCook4D)
- Zero-shot clustering (e.g., HiERO [PL4])

**Output:** List of tuples `(start, end)` for each step

**Process:**
- Compute step-level embedding by averaging video features within (start, end) boundaries
- Result: Sequence of step-level embeddings per video

### Substep 2: Simple Task-Verification Baselines

- Train baseline using binary labels (correct/incorrect executions)
- **Model:** Transformer layer + binary classification head
- **Evaluation:** Leave-one-out (train on k-1 recipes, test on k-th)

### Substep 3: Task-Graph Encoding + Step Matching

**Process:**
1. Encode task graph step descriptions using EgoVLP/PE textual encoder
2. Match visual steps to task graph nodes using **Hungarian matching algorithm**
3. Assumptions:
   - Each visual step → at most one node
   - Each node → at most one visual step
4. Update matched node features with learnable projection (node features + visual features)

### Substep 4: Classification of Observed Task-Graph

**Model:** Graph Neural Network (GNN) classifier
- Predict correct/incorrect from task-graph realization
- Consider **DAGNN** [O1] - specifically designed for DAGs
- Reference: Use provided notebook for training

---

## 📝 Project Deliverables

### Report Requirements

- **Template:** CVPR format
- **Length:** 8 pages
- **Structure:** Paper-like format
  - Abstract
  - Introduction
  - Related Works
  - Method
  - Experiments
  - Conclusion
- **Focus:** Implementation of extension

---

## 💻 Technical Setup

### Environment
- **Platform:** Google Colab (free GPU resources)
- **Storage:** Google Drive for data
- **Version Control:** GitHub (fork project repository)

### Workflow
1. Keep code on GitHub
2. Store features/videos/annotations on Google Drive
3. Copy needed data from Drive to Colab instance on startup

---

## 🗺️ Project Roadmap

### Phase 1: Foundation (Step 1-2)
- [ ] Complete literature review
- [ ] Download pre-extracted features (Omnivore, SlowFast)
- [ ] Implement V1 baseline (MLP)
- [ ] Implement V2 baseline (Transformer)
- [ ] Analyze performance on different error types
- [ ] Propose new baseline (RNN/LSTM)
- [ ] Extract features with new backbone (EgoVLP/PE)

### Phase 2: Extension (Task Verification)
- [ ] Discuss extension with TA
- [ ] Implement step localization (ActionFormer or HiERO)
- [ ] Compute step-level embeddings
- [ ] Train simple task-verification baseline
- [ ] Implement task-graph encoding
- [ ] Implement Hungarian matching for steps
- [ ] Train GNN classifier (DAGNN)
- [ ] Evaluate leave-one-out

### Phase 3: Report & Submission
- [ ] Write CVPR format report (8 pages)
- [ ] Document all experiments and results
- [ ] Analyze extension performance
- [ ] Prepare final submission

---

## 📊 Key Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **AUC**
- Performance per error type

---

## 🔑 Key Terminology

- **Procedural Activity:** Sequence of steps to complete a task (e.g., recipe)
- **Task Graph:** DAG representing all possible correct executions
- **Step Localization:** Detecting individual steps in video
- **SupervisedER:** Supervised Error Recognition task
- **Task Verification:** Predicting correctness of entire recipe (not just steps)
- **Hungarian Matching:** Algorithm for optimal one-to-one matching
- **GNN:** Graph Neural Network
- **DAGNN:** GNN specialized for Directed Acyclic Graphs

---

## 🤔 Questions to Explore

1. What are the inputs/outputs of pre-trained models in feature extraction?
2. How do different error types affect model performance?
3. How does the new baseline (RNN/LSTM) compare to V1/V2?
4. What's the performance difference between different feature backbones?
5. How well does step localization work in practice?
6. How effective is Hungarian matching for step-to-node alignment?
7. What GNN architecture works best for task verification?
8. How does task-verification performance compare to step-level error detection?

---

## 📌 Notes & Ideas

### Use Case Example
**Morning Omelette:**
- Break eggs → Mix ingredients → Cook
- Some steps can be omitted (e.g., cheese)
- Some steps can be reordered (ingredient mixing order)
- **Mistake:** Forget about pan → burnt omelette

### Feature Extraction Backbones

| Backbone | Type | Purpose |
|----------|------|---------|
| Omnivore | Multi-modal | Visual features |
| SlowFast | Video | Action recognition |
| EgoVLP | Video-Language | Aligned embeddings |
| PerceptionEncoder | Vision | Best visual embeddings |

### Task Graph Properties
- Directed Acyclic Graph (DAG)
- Represents all valid recipe executions
- Nodes = recipe steps
- Edges = step dependencies/order
- Multiple paths possible = flexibility in execution

---

## 🔗 Important Links

- Discord channel: [Join for support]
- Dataset: CaptainCook4D
- Pre-extracted features: [Download link]
- Resized dataset: [Download link]
- GNN training notebook: [Reference notebook]

---

## ⏰ Timeline

- **Weekly meetings** until end of course
- Regular TA office hours on Discord
- Incremental development recommended
- Discuss extension early with TA

---

## 🎓 Learning Objectives

1. Understand procedural activity understanding
2. Master transfer learning with pre-trained backbones
3. Implement transformer-based models
4. Work with task graphs and DAGs
5. Apply Graph Neural Networks
6. Conduct thorough experimental evaluation
7. Write academic-style technical reports

---

*Last Updated: December 20, 2025*
