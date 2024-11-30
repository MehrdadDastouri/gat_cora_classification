# Graph Attention Network (GAT) on Cora Dataset

"This project implements a Graph Attention Network (GAT) using PyTorch Geometric to classify nodes in the Cora citation network dataset. GAT leverages attention mechanisms to learn node representations by aggregating information from neighboring nodes."

Features:
  - Loads the Cora dataset using PyTorch Geometric, which includes:
      - 2708 nodes (scientific publications).
      - 5429 edges (citations between publications).
      - 1433 features per node (word embeddings).
      - 7 classes (topics).
  - Implements a GAT model with:
      - Two GAT layers using multi-head attention in the first layer.
      - `ELU` activation between layers.
      - Log-softmax activation for classification.
  - Evaluates the model with:
      - Negative Log-Likelihood (NLL) loss for node classification.
      - Accuracy on test nodes.
  - Visualizes learned node embeddings using t-SNE.

Dataset:
  - Name: Cora
  - Number of nodes: 2708
  - Number of edges: 5429
  - Number of features per node: 1433
  - Number of classes: 7
  - Training nodes: 140
  - Validation nodes: 500
  - Test nodes: 1000

Model Architecture:
  - Layer 1: GATConv (1433 input features, 8 hidden features per head, 8 attention heads, dropout=0.6)
  - Activation: ELU
  - Layer 2: GATConv (64 input features from multi-head output, 7 output classes, 1 attention head, dropout=0.6)
  - Output Activation: Log-Softmax for classification.
