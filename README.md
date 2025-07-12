# Mini-Project-XAI-SS25
This project explores GNN explainability on RDF graphs using FastRGCN and three explanation methods: Captum, GraphLIME, and PGExplainer. Implemented in PyTorch Geometric and evaluated on the AIFB dataset, it provides node and edge-level insights through visual and quantitative analysis.

# Strategy 1: Explaining GNN Predictions on RDF Data
This repository contains the complete implementation of Strategy 1 for the XAI Mini Project (SS25). I train a FastRGCN model on the AIFB RDF dataset and explain its predictions using three state-of-the-art explainability techniques:
1. Captum (Integrated Gradients)
2. GraphLIME
3. PGExplainer
   
I emphasize interpretability by using feature selection and providing human-readable RDF labels in the explanations.

# Setup Instructions
Run the following in Google Colab or a compatible environment

      #zip file uploading here
      from google.colab import files
      uploaded = files.upload()
      
      #extracting the zip file.
      import zipfile
      import os
      zip_file = list(uploaded.keys())[0]  
      extract_dir = "data"
      os.makedirs(extract_dir, exist_ok=True)
      with zipfile.ZipFile(zip_file, 'r') as zip_ref:
          zip_ref.extractall(extract_dir)
      print(f"Extracted to '{extract_dir}/'") 

# Install all required packages
    pip install -r requirements.txt
    
# Requirements
1. torch==2.1.0

2. torch-scatter

3. torch-sparse

4. torch-geometric

5. rdflib

6. pandas

7. matplotlib

8. scikit-learn

9. captum

10. networkx

11. graphviz

12. graphlime

# Model Training
You can train the FastRGCN model by executing the training cells. Below is the breakdown:

# Step 1: Load RDF Dataset & Build the Graph
1. Load the aifbfixed_complete.n3 RDF file using rdflib
2. Convert triples to a graph structure: nodes, edges, and edge types
3. Save RDF triples to all_triples.txt

# Step 2: Encode Node Features and Labels
1. Extract literals from RDF (e.g., names, descriptions)
2. Apply TF-IDF or manual features for nodes
3. Map RDF URIs to unique node indices
4. Assign ground-truth labels from trainingSet.tsv and testSet.tsv

# Step 3: Apply Feature Selection
1. Instead of PCA, apply SelectFromModel from sklearn
2. Use a linear classifier to select the top informative features
3. This helps generate human-interpretable explanations

# Step 4: Train the FastRGCN Model
1. Mask features of test nodes (x_masked) to prevent leakage
2. Use FastRGCNConv layers from PyG to model relational data
3. Train for 50 epochs using Adam optimizer and negative log-likelihood loss

# Output from Training 
    Epoch 01 | Loss: 1.5888
    Train Accuracy: 0.4429
    Test Accuracy : 0.4444
    ----------------------------------------
    Epoch 02 | Loss: 1.3060
    Train Accuracy: 0.5714
    Test Accuracy : 0.5278
    ----------------------------------------
    .
    .
    .
    ----------------------------------------
    Epoch 49 | Loss: 0.0001
    Train Accuracy: 0.8000
    Test Accuracy : 0.7222
    ----------------------------------------
    Epoch 50 | Loss: 0.0001
    Train Accuracy: 0.8000
    Test Accuracy : 0.7222
    ----------------------------------------
    Median epoch time: 0.2458 s

<img width="1027" height="450" alt="image" src="https://github.com/user-attachments/assets/bb2c3b12-7f12-45e9-a98a-b368b1706ed7" />



Note: The test features are masked before training as per the requirement.
Files generated:
1. all_triples.txt
2. test_predictions.txt
 
# Explainability: How Each Explainer Works
Each explainer targets a different aspect of interpretability:

# Captum (Integrated Gradients)
Explains feature-level importance for individual nodes.
1. Computes gradients of the prediction output with respect to input features.
2. Run on selected test nodes.
3. Visualizes top features in CaptumExplainer_Node_Visuals/.
   
Reveals: Which node features most influenced a specific prediction?

# Output 

    Selected one test node per class: [471, 659, 639, 568]

    Node 471 explanation:
    • URI            : http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL/id1855instance
    • True class URI : http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance
    • Pred class URI : http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance
    • Prediction     : Correct
    • True RDF Name  : Business Information and Communication Systems
    • Pred RDF Name  : Business Information and Communication Systems
    
    → This person was correctly predicted to belong to group 0.

# GraphLIME
Provides local feature explanations using kernel-based Lasso regression.
1. Builds a k-hop ego graph and fits a local linear model.
2. Run with different rho values to test feature sparsity.
3. Visual results are saved in GraphLIME_Node_Visuals/ and prediction logs in GraphLIME_Node_Prediction_Visuals/.

Reveals: Which features matter most within the local graph neighborhood?

# Output

    Selected one test node per class: [471, 659, 639, 568]
    GraphLIME Explanation for Node 471
    • URI            : http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL/id1855instance
    • True class URI : http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance
    • Pred class URI : http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance
    • Prediction     : Correct
    • True RDF Name  : Business Information and Communication Systems
    • Pred RDF Name  : Business Information and Communication Systems
    • Explanation    : This person was correctly predicted.
    • Top features   : [12, 11, 10, 9, 8]
    • Saved PDF      : Explanations/GraphLIME_Node_Visuals/node_471.pdf


Output with rho:

    === Explaining with rho=0.1 ===
    Node 471: 0 non-zero features (rho=0.1)
    Node 659: 0 non-zero features (rho=0.1)
    Node 639: 0 non-zero features (rho=0.1)
    Node 555: 0 non-zero features (rho=0.1)

# PGExplainer
Learns to generate edge importance masks for a node’s neighborhood.
1. Trained on a subset of nodes with known labels.
2. Explains one test node per class.
3. Visualized edge-level importance in PGExplainer_Node_Visuals/.
   
Reveals: Which edges (relations) are most critical for the model’s decision?

# Output 

    Training PGExplainer on 5 nodes for 50 epochs...
    PGExplainer training done.
    
    PGExplainer Explanation for Node 471
    • URI            : http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL/id1855instance
    • True class URI : http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance
    • Pred class URI : http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance
    • Prediction     : Correct
    • True RDF Name  : Business Information and Communication Systems
    • Pred RDF Name  : Business Information and Communication Systems
    • Explanation    : This person was incorrectly predicted. True class is 0, but predicted 0.
    • Top edges      : [14068, 13750, 13745, 14115, 14110]
    • Saved PDF      : Explanations_Visuals/PGExplainer_Node_Visuals/node_471.pdf



  

