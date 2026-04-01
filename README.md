# COMPSCI 4NL3 – Project Step 1  
## Team 10: Idiom Classification

### Team Members
- Claire Nielsen
- Rebecca Di Filippo
- Leo Vugert
- Tim Pokanai

---

## 📌 Project Overview

This project focuses on **idiom and figurative language classification** using the PIE-English Corpus. The goal is to build models capable of identifying and classifying different types of literary devices in natural language, including metaphors, similes, euphemisms, irony, and more.

---

## 📊 Dataset

We use the **PIE-English Corpus (Potential Idiomatic Expression Corpus)**, which is derived from:
- British National Corpus (BNC) (~96.9%)
- UK Web Corpus (UKWaC) (~3.1%)

### Labels include:
- Metaphor  
- Simile  
- Euphemism  
- Personification  
- Parallelism  
- Oxymoron  
- Paradox  
- Hyperbole  
- Literal  
- Irony  

### Preprocessing
- Converted token-level data into full sentences  
- Lowercased text  
- Randomized sampling for annotation  
- Stored in CSV format for training and evaluation  

---

## 🏷️ Annotation Process

We built a custom annotation workflow using a shared Excel spreadsheet.

### Workflow:
- Each team member annotated ~300 data points
- 45 overlapping samples per annotator were re-labeled for agreement testing
- Final dataset contains **1200 annotated examples**

### Agreement Metrics:
- Cohen’s Kappa used for pairwise agreement
- Krippendorff’s Alpha used for overall reliability

---

## 🧠 Models

We experimented with multiple approaches:

- Baseline model
- Logistic Regression classifier
- Transformer-based model (RoBERTa)
  
---
*STAY TUNED FOR RESULTS*
