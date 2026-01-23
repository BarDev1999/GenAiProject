# 🎵 Vinyl Album Identification from CCTV Footage

**Final Project - Computer Vision Course** *Bridging the gap between low-quality security footage and clean digital catalogs using Deep Metric Learning.*

---

## 📌 Project Overview & Motivation
Retail stores suffer from a significant **"Intent Gap"**: while Point-of-Sale (POS) data reveals what customers *bought*, store owners remain blind to what customers *browsed, picked up, and returned* to the shelf. 

**The Challenge:** Identifying vinyl albums from **CCTV footage** is difficult due to low resolution, grayscale imagery, and heavy occlusions (hands blocking the cover). 

**Our Objective:** To develop a robust **Image Retrieval System** that matches a noisy query image (Probe) to its clean counterpart in a digital catalog (Gallery).

---

## 🚀 Key Achievements
* **89% Top-5 Accuracy** on completely **unseen** album covers (Zero-Shot setting).
* **77% Top-1 Accuracy**, outperforming random guess (2%) by a factor of 38.
* Successfully bridged the domain gap using **Synthetic Data Generation (GLIGEN)**.

---

## 🛠️ Methodology & Tech Stack

### 1. Data Generation (Novelty)
We created a **Synthetic-to-Real Pipeline** to overcome data scarcity:
* **Engine:** Used **GLIGEN** (Grounded-Language-to-Image Generation) for realistic inpainting of human hands holding records.
* **Scale:** Generated **5,000 training samples**.

### 2. Model Architecture
* **Backbone:** **ResNet50** (Pretrained).
* **Training:** **Siamese Network** with **Triplet Margin Loss** (Margin=1.0) to map disparate domains into a shared embedding space.

---

## 📊 Experimental Results

| Model Setup | Precision 1 (P1) | Precision 5 (P5) | Improvement |
| :--- | :--- | :--- | :--- |
| Random Guess | 2.0% | 10.0% | - |
| ResNet18 (Baseline) | 67.0% | 78.0% | +65% |
| **ResNet50 (Final)** | **77.0%** | **89.0%** | **+11%** |

---

## 📂 Repository Structure
* **Code/**: Training, evaluation, and data generation scripts (commented).
* **Slides/**: All project presentations in **PPT and PDF** formats.
* **Data/**: Metadata files (JSON/CSV). Images hosted externally.
* **Results/**: Raw experiment outputs and JSON metrics.
* **Visuals/**: Visual Abstract and result plots.

---

## 🔗 External Data Resources (OneDrive)
* **[Full Synthetic Dataset (CCTV Query Images)](https://1drv.ms/f/c/ac19e521377b4d69/IgCQyUFf7MNZToYf7xa2h57UAcoUpXVUo9zEWjdq23Mrr6I?e=seBOPM)**
* **[Clean Catalog Images (Reference Gallery)](https://1drv.ms/f/c/ac19e521377b4d69/IgA3miqNLP7pSo1gfSObj27CAeeOcahAao2_k_xZP5BSsyo?e=Rhcm61)**

---

## 👥 Team
* **Arbel Koren**
* **Bar Sberro**
* **Noy Leibovitch**

*Submitted as part of the Computer Vision Final Project, 2026.*
!
