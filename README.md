

***

# Istanbul Air Quality Analysis and Prediction

> A data-driven analysis of Istanbul's air quality, revealing its complex pollution landscape through machine learning. This project explores whether Istanbul is a single city or multiple distinct pollution zones, identifies the primary drivers of pollution, and develops a predictive model for future air quality levels.

This repository contains the code and findings for the "Istanbul Air Quality Report," a project for the Machine Learning with Big Data (VIA 509E) course at Istanbul Technical University.

## Team

*   Aysan Pakmanesh (528241005)
*   Melis Ciğer (528241017)
*   Samet Erkek (528241041)
*   Selin Bozkurt (528241003)

**Advisor:** Dr. Suha Tuna, Assistant Professor of Computational Science & Engineering

***

## Table of Contents

1.  [Executive Summary](#executive-summary)
2.  [Key Research Questions](#key-research-questions)
3.  [Methodology](#methodology)
    *   [1. The Four Istanbuls: K-Means Clustering](#1-the-four-istanbuls-k-means-clustering)
    *   [2. The Drivers of Pollution: Principal Component Analysis (PCA)](#2-the-drivers-of-pollution-principal-component-analysis-pca)
    *   [3. Predicting the Future: PM2.5 Forecasting](#3-predicting-the-future-pm25-forecasting)
    *   [4. Contextualizing Our Research: NLP on Academic Literature](#4-contextualizing-our-research-nlp-on-academic-literature)
4.  [Key Findings](#key-findings)
5.  [Data Sources](#data-sources)
6.  [Technologies Used](#technologies-used)
7.  [Repository Structure](#repository-structure)
8.  [Conclusion](#conclusion)

## Executive Summary

This project presents a deep dive into Istanbul's air quality using a dataset of over 9 million measurements from 2020 to 2025. By applying a series of machine learning techniques, we move beyond simple metrics to uncover the hidden patterns of pollution in this vast, transcontinental city. Our analysis reveals that Istanbul does not behave as a single entity but as four distinct pollution zones, each with its own unique characteristics and drivers. We identify a critical infrastructure gap in PM2.5 monitoring, build a highly accurate predictive model to forecast pollution levels, and situate our findings within the broader academic landscape. The conclusion is clear: a one-size-fits-all approach to air quality management is insufficient for Istanbul.

## Key Research Questions

*   Is Istanbul truly one city from an air quality perspective, or does it function as multiple distinct zones?
*   What are the underlying drivers and hidden patterns of air pollution across the city?
*   Can we accurately forecast future PM2.5 levels to enable proactive public health measures?
*   How does this research fit into the existing body of academic work on air quality?

## Methodology

Our analysis is a multi-step process, with each chapter of our report building on the last.

### 1. The Four Istanbuls: K-Means Clustering

To understand the city's complex pollution landscape, we applied K-means clustering to segment Istanbul's districts based on pollution data, population density, and traffic patterns.

*   **Result:** The algorithm identified four distinct clusters, or "Istanbuls," each with a unique pollution profile.
    *   **Cluster 1: PM10-Heavy Districts:** Dominated by particulate matter, likely from traffic and construction.
    *   **Cluster 2: NO2-Dominant Urban Core:** Characterized by intense traffic and dense urban development.
    *   **Cluster 3: The Sultangazi Anomaly:** High PM10 with no corresponding CO, pointing to unique local sources or a critical gap in PM2.5 sensor coverage.
    *   **Cluster 4: O3-Dominant Coastal/Peripheral:** Characterized by ozone formed from transported pollutants.

### 2. The Drivers of Pollution: Principal Component Analysis (PCA)

To identify the primary sources of pollution, we used PCA to reduce the dimensionality of our six-pollutant dataset.

*   **Result:** We identified three principal components that explain 67.8% of the total variance in air quality.
    *   **Principal Component 1: Traffic-Urban Pollution (29.3%):** A mix of NO2, CO, and PM2.5 from vehicle exhaust.
    *   **Principal Component 2: Photochemical Activity (21.7%):** Ozone (O3) and secondary particulate matter formation.
    *   **Principal Component 3: Industrial Signatures (16.8%):** SO2 and PM10 from industrial processes and coal combustion.

### 3. Predicting the Future: PM2.5 Forecasting

Building on our analysis, we developed a predictive model to forecast PM2.5 concentrations using historical data.

*   **Model:** A Random Forest Regressor was trained on engineered features like rolling averages, lag values, and temporal indicators.
*   **Result:** The model achieved a high degree of accuracy, explaining 94% of the variance in PM2.5 levels with a Mean Absolute Error (MAE) of just 1.15 µg/m³. The most important features were short-term historical data (3-hour rolling mean and 1-hour change).

### 4. Contextualizing Our Research: NLP on Academic Literature

To understand how our work fits into the existing scientific dialogue, we analyzed 236 academic papers using TF-IDF, t-SNE, and anomaly detection.

*   **Result:** Our analysis revealed that while many studies focus on modeling, very few combine IoT sensor data with big data approaches for Turkish cities. This highlights a key innovation of our project and an opportunity for future research.

## Key Findings

1.  **Istanbul is Not One City:** A uniform, city-wide policy for air quality is insufficient. Istanbul requires zone-specific interventions tailored to the four distinct pollution profiles we identified.
2.  **Critical Infrastructure Gap:** A significant number of districts, particularly in the "Sultangazi Anomaly," lack PM2.5 monitoring. This creates a public health blind spot and artificially inflates compliance statistics.
3.  **Pollution is Driven by Three Key Factors:** The interplay between traffic, industrial emissions, and photochemical activity is the primary driver of Istanbul's air quality.
4.  **Accurate Prediction is Possible:** We demonstrated the ability to forecast PM2.5 levels with 94% accuracy, enabling proactive pollution management and public health warnings.
5.  **A Better Approach for the Region:** Our project addresses a gap in the literature by applying a big data and machine learning approach to on-the-ground sensor data in Istanbul.

## Data Sources

*   **Air Quality Data:** Over 9 million measurements from 2020-2025, obtained from the Turkish Ministry of Environment, Urbanization and Climate Change's continuous monitoring system (`sim.csb.gov.tr`).
*   **Academic Literature:** 236 academic papers sourced from the OpenAlex database (`openalex.org`).

## Technologies Used

*   **Primary Framework:** Apache PySpark (for distributed data processing)
*   **Machine Learning Libraries:** PySpark ML, Scikit-learn
*   **Core Language:** Python
*   **Key Methodologies:** K-Means Clustering, Principal Component Analysis (PCA), Random Forest, TF-IDF, t-SNE

## Repository Structure

```
.
├── data/
│   ├── processed_data/
│   │   └── istanbul_aksaray.parquet        # 1 sample station
│   ├── raw_air_quality_data/
│   │   └── Aksaray.csv                     # 1 sample station
│   └── research_data/
│       └── openalex_metadata.csv           # list of the research papers used in the actual project
│       └── W1585705538_Air Pollution in Mega Cities_ A Case Study of Istanbul.txt           # 1 sample text
│
├── scripts/
│   ├── 01_district_clustering.py
│   ├── 02_pollution_pattern_analysis.py
│   ├── 03_pm25_forecasting.py
│   └── 04_literature_analysis.py
│
├── report/                                 
│   └── Istanbul_Air_Quality_Report.pdf
│
├── README.md
└── LICENSE

```

## Conclusion

This project demonstrates that Istanbul's air pollution landscape is far more complex than a city-wide average suggests. The identification of four distinct pollution zones, the quantification of pollution drivers, and the creation of a predictive model provide a foundation for effective air quality management strategy.
