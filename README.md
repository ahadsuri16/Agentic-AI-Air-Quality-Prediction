# 🌍 Agentic AI for Real-Time Air Quality Prediction and Autonomous Environmental Response

[![IEEE Paper](https://img.shields.io/badge/Paper-IEEE%20Format-00629B?style=for-the-badge&logo=ieee&logoColor=white)](./paper/Agentic_AI_for_Real_Time.tex)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20(R%C2%B2%3D0.9929)-orange?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Multi-Agent AI](https://img.shields.io/badge/Framework-LangChain%20Multi--Agent-teal?style=for-the-badge)](https://www.langchain.com/)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Research Publication & Software Platform:** A hybrid Artificial Intelligence platform combining high-precision machine learning forecasting (**XGBoost**, $R^2 = 0.9929$) with an autonomous **Multi-Agent Large Language Model (LLM)** framework to transition smart cities from passive environmental monitoring to proactive public health intervention.

---

## 👥 Authors & Affiliations

* **Muhammad Saad Khalid Khan Suri** — *Department of Computer Science, University of Engineering and Technology (UET), Lahore, Pakistan*
* **Abqa Javed** — *Department of Computer Science, University of Engineering and Technology (UET), Lahore, Pakistan*
* **Aatif Hussain** — *Department of Computer Science, University of Engineering and Technology (UET), Lahore, Pakistan*
* **Abdul Ahad Khan Suri** — *Department of Computer Science, The University of Lahore, Lahore, Pakistan*

📄 **Research Paper:** The full LaTeX publication source and figures are accessible in the [`paper/`](./paper) directory.

---

## 📑 Table of Contents
- [Executive Abstract](#-executive-abstract)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Multi-Agent Orchestration](#-multi-agent-orchestration)
- [Explainable AI with SHAP](#-explainable-ai-with-shap)
- [Predictive Performance Benchmarks](#-predictive-performance-benchmarks)
- [Interactive Streamlit Dashboard](#-interactive-streamlit-dashboard)
- [Docker Microservices Deployment](#-docker-microservices-deployment)
- [Installation & Quickstart](#-installation--quickstart)
  - [Option A: Local Development Setup](#option-a-local-development-setup)
  - [Option B: One-Command Docker Compose](#option-b-one-command-docker-compose)
- [API Documentation](#-api-documentation)
- [Citation (BibTeX)](#-citation-bibtex)
- [License](#-license)

---

## 💡 Executive Abstract

Rapid urbanization and industrial expansion have intensified atmospheric pollution, posing grave risks to global public health. Traditional environmental monitoring infrastructure provides static, retrospective visualizations but lacks the capacity for autonomous, real-time intervention. 

This platform introduces an end-to-end framework integrating:
1. **Deterministic Machine Learning:** An Extreme Gradient Boosting (**XGBoost**) regressor trained on multi-pollutant concentrations and meteorological indices, attaining an **$R^2$ score of 0.9929** and a **Mean Absolute Error (MAE) of 0.85**.
2. **Autonomous Multi-Agent Orchestration:** A LangChain-driven multi-agent system coordinating specialized agents (*Monitoring, Prediction, Recommendation, Reporting*) to dispatch targeted health advisories and dynamic municipal mitigation strategies upon pollutant threshold breaches.
3. **Containerized Microservices:** Asynchronous FastAPI backend delivering sub-10ms telemetry ingestion latency, coupled with a glassmorphic Streamlit monitoring dashboard and simulated IoT sensor network.

---

## 🏗 System Architecture

The system operates across a dual-layer hybrid architecture: a deterministic predictive layer that computes real-time AQI scores, paired with an asynchronous agentic layer executing prompt-chained LLM interventions.

<div align="center">
  <img src="assets/image1.jpeg" width="95%" alt="System Architecture Diagram"/>
  <p><i>Figure 1: Hybrid System Architecture Pipeline integrating IoT telemetry, XGBoost predictive modeling, and LangChain multi-agent orchestration.</i></p>
</div>

---

## 🚀 Key Features

* **High-Accuracy AQI Forecasting:** Predicts continuous EPA Air Quality Index values using $PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$, Temperature, Relative Humidity, Wind Speed, and Atmospheric Pressure.
* **Explainable AI (XAI):** Incorporates **SHAP (SHapley Additive exPlanations)** values to uncover pollutant contribution rankings and model transparency for regulatory compliance.
* **Autonomous Multi-Agent AI:** 4 specialized agents collaborate via event-driven prompt chaining to issue municipal interventions and sensitive population alerts.
* **Sub-10ms Ingestion Latency:** FastAPI async architecture with background worker tasks ensuring real-time IoT packet processing without request blocking.
* **Streamlit Glassmorphism Dashboard:** Interactive Plotly gauges, time-series pollutant curves, automated recommendations, and a conversational citizen advisory chat interface.
* **IoT Sensor Telemetry Simulator:** Python-based multi-sensor generator with automated backend health verification and customizable anomaly spike injection.

---

## 🤖 Multi-Agent Orchestration

<div align="center">
  <img src="assets/image3.jpeg" width="85%" alt="Multi-Agent Communication Architecture"/>
  <p><i>Figure 2: Multi-Agent Communication Architecture illustrating specialized agent workers orchestrated via LangChain.</i></p>
</div>

The multi-agent framework comprises four specialized roles:
1. **Monitoring Agent:** Evaluates streaming environmental telemetry, classifies hazard levels, and triggers anomaly alerts when AQI thresholds are breached.
2. **Prediction Agent:** Interfaces with the XGBoost inference pipeline to forecast next-hour AQI trends.
3. **Recommendation Agent:** Synthesizes meteorological context to generate tailored medical guidance for vulnerable groups and traffic/industrial mitigation advisories for city authorities.
4. **Reporting Agent:** Maintains conversational context to answer direct public health and atmospheric queries from citizens.

---

## 🔍 Explainable AI with SHAP

<div align="center">
  <img src="assets/image2.jpeg" width="80%" alt="SHAP Summary Plot"/>
  <p><i>Figure 3: SHAP summary plot demonstrating feature importance and directional impact of pollutant concentrations on AQI predictions.</i></p>
</div>

Feature attribution demonstrates that fine particulate matter ($PM_{2.5}$) and inhalable particles ($PM_{10}$) exhibit the strongest positive correlation with elevated AQI scores, with atmospheric wind speed acting as the primary mitigating dispersion factor.

---

## 📊 Predictive Performance Benchmarks

Comparative evaluation across standardized environmental benchmark datasets:

| Model | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **XGBoost (Proposed)** | **0.9929** | **0.85** | **1.12** |
| Random Forest Regressor | 0.9781 | 1.42 | 1.95 |
| Linear Regression | 0.8842 | 4.12 | 5.21 |
| Support Vector Regressor (SVR) | 0.8610 | 4.88 | 6.04 |

---

## 🖥 Interactive Streamlit Dashboard

<div align="center">
  <img src="assets/image4.jpeg" width="95%" alt="Streamlit Air Quality Dashboard"/>
  <p><i>Figure 4: Real-time glassmorphism monitoring dashboard with dynamic AQI gauges, pollutant telemetry grids, and autonomous agent advisories.</i></p>
</div>

---

## 🐳 Docker Microservices Deployment

<div align="center">
  <img src="assets/image5.jpeg" width="80%" alt="Dockerized Microservices Architecture"/>
  <p><i>Figure 5: Containerized deployment topology utilizing isolated bridge networking across API, Dashboard, and IoT Simulation containers.</i></p>
</div>

---

## 📂 Repository Directory Tree

```text
Agentic-AI-Air-Quality-Prediction/
├── assets/                          # Architectural diagrams and SHAP figures
├── docker/                          # Production container specifications
│   ├── Dockerfile.api               # FastAPI Backend Container
│   ├── Dockerfile.dashboard         # Streamlit Dashboard Container
│   └── Dockerfile.simulator         # IoT Sensor Simulator Container
├── paper/                           # Academic Research Publication
│   ├── Agentic_AI_for_Real_Time.tex # Complete IEEE LaTeX Source
│   └── figures/                     # High-resolution publication figures
├── src/
│   ├── model/                       # XGBoost Predictive Pipeline & SHAP
│   │   └── predictive_model.py
│   ├── agents/                      # LangChain Multi-Agent Framework
│   │   └── multi_agent_system.py
│   ├── api/                         # Asynchronous FastAPI Microservice
│   │   └── main.py
│   ├── dashboard/                   # Glassmorphic Streamlit Dashboard
│   │   └── app.py
│   └── simulator/                   # Distributed IoT Environmental Simulator
│       └── iot_simulator.py
├── docker-compose.yml               # Multi-container orchestration
├── requirements.txt                 # Python project dependencies
├── .env.example                     # Environment variables template
├── LICENSE                          # MIT License
└── README.md                        # Master Documentation
```

---

## ⚡ Installation & Quickstart

### Option A: Local Development Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/<USERNAME>/Agentic-AI-Air-Quality-Prediction.git
cd Agentic-AI-Air-Quality-Prediction
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Backend API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 5. Launch the Streamlit Dashboard (In a separate terminal)
```bash
streamlit run src/dashboard/app.py
```

#### 6. Start the IoT Telemetry Simulator (In a separate terminal)
```bash
python src/simulator/iot_simulator.py
```

---

### Option B: One-Command Docker Compose

Spin up the entire microservice ecosystem (API, Dashboard, Simulator) with Docker Compose:

```bash
docker-compose up --build
```
* Access the Dashboard at: `http://localhost:8501`
* Access the FastAPI Swagger Documentation at: `http://localhost:8000/docs`

---

## 📡 API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/sensor_data` | Ingests IoT telemetry packet with sub-10ms response time |
| `GET` | `/api/dashboard_data` | Fetches latest state, predicted AQI, category, and active advisories |
| `POST` | `/api/predict` | Synchronous direct prediction on custom feature payload |
| `POST` | `/api/chat` | Interacts with the multi-agent conversational reporting assistant |
| `GET` | `/health` | Health check probe for container orchestrators and IoT nodes |

---

## 📚 Citation (BibTeX)

If you find this research paper or codebase helpful in your work, please cite:

```bibtex
@inproceedings{suri2026agentic,
  title={Agentic AI for Real-Time Air Quality Prediction and Autonomous Environmental Response},
  author={Suri, Muhammad Saad Khalid Khan and Javed, Abqa and Hussain, Aatif and Suri, Abdul Ahad Khan},
  booktitle={IEEE Conference on Artificial Intelligence and Environmental Systems},
  year={2026}
}
```

---

## 📜 License
This software and research artifact are distributed under the [MIT License](LICENSE).
