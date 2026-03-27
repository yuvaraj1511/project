# 🚦 AI Traffic Prediction & Smart Routing System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900?style=flat-square&logo=leaflet&logoColor=white)](https://leafletjs.com/)

A premium, industry-level Intelligent Transportation System (ITS) that leverages **Graph Neural Networks (GNN)** and **Gated Recurrent Units (GRU)** to predict urban traffic congestion and provide real-time smart routing for emergency, heavy, and civilian vehicles.

![Dashboard Preview](file:///C:/Users/student/.gemini/antigravity/brain/9c27eb9d-c12c-46df-b37b-7af9a37b6379/full_dashboard_1774591972861.png)

---

## 🚀 Key Features

-   **Deep Learning Prediction**: Hybrid model (GNN for spatial features, GRU for temporal patterns) trained on urban traffic telemetry.
-   **Smart Routing Engine**: Specialized algorithms for different vehicle types:
    -   🚑 **Emergency**: Prioritizes speed and avoids congestion at all costs.
    -   🚛 **Heavy Vehicles**: Prefers highways and avoids narrow urban corridors.
    -   🚗 **Civilian**: Balances distance and travel time.
-   **Interactive Dashboard**: Real-time Leaflet.js map with live congestion heatmaps and node telemetry.
-   **Secure Access**: Premium glassmorphic login system with session management.
-   **Dynamic Graph Visualization**: Real-time representation of the city road network.

## 🛠️ Technology Stack

-   **Backend**: FastAPI (Python 3.11), Uvicorn
-   **AI Framework**: PyTorch, NetworkX
-   **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism), JavaScript (ES6+)
-   **Data Viz**: Chart.js, Leaflet.js
-   **Algorithm**: Dijkstra's & A* (Congestion-weighted)

## 📦 Installation

### 1. Prerequisites
-   Python 3.11 (Highly recommended for Torch compatibility)
-   `pip` or `py launcher`

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/bala2872005-crypto/Traffic-predication.git
cd Traffic-predication

# Install dependencies
py -3.11 -m pip install -r backend/requirements.txt
```

### 3. Run the System
```bash
# Start the FastAPI server
py -3.11 -m uvicorn main:app --reload --app-dir backend
```
Access the dashboard at: `http://localhost:8000/app`

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/login` | Serves the login page |
| `POST` | `/login` | Authenticates user (admin/traffic123) |
| `GET` | `/traffic-data` | Returns live congestion telemetry |
| `POST` | `/route` | Calculates optimal paths |
| `POST` | `/predict` | Returns model inference results |

## 🏗️ Architecture

```mermaid
graph TD
    A[Frontend Dashboard] <--> B[FastAPI Backend]
    B <--> C[TrafficPredictor (GNN + GRU)]
    B <--> D[Routing Engine (NetworkX)]
    C --> E[Synthetic/Real Telemetry]
    D --> F[Road Graph Network]
```

## 📜 License
Internal Project - All Rights Reserved.

---
*Created by Antigravity AI for Smart Cities.*
