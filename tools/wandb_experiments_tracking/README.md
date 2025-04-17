# ML Concepts
Repo for learning tools and ideas of ML/AI

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![CI](https://github.com/MKaczkow/ml_concepts/actions/workflows/ci.yml/badge.svg)](https://github.com/MKaczkow/ml_concepts/actions/workflows/ci.yml)

---

## W&B vs MLflow vs Hydra

| Feature                | Hydra                         | MLflow                        | Weights & Biases (WandB)     |
|------------------------|-------------------------------|-------------------------------|------------------------------|
| **Primary Focus** | Configuration Management      | End-to-end ML Lifecycle      | Experiment Tracking & Vis.    |
| **Configuration** | Hierarchical YAML, Composition| Part of Tracking (parameters) | Logged as parameters         |
| **Experiment Tracking**| Limited built-in            | Comprehensive                 | Comprehensive, Real-time     |
| **Metrics Tracking** | Relies on integration        | Built-in                      | Built-in, Rich Visualization |
| **Artifact Tracking** | Relies on integration        | Built-in                      | Built-in, Versioning         |
| **Visualization** | Limited built-in            | Basic UI for comparison     | Interactive, Rich Dashboards |
| **Hyperparameter Sweep**| Built-in                      | Integrated (via plugins)      | Advanced, Built-in           |
| **Model Management** | Not a primary focus          | Built-in Registry             | Part of Artifact Management  |
| **Deployment** | Not a primary focus          | Integrated features         | Less focused                |
| **UI** | Typically CLI-centric        | Web-based UI                  | Web-based UI                 |
| **Collaboration** | Less direct                  | Through the UI                | Strong collaboration features|
| **Ease of Setup** | Relatively easy              | Moderate                      | Relatively easy (requires account) |
| **Open Source** | Yes                           | Yes                           | Partially (core library)     |