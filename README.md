# Bangkok Air Quality Forecasting and Monitoring

## Introduction

This project develops a machine learning system to forecast PM2.5 concentrations in Bangkok using historical air quality and weather data. Air pollution, particularly PM2.5, has significant impacts on public health and the environment. By predicting PM2.5 levels, this system helps authorities and individuals take proactive measures to reduce exposure and mitigate health risks.

The forecasting system is built as an end-to-end MLOps pipeline that covers:
- Automated data collection from Open-Meteo API
- Data preprocessing and feature engineering
- Model training and evaluation
- Continuous deployment and monitoring
- Interactive web dashboard for visualization

Through accurate forecasting of air quality, this project aims to contribute to better public health outcomes and environmental management in Bangkok.

## Project Setup

### Prerequisites

- Python 3.10+
- Git
- Docker and Docker Compose
- GitHub account
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bqa.git
cd bqa
```

2. Set up the development environment using UV:
```bash
uv sync
```


## Resources

- [Open-Meteo API Documentation](https://open-meteo.com/en/docs)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
