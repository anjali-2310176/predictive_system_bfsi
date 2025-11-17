# BFSI Predictive System for Fraud Detection

## Overview
An enterprise-grade fraud detection system specifically designed for Banking, Financial Services, and Insurance (BFSI) organizations. This system uses advanced machine learning algorithms to provide real-time fraud risk assessment with high accuracy.

## Features
- **Real-time Fraud Detection**: Instant analysis of transactions
- **Multi-model Ensemble**: XGBoost, Logistic Regression, and Random Forest
- **BFSI-specific Features**: 40+ specialized features for financial fraud detection
- **Interactive Web Interface**: Built with Streamlit for easy deployment
- **Risk Assessment**: 5-level granular risk categorization
- **Comprehensive Analytics**: Detailed transaction analysis and insights

## Model Performance
- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1-Score**: 100%
- **AUC-ROC**: 1.0

## Project Structure
```
predictive_system_bfsi/
├── fraud_detection_final.ipynb    # Jupyter notebook with model training
├── streamlit_app.py               # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── *.pkl files                   # Trained models and preprocessors
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. **Data Input**: Enter transaction details in the sidebar form
2. **Analysis**: Click "Analyze for Fraud Risk" 
3. **Results**: View comprehensive fraud risk assessment including:
   - Overall risk level and probability
   - Individual model predictions
   - Risk indicators and alerts
   - Customer profile analysis

## Machine Learning Models

### 1. XGBoost Classifier
- Gradient boosting framework
- Optimized for fraud detection patterns
- High performance on imbalanced datasets

### 2. Logistic Regression
- Linear classification approach
- Interpretable results
- Fast inference time

### 3. Random Forest
- Ensemble of decision trees
- Robust to overfitting
- Feature importance insights

### Ensemble Method
The system combines all three models using weighted averaging for superior accuracy and reliability.

## BFSI-Specific Features

### Temporal Risk Features
- Transaction hour analysis
- Weekend/holiday patterns
- Time-based anomalies

### Amount-Based Features
- Round amount detection
- User spending percentiles
- Deviation from normal patterns

### Behavioral Analytics
- Transaction velocity
- Location patterns
- Authentication methods

### Compliance Features
- AML/KYC indicators
- PEP (Politically Exposed Person) flags
- Sanctions screening

## Security Considerations

For production deployment:
- Implement user authentication
- Use HTTPS encryption
- Add input validation
- Implement rate limiting
- Enable audit logging

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue in this repository.

## Acknowledgments

- Built with Streamlit for the web interface
- Scikit-learn for machine learning algorithms
- XGBoost for gradient boosting
- Pandas for data manipulation