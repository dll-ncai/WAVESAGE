# WAVESAGE  
**WAVELET AND SHAP EMPOWERED EXPLAINABLE AI FOR EEG MICRO-EVENT LOCALISATION**

This repository contains code for **WAVSAGE** and various **Explainable AI (XAI) methods** applied to 2-second EEG windows.

## Folder Structure

- **WAVSAGE/** – Contains the code for the WAVSAGE framework.  
- **ExplainableAI/** – Contains code for Grad-CAM, Integrated Gradients, Occlusion, LIME, SmoothGrad, DDT, and RISE.  
  Each method has scripts for:
  - **Single window processing**  
  - **Batch processing** with average metrics  

## Usage

Run the respective single or batch processing script for the method of your choice.  

## Outputs

- Visualizations of EEG signals and importance maps in case of single window process.  
- CSV files containing metrics (Coverage, Precision, IoU) for batch processing.  

## Requirements

- Python 3.8+  
- TensorFlow 2.x  
- Keras  
- NumPy  
- Matplotlib  
- scikit-learn  
- joblib  
- pandas
