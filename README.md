# Abstract
This study addresses the crucial challenge of monitoring the State of Health (SOH) of Lithium- Ion Batteries (LIBs) in response to the escalating demand for renewable energy systems and the imperative to reduce CO2 emissions. The research introduces deep learning (DL) models, namely Encoder-Long Short-Term Memory (E-LSTM) and Convolutional Neural Network-LSTM (CNN-LSTM), each designed to forecast battery SOH. E-LSTM integrates an encoder for dimensionality reduction and a LSTM model to capture data dependencies. CNN-LSTM, on the other hand, employs CNN layers for encoding followed by LSTM layers for precise SOH estimation. Significantly, we prioritize model explanability by employing a game-theoretic approach known as SHapley Additive exPlanations (Shap) to elucidate the output of our models. Furthermore, a method based on pattern mining was developed, synergizing with the model, to identify patterns contributing to abnormal SOH decrease. These insights are presented through informative plots. The proposed approach relies on the battery dataset from the Massachusetts Institute of Technology (MIT) and showcases promising results in accurately predicting SOH values, in which the E-LSTM model outperformed the CNN-LSTM model with a Mean Absolute Error (MAE) of less than 1%.

## Introduction
This repo contains code for the paper: Data-Driven Strategy for State of Health Prediction and Anomaly Detection in Lithium-Ion
Batteries
```latex
@article{slimane2024,
  title={Data-Driven Strategy for State of Health Prediction and Anomaly Detection in Lithium-Ion
Batteries},
  author={Slimane Arbaoui, Ahmed Samet, Ali Ayadi, Tedjani Mesbahi, Romuald Boné},
  journal={Energy and AI},
  year={2024}
}
```
### Requirements

* python>=3.10.10
* tensorflow==2.11.1
* keras==2.11.0
* h5py==3.7.0


## How to Use
#### 1. Load the MIT Battery Dataset
- Download the following three files from the provided link [MIT](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204):
     + '2017-05-12_batchdata_updated_struct_errorcorrect.mat'
     + '2017-06-30_batchdata_updated_struct_errorcorrect.mat'
     + '2018-04-12_batchdata_updated_struct_errorcorrect.mat'
     
     
- Execute the 'loading_data_MIT.ipynb' notebook, ensuring that the 'path_to_file' variable is set to the repository containing the downloaded files.

#### 2. Prepare the Data
Execute the 'data_preparing_MIT.ipynb' notebook to create an encoder for your data. This step is crucial for data preprocessing and dimensionality reduction.

#### 3. Generate Models
Execute the 'model_generating_MIT.ipynb' notebook to train the E-LSTM and CNN-LSTM models. After completing this step, run the 'combined_model_MIT.ipynb' notebook to integrate the LSTM model with the encoder.

#### 4. SHap Values Calculation
To gain insights into model predictions and enhance interpretability, execute the 'SHap_explaining.ipynb' notebook to calculate SHapley Additive exPlanations (Shap) values.

#### 5. Abnormal SOH Drop Detection
In this phase, we use the E-LSTM model to detect abnormal decreases in SOH. Follow the instructions in the 'Abnormal_SOH_Detection.ipynb' notebook. This notebook will leverage pattern mining techniques to identify and visualize patterns contributing to abnormal SOH deterioration.
## Results
Table 1: E-LSTM performance results, with MAE, MSE, RMSE, and MAPE.
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th rowspan="1"></th>
      <th colspan="2">(10,10)</th>
      <th colspan="2">(25,25)</th>
      <th colspan="2">(25,50)</th>
    </tr>
    <tr style="text-align: center;">
      <th>Metric</th>
      <th>Mean</th>
      <th>Std</th>
      <th>Mean</th>
      <th>Std</th>
      <th>Mean</th>
      <th>Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MAE (10^{-2})</th>
      <td>0.86</td>
      <td>0.06</td>
      <td>0.83</td>
      <td>0.07</td>
      <td>0.89</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>MSE (10^{-3})</th>
      <td>0.17</td>
      <td>0.02</td>
      <td>0.16</td>
      <td>0.02</td>
      <td>0.19</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>RMSE (10^{-2})</th>
      <td>1.30</td>
      <td>0.44</td>
      <td>1.26</td>
      <td>0.44</td>
      <td>1.37</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>MAPE</th>
      <td>0.91</td>
      <td>0.06</td>
      <td>0.89</td>
      <td>0.07</td>
      <td>0.95</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
Table 2: CNN-LSTM performance results, with MAE, MSE, RMSE, and MAPE.
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th rowspan="1"></th>
      <th colspan="2">(10,10)</th>
      <th colspan="2">(25,25)</th>
      <th colspan="2">(25,50)</th>
    </tr>
    <tr style="text-align: center;">
      <th>Metric</th>
      <th>Mean</th>
      <th>Std</th>
      <th>Mean</th>
      <th>Std</th>
      <th>Mean</th>
      <th>Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MAE (10^{-2})</th>
      <td>0.90</td>
      <td>0.08</td>
      <td>1.11</td>
      <td>0.23</td>
      <td>1.19</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>MSE (10^{-3})</th>
      <td>0.22</td>
      <td>0.03</td>
      <td>0.35</td>
      <td>0.12</td>
      <td>0.35</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>RMSE 10^{-2})</th>
      <td>1.40</td>
      <td>0.54</td>
      <td>1.87</td>
      <td>1.07</td>
      <td>1.87</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>MAPE</th>
      <td>0.96</td>
      <td>0.09</td>
      <td>1.25</td>
      <td>0.24</td>
      <td>1.27</td>
      <td>0.20</td>
    </tr>
  </tbody>
</table>
Figure 1: SOH prediction using 10-step input window for 10-
step output window.
<br>
<img  
  src="https://github.com/arslimane/Energitic-project-1/assets/60701015/4c23b33b-d892-4d87-b01d-f166da32862c" 
  alt="Alt text" 
  title="Optional title" 
  style="width: 400px"
  >
  <br>
  Figure 2: The average contribution of each feature in the
model’s prediction using E-LSTM.
<br>
Figure 1: SOH prediction using 10-step input window for 10-
step output window.
<br>
<img  
  src="https://github.com/arslimane/Energitic-project-1/assets/60701015/4c23b33b-d892-4d87-b01d-f166da32862c" 
  alt="Alt text" 
  title="Optional title" 
  style="width: 400px"
  >
  <br>
  Figure 2: The average contribution of each feature in the
model’s prediction using E-LSTM.
<br>
<img  
  src="https://github.com/arslimane/Energitic-project-1/assets/60701015/5a21e626-3fa8-4554-9b00-95833fb32da5" 
  alt="Alt text" 
  title="Optional title" 
  style="width: 400px"
  >




  




## Contact
If you have any issues or questions about this repo, feel free to contact slimane.arbaoui@insa-strasbourg.fr.



