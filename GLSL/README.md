GLSL acts as a "wrapper" for any base GNN kernel to achieve the WSN anomaly detection task.
Our implementation of "A Novel Self-Supervised Learning-Based Anomaly Node Detection Method Based on an Autoencoder in Wireless Sensor Networks".
- At present, the test part and model part code is available now in this repo, and the code of positive and negative pair generation part and training part will be available after the manuscript is accepted.
## Getting Started
To clone the repo:
```
git clone https://github.com/GuetYe/anomaly_detection.git

cd GLSL
```
Visual analysis of point anomaly:
```
python point_anomaly_test.py
```
Visual analysis of contextual anomaly:
```
python contextual_anomaly_test.py
```
Visual analysis of periodic test:
```
python periodic_test_1.py
python periodic_test_2.py
```
More details can be found in "A Novel Self-Supervised Learning-Based Anomaly Node Detection Method Based on an Autoencoder in Wireless Sensor Networks".
Test process:
```
python test.py
```
## Dataset
We use IBRL WSN dataset for experiment. The origin data can be found in http://db.csail.mit.edu/labdata/labdata.html
The dataset used in this papaer is a part of the IBRL, which is placed in "GLSL/other/cut_IBRL".
## Requirements:
PyTorch
PyTorch Geometric
matplotlib