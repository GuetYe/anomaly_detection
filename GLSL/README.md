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
We use IBRL WSN dataset for experiment. The origin data can be found in http://db.csail.mit.edu/labdata/labdata.html.

The dataset used in this papaer is a part of the IBRL, which is placed in "GLSL/other/cut_IBRL".
## Requirements:
PyTorch
PyTorch Geometric
matplotlib

## GLSL+
The sizes of WSN datasets in the real world are very large, and with the increase in the number of layers in GNNs, the computational cost and required memory space increase exponentially, so the computational and storage complexities encountered when training large-scale GNNs are very high. To cope with the high time consumption caused by the massive numbers of sensor nodes and recording moments in large-scale WSN scenarios, this paper proposes an expansion strategy called GLSL+ based on K-means and piecewise aggregate approximation (PAA).

cd GLSL+

There are four control parameters in GLSL+/config.py
```
    parser.add_argument('--use_kmeans_paa', default=True, help="if True means that GLSL+ is in effect")
    parser.add_argument('--cluster_num', default=3, help="The number of clusters that the entire node set is divided into")
    parser.add_argument('--use_cluster_block', default=0, help="The cluster number that we consider now")
    parser.add_argument('--paa_out_dimension', default=2500, help="paa_out_dimension")
```

Visual analysis of anomaly:
```
python anomaly_visial.py
```
training part will be available after the manuscript is accepted.

