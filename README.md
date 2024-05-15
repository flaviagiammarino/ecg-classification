# ECG classification on the PTB-XL dataset

PyTorch implementation of the FCN model [1] (<i>fcn_wang</i>) for ECG classification 
proposed in [2] and corresponding LSTM-FCN [3] extension. Both models are trained and
evaluated on the PTB-XL dataset [4].

## Usage
1. Download the data from PhysioNet: https://doi.org/10.13026/x4td-x982
2. Install the requirements `pip install -r requirements.txt`.
3. Run `notebook.ipynb`.

## References
[1] Zhiguang Wang, Weizhong Yan, and Tim Oates. “Time series classification from scratch with deep neural networks: A strong baseline”. In: 2017 International Joint Conference on Neural Networks (IJCNN). 2017, pp. 1578–1585. https://doi.org/10.1109/IJCNN.2017.7966039.

[2] Nils Strodthoff et al. “Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL”. In: IEEE Journal of Biomedical and Health Informatics 25.5 (2021),
pp. 1519–1528. https://doi.org/10.1109/JBHI.2020.3022989.

[3] Fazle Karim et al. “LSTM Fully Convolutional Networks for Time Series Classification”. In: IEEE Access 6 (2018),
pp. 1662–1669. https://doi.org/10.1109/ACCESS.2017.2779939.

[4] Patrick Wagner et al. “PTB-XL, a large publicly available electrocardiography dataset”. In: Scientific Data 7.1 (2020), p. 154. https://doi.org/10.1038/s41597-020-0495-6.
