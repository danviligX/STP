# Nerual Networks for Social Trajectory Prediction

Datasets can be found: https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v4.0

Train model use: train.sh

meta:
- meta_info: [file_code, pidx, start_fidx, end_fidx]
- set_file: [file_code][fidx]: [pidx, pos_x, pos_y]

Models:
1. GRSMS: GRU Seq2Seq
1. GRRMS: GRU Recurrent
1. Linear: Linear DMS
1. LRRMS: LSTM Recurrent
1. MLP: MLP DMS
1. SLSTM: Social LSTM
1. MPR: Mean pooling RNN, pooling once after encode
1. PNR: Pooling net, pooling once after encode, IMS
1. TDGAT: Track Decompose Graph Attention Network