Datasets can be found: https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v4.0

Train model use: train.sh

meta:
- meta_info: [file_code, pidx, start_fidx, end_fidx]
- set_file: [file_code][fidx]: [pidx, pos_x, pos_y]

Networks:
1. GCN: GRU GCN IMS
1. GRIMS: GRU RNN IMS
1. GRRMS: GRU RNN RMS
1. Linear: Linear DMS
1. LRRMS: LSTM RNN RMS
1. MLP: MLP DMS
1. MPP: Mean pooling per time step, IMS
1. MPR: Mean pooling RNN, pooling once after encode, IMS
1. PNR: Pooling net, pooling once after encode, IMS
1. SA: Social Attention Pooling, IMS
1. SN: Stereoscopic network