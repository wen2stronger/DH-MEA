import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.abspath(os.path.join(current_dir, '..', 'datasets'))
model_att_root = os.path.abspath(os.path.join(current_dir, '..', 'model_att'))