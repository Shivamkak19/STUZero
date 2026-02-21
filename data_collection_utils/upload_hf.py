from huggingface_hub import HfApi

api = HfApi()
repo_id = "Shivamkak/STUZero-Atari-Dynamics"

# 1. Move existing Pong files into pong_100K/ subfolder
api.upload_folder(
    folder_path="offline_training/dynamics_dataset",
    path_in_repo="pong_100K",
    repo_id=repo_id,
    repo_type="dataset",
)

# 2. Upload Asterix files into asterix_110K/ subfolder
api.upload_folder(
    folder_path="dynamics_dataset_asterix",
    path_in_repo="asterix_110K",
    repo_id=repo_id,
    repo_type="dataset",
)


# Upload Model Files //////////////////////////////
api.upload_file(                                                       
    path_or_fileobj='models/pong_model_100000.p',
    path_in_repo='pong_100K/pong_model_100000.p',
    repo_id=repo_id,
    repo_type='dataset',
)
print('Uploaded pong model')

api.upload_file(
    path_or_fileobj='models/asterix_model_110000.p',
    path_in_repo='asterix_110K/asterix_model_110000.p',
    repo_id=repo_id,
    repo_type='dataset',
)
print('Uploaded asterix model')