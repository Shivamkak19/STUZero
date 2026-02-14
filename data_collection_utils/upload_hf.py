from huggingface_hub import HfApi

api = HfApi()
api.create_repo('Shivamkak/STUZero-Pong-Dynamics', repo_type='dataset')
api.upload_folder(folder_path='dynamics_dataset', repo_id='Shivamkak/STUZero-Pong-Dynamics',
repo_type='dataset')