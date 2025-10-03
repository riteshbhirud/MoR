import os
import shutil
from huggingface_hub import hf_hub_download, list_repo_files


def download_hf_file(repo, file, repo_type="dataset", save_as_file=None):
    """
    Downloads a file from a Hugging Face repository and saves it to the specified path.

    Args:
        repo (str): The repository name.
        file (str): The file path within the repository to download.
        repo_type (str): The type of the repository (e.g., 'dataset').
        save_as_file (str, optional): The local file path to save the downloaded file. 
                                      If not provided, saves the file in the current directory 
                                      with the same name as the original file.
    """
    # Download the file from the repository
    file_path = hf_hub_download(repo, file, repo_type=repo_type)
    
    # Determine the save path
    if save_as_file is None:
        return file_path
    
    # Create necessary directories
    os.makedirs(os.path.dirname(save_as_file), exist_ok=True)
    
    # Copy the downloaded file to the desired location
    if not os.path.exists(save_as_file) and file_path != save_as_file:
        shutil.copy2(file_path, save_as_file)
    
    print(f"Downloaded <file:{file}> from <repo:{repo}> to <path:{save_as_file}>!")
    return save_as_file

def download_hf_folder(repo, folder, repo_type="dataset", save_as_folder=None):
    """
    Downloads a folder from a Hugging Face repository and saves it to the specified directory.

    Args:
        repo (str): The repository name.
        folder (str): The folder path within the repository to download.
        repo_type (str): The type of the repository (e.g., 'dataset').
        save_as_folder (str, optional): The local directory to save the downloaded folder. 
                                        Defaults to "data/".
    """
    from huggingface_hub import snapshot_download
    import os
    
    # Download the entire repo to cache (or use existing cache)
    cache_dir = snapshot_download(repo, repo_type=repo_type, allow_patterns=f"{folder}/*")
    
    # The folder we want is inside the cache
    result_folder = os.path.join(cache_dir, folder)
    
    if save_as_folder is not None:
        # Copy from cache to specified folder
        import shutil
        os.makedirs(save_as_folder, exist_ok=True)
        if os.path.exists(result_folder):
            for item in os.listdir(result_folder):
                src = os.path.join(result_folder, item)
                dst = os.path.join(save_as_folder, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            result_folder = save_as_folder
    
    if os.path.exists(result_folder):
        print(f"Using folder at {result_folder}")
        return result_folder
    else:
        print(f"WARNING: Folder not found: {result_folder}")
        return None
