import requests
import os
import tarfile
import zipfile
import argparse

from tqdm import tqdm  # Import tqdm for the loading bar

import sys
sys.path.append(os.getcwd())

from helpers.constants import ENV_CODA_ROOT_DIR

parser = argparse.ArgumentParser()
parser.add_argument("-d", '--download_parent_dir', required=True, help="Parent directory to download CODa split to")
parser.add_argument("-t", '--type', required=True,
                    help="Download by sequence (recommended for visualization) or by split (recommended for experiments) Options: ['sequence', 'split'] ")
parser.add_argument("-sp", '--split', default="tiny", 
                    help="CODa split to download. Only applies when type=split Options: ['tiny', 'small', 'medium', 'full']")
parser.add_argument("-se", '--sequence', default="0", 
                    help="CODa sequence to download. Only applies when type=sequence Options: [0 - 22]")

CHUNK_SIZE = 1024 * 1024  # 1 MB chunk size (adjust as needed)

def get_remote_file_size(url):
    try:
        response = requests.head(url, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        return int(response.headers.get("Content-Length", 0))
    except requests.exceptions.RequestException:
        return 0

def get_filename_from_content_disposition(headers):
    content_disposition = headers.get("Content-Disposition")
    if content_disposition:
        header_str = "filename*=UTF-8"
        start_index = content_disposition.find(header_str) # specific to TDR headers

        if start_index != -1:
            start_index += len(header_str)
            end_index = content_disposition.find(";", start_index)
            if end_index == -1:
                end_index = None
            filename = content_disposition[start_index:end_index]
            return filename.strip('"\'')
    return None


def download_file(url, output_path):
    try:
        # Send an HTTP GET request to the URL with streaming enabled
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes

        # Get the filename from the URL
        filename = get_filename_from_content_disposition(response.headers) or os.path.basename(url)
        file_path = os.path.join(output_path, filename)

        # Get the total file size for the loading bar
        total_size = int(response.headers.get("Content-Length", 0))
        total_size_gb = total_size/1e9

        # Prompt the user to confirm the download
        confirmation = input(f"Download %0.2f GB of files? (Y/N): "%total_size_gb).lower()
        if confirmation.lower() != "Y".lower():
            print("Download canceled.")
            return

        # Download the file in chunks and save to disk with a loading bar
        with open(file_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=filename, ncols=80
        ) as pbar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"\nFile downloaded successfully: {file_path}")

        # Untar the downloaded file if it's a tar archive]
        is_tar_compressed = file_path.lower().find(".tar.gz")!=-1
        if is_tar_compressed:
            print(f"Extracting file: {file_path}")
            with tarfile.open(file_path, "r:gz" if is_tar_compressed else "r") as tar:
                # Get the total number of members (files and directories) in the tar archive
                total_members = len(tar.getmembers())

                # Extract each member (file or directory) and update the progress bar
                with tqdm(total=total_members, unit="file", desc="Extracting", ncols=80) as pbar_extract:
                    for member in tar.getmembers():
                        tar.extract(member, output_path)
                        pbar_extract.set_description_str(f"Extracting: {member.name}")
                        pbar_extract.update(1)

                        # Set permissions for the extracted file
                        extracted_file_path = os.path.join(output_path, member.name)
                        os.chmod(extracted_file_path, 0o755)  # Set permissions to 644 (Owner: RW, Group: R, Others: R)
            
            coda_extracted_path = file_path.split('.')[0]

        is_zip_compressed = file_path.lower().find(".zip")!=-1
        if is_zip_compressed:
            print(f"Extracting file: {file_path}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                total_files = len(zip_ref.namelist())
                with tqdm(total=total_files, unit='file') as pbar:
                    for file in zip_ref.namelist():
                        zip_ref.extract(file, output_path)
                        pbar.update(1) 

            coda_extracted_path = output_path

        print(f"\nFile extracted successfully to {output_path}")
        print(f"REQUIRED: Set the environment variable {ENV_CODA_ROOT_DIR} for scripts to work correctly:")
        print(f'RUN: export {ENV_CODA_ROOT_DIR}={coda_extracted_path}')
        print("REQUIRED: Run the following command to add this to your .bashrc file too!")
        print(f'RUN: echo \'export {ENV_CODA_ROOT_DIR}={coda_extracted_path}\' >> ~/.bashrc ')

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

def main(args):
    download_dir = args.download_parent_dir
    download_type       = str(args.type)
    download_split      = str(args.split)
    download_sequence   = str(args.sequence)

    if not os.path.exists(download_dir):
        print(f'Making output dir here {download_dir}')
        os.makedirs(download_dir)

    split_to_download_url = {
        'tiny': "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/splits/CODa_tiny_split.zip",
        'small': "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/splits/CODa_sm_split.zip",
        "medium": "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/splits/CODa_md_split.zip",
        "full": "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/splits/CODa_full_split.zip"
    }

    sequence_to_download_url = {
        str(i): f'https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences/{i}.zip' 
        for i in range(23) 
    }

    assert download_type=='sequence' or download_type=='split', f'Invalid download type argument {download_type}'
    if download_type=='split':
        valid_splits = list(split_to_download_url.keys())
        assert download_split in valid_splits, f'Invalid split specified {download_split}'
        download_url = split_to_download_url[download_split]
    elif download_type=='sequence':
        valid_sequences = list(sequence_to_download_url.keys())
        assert download_sequence in valid_sequences, f'Invalid split specified {download_sequence}'
        download_url = sequence_to_download_url[download_sequence]

    print(f'Downloading dataset from url {download_url}')
    download_file(download_url, download_dir)

if __name__ == "__main__":
    # Replace the URL and output_path with your desired value
    args = parser.parse_args()
    main(args)