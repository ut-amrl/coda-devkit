import requests
import os
import tarfile
import argparse

from tqdm import tqdm  # Import tqdm for the loading bar

parser = argparse.ArgumentParser()
parser.add_argument('--download_url', default="https://dataverse.tdl.org/api/access/datafile/288159",
                    help="Download url for dataset files, defaults to CODa_tiny")
parser.add_argument('--api_token', default="DEFAULT_API_TOKEN",
                    help="Paste API Token used to download file")

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
        is_compressed = file_path.lower().find(".tar.gz")!=-1
        if is_compressed:
            print(f"Extracting file: {file_path}")
            with tarfile.open(file_path, "r:gz" if is_compressed else "r") as tar:
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

            print("Setting CODA_INDIR to be extracted location %s"%file_path)
            os.environ["CODA_INDIR"] = file_path
            print(f"\nFile extracted successfully: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

def main(args):
    download_url = args.download_url

    download_dir = os.environ.get("CODA_DIR", "")
    assert download_dir!="", 'Download directory {download_dir} environment variable is empty, exiting...'

    if not os.path.exists(download_dir):
        print(f'Making output dir here {download_dir}')
        os.makedirs(download_dir)
    
    print(f'Downloading dataset from url {download_url}')
    download_file(download_url, download_dir)

if __name__ == "__main__":
    # Replace the URL and output_path with your desired value
    args = parser.parse_args()
    main(args)