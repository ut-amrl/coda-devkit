import requests
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--download_url', default="https://dataverse.tdl.org/api/access/datafile/288159",
                    help="Download url for dataset files, defaults to CODa_tiny")
parser.add_argument('--output_dir', default="CODa",
                    help="Download url for dataset files, defaults to CODa_tiny")
parser.add_argument('--api_token', default="DEFAULT_API_TOKEN",
                    help="Paste API Token used to download file")

CHUNK_SIZE = 1024 * 1024  # 1 MB chunk size (adjust as needed)

def get_remote_file_size(url):
    try:
        response = requests.head(url)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        return int(response.headers.get("Content-Length", 0))
    except requests.exceptions.RequestException:
        return 0

def download_file(url, output_path):
    try:
        # Get the filename from the URL
        filename = os.path.basename(url)
        # Construct the output file path
        file_path = os.path.join(output_path, filename)

        # Check if the file already exists and if it's complete
        remote_file_size = get_remote_file_size(url)
        if os.path.exists(file_path) and os.path.getsize(file_path) == remote_file_size:
            print("File already exists and matches the size on the server. Skipping download.")
            return

        # Send an HTTP GET request to the URL with streaming enabled
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes

        # Download the file in chunks and save to disk
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)

        print(f"File downloaded successfully: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

def main(args):
    download_url = args.download_url
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        print(f'Making output dir here {output_dir}')
        os.makedirs(output_dir)
    
    print(f'Downloading dataset from url {download_url}')
    download_file(download_url, output_dir)

if __name__ == "__main__":
    # Replace the URL and output_path with your desired value
    args = parser.parse_args()
    main(args)