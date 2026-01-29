import os
import logging
from google.cloud import storage

logger = logging.getLogger(__name__)

def download_file_from_gcs(gcs_uri, local_path):
    """Downloads a file from GCS to a local path using the SDK."""
    try:
        storage_client = storage.Client()
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        
        path_parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        logger.info(f"Downloading {gcs_uri} to {local_path}...")
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        logger.error(f"Failed to download from GCS: {e}")
        raise e

def upload_file_to_gcs(local_path, gcs_uri):
    """Uploads a local file to GCS using the SDK."""
    try:
        storage_client = storage.Client()
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
            
        path_parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        logger.info(f"Uploading {local_path} to {gcs_uri}...")
        blob.upload_from_filename(local_path)
        return True
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        raise e

def list_gcs_files(gcs_path, extensions=None):
    """Lists files in a GCS path with optional extension filtering."""
    try:
        storage_client = storage.Client()
        path = gcs_path.replace("gs://", "")
        parts = path.split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        files = []
        for blob in blobs:
            name = blob.name
            if extensions:
                if any(name.lower().endswith(ext) for ext in extensions):
                    files.append(os.path.basename(name))
            else:
                if name != prefix: # Exclude the directory itself if it's an object
                    files.append(os.path.basename(name))
        return sorted(list(set(files))) # Remove duplicates and sort
    except Exception as e:
        logger.error(f"Failed to list GCS files: {e}")
        return []

def download_bytes_from_gcs(gcs_uri):
    """Downloads a file from GCS and returns as bytes."""
    try:
        storage_client = storage.Client()
        path = gcs_uri.replace("gs://", "")
        bucket_name, blob_name = path.split("/", 1)
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    except Exception as e:
        logger.error(f"Failed to download bytes from GCS: {e}")
        raise e
