# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Google Photos API integration for uploading images."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import requests

logger = logging.getLogger(__name__)

# Scopes required for uploading to Google Photos
SCOPES = [
    "https://www.googleapis.com/auth/photoslibrary.appendonly",
    "https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata",
]

# Default paths for credentials and tokens
DEFAULT_TOKEN_PATH = Path.home() / ".muimg" / "google-photos-token.json"
DEFAULT_CREDENTIALS_PATH = (
    Path.home() / ".muimg" / "google-photos-credentials.json"
)


class GooglePhotosClient:
    """Client for interacting with Google Photos API."""

    def __init__(
        self,
        token_path: Optional[Path] = None,
        credentials_path: Optional[Path] = None,
    ):
        """Initialize Google Photos client.

        Args:
            token_path: Path to stored OAuth2 token (refresh token)
            credentials_path: Path to OAuth2 client credentials from Google Cloud Console
        """
        self.token_path = token_path or DEFAULT_TOKEN_PATH
        self.credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
        self.creds = None

    def authenticate(self, force_reauth: bool = False) -> bool:
        """Authenticate with Google Photos API.

        Args:
            force_reauth: If True, force re-authentication even if token exists

        Returns:
            True if authentication successful, False otherwise
        """
        # Load existing token if available
        if not force_reauth and self.token_path.exists():
            logger.info(f"Loading existing token from {self.token_path}")
            self.creds = Credentials.from_authorized_user_file(
                str(self.token_path), SCOPES
            )

        # Refresh or re-authenticate if needed
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                logger.info("Refreshing expired token")
                self.creds.refresh(Request())
            else:
                # Need to do interactive OAuth flow
                if not self.credentials_path.exists():
                    logger.error(
                        f"OAuth2 credentials not found at {self.credentials_path}"
                    )
                    logger.error(
                        "Please download OAuth2 client credentials from Google Cloud Console"
                    )
                    logger.error(
                        "and save them to the path above, or specify --credentials-path"
                    )
                    return False

                logger.info("Starting OAuth2 authentication flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), SCOPES
                )
                self.creds = flow.run_local_server(port=0)

            # Save the credentials for future use
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_path, "w") as token_file:
                token_file.write(self.creds.to_json())
            logger.info(f"Token saved to {self.token_path}")

        logger.info("Successfully authenticated with Google Photos API")
        return True

    def list_albums(self) -> list:
        """List all albums in the user's Google Photos library.

        Returns:
            List of album dictionaries with 'id', 'title', and 'productUrl' keys
        """
        if not self.creds:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        albums = []
        page_token = None
        url = "https://photoslibrary.googleapis.com/v1/albums"

        while True:
            params = {"pageSize": 50}
            if page_token:
                params["pageToken"] = page_token

            headers = {"Authorization": f"Bearer {self.creds.token}"}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            results = response.json()
            albums.extend(results.get("albums", []))
            page_token = results.get("nextPageToken")

            if not page_token:
                break

        logger.info(f"Found {len(albums)} albums")
        return albums

    def get_or_create_album(self, album_title: str) -> str:
        """Get album ID by title, or create it if it doesn't exist.

        Args:
            album_title: Title of the album

        Returns:
            Album ID
        """
        if not self.creds:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        # Check if album already exists
        albums = self.list_albums()
        for album in albums:
            if album.get("title") == album_title:
                logger.info(f"Found existing album: {album_title}")
                return album["id"]

        # Create new album
        logger.info(f"Creating new album: {album_title}")
        url = "https://photoslibrary.googleapis.com/v1/albums"
        headers = {
            "Authorization": f"Bearer {self.creds.token}",
            "Content-type": "application/json"
        }
        request_body = {"album": {"title": album_title}}
        response = requests.post(url, headers=headers, json=request_body)
        response.raise_for_status()
        
        result = response.json()
        album_id = result["id"]
        logger.info(f"Created album with ID: {album_id}")
        return album_id

    def upload_image(
        self, image_path: Path, album_title: Optional[str] = None
    ) -> dict:
        """Upload an image to Google Photos.

        Args:
            image_path: Path to the image file
            album_title: Optional album title. If provided, image will be added to this album.

        Returns:
            Dictionary with upload result including 'mediaItemId' and 'productUrl'
        """
        if not self.creds:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Step 1: Upload the bytes to get an upload token
        logger.info(f"Uploading {image_path.name}...")
        upload_url = "https://photoslibrary.googleapis.com/v1/uploads"

        # Determine MIME type from file extension
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
        }
        content_type = mime_types.get(suffix, "application/octet-stream")

        headers = {
            "Authorization": f"Bearer {self.creds.token}",
            "Content-type": "application/octet-stream",
            "X-Goog-Upload-Content-Type": content_type,
            "X-Goog-Upload-File-Name": image_path.name,
            "X-Goog-Upload-Protocol": "raw",
        }

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = requests.post(upload_url, headers=headers, data=image_bytes)

        if response.status_code != 200:
            raise RuntimeError(
                f"Upload failed: {response.status_code} - {response.text}"
            )

        upload_token = response.text
        logger.info("Upload token received")

        # Step 2: Create the media item from the upload token
        create_body = {
            "newMediaItems": [
                {
                    "description": f"Uploaded by muimg",
                    "simpleMediaItem": {
                        "uploadToken": upload_token,
                        "fileName": image_path.name,
                    },
                }
            ]
        }

        # Add to album if specified
        if album_title:
            album_id = self.get_or_create_album(album_title)
            create_body["albumId"] = album_id

        # Create media item
        create_url = "https://photoslibrary.googleapis.com/v1/mediaItems:batchCreate"
        create_headers = {
            "Authorization": f"Bearer {self.creds.token}",
            "Content-type": "application/json"
        }
        create_response_obj = requests.post(create_url, headers=create_headers, json=create_body)
        
        if create_response_obj.status_code != 200:
            logger.error(f"Upload failed with status {create_response_obj.status_code}")
            logger.error(f"Response: {create_response_obj.text}")
            create_response_obj.raise_for_status()
            
        create_response = create_response_obj.json()

        # Check for errors
        new_items = create_response.get("newMediaItemResults", [])
        if not new_items:
            raise RuntimeError("No media items created")

        result = new_items[0]
        if "status" in result and result["status"].get("message") != "Success":
            raise RuntimeError(f"Upload failed: {result['status']}")

        media_item = result["mediaItem"]
        logger.info(f"Successfully uploaded: {media_item['productUrl']}")

        return {
            "mediaItemId": media_item["id"],
            "productUrl": media_item["productUrl"],
            "filename": media_item["filename"],
        }
