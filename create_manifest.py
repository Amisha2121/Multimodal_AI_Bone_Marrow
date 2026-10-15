"""
Create TCIA manifest file for NBIA Data Retriever

This script creates a manifest file that NBIA Data Retriever can open
to download the Bone Marrow Cytomorphology dataset.
"""

import requests
import json

# TCIA API endpoint (public access)
BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer"

def get_series_list():
    """Get list of all series in the collection (no auth needed for this collection)."""
    url = f"{BASE_URL}/getSeries"
    params = {
        "Collection": COLLECTION,
        "format": "json"
    }
    
    print(f"Fetching series list from TCIA...")
    print(f"Collection: {COLLECTION}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        series_list = response.json()
        
        print(f"✓ Found {len(series_list)} series")
        return series_list
    except Exception as e:
        print(f"✗ Error: {e}")
        return []

def create_manifest(series_list, output_file="bone_marrow_manifest.tcia"):
    """Create a .tcia manifest file for NBIA Data Retriever."""
    
    # TCIA manifest format is a simple text file with specific structure
    manifest_lines = []
    
    # Header
    manifest_lines.append("ListOfSeriesToDownload=")
    
    # Add each series UID
    for series in series_list:
        series_uid = series.get('SeriesInstanceUID', '')
        if series_uid:
            manifest_lines.append(series_uid)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(manifest_lines))
    
    print(f"\n✓ Manifest file created: {output_file}")
    print(f"✓ Contains {len(series_list)} series")
    print(f"\nNext steps:")
    print(f"1. Open NBIA Data Retriever")
    print(f"2. Click the + button")
    print(f"3. Select this file: {output_file}")
    print(f"4. Choose download location: data/raw")
    print(f"5. Start download")

def main():
    print("="*70)
    print("TCIA Bone Marrow Dataset - Manifest Creator")
    print("="*70)
    print()
    
    # Get series list from TCIA
    series_list = get_series_list()
    
    if not series_list:
        print("\n✗ Failed to fetch series list.")
        print("Possible reasons:")
        print("  - No internet connection")
        print("  - TCIA API is down")
        print("  - Collection name has changed")
        return
    
    # Create manifest file
    create_manifest(series_list)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)

if __name__ == "__main__":
    main()
