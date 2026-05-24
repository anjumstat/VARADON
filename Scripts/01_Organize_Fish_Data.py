# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:31:01 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
Organize Fish HDF5 and TSV Files for Unified Processor
These files are already HDF5 format - just need proper organization
"""

import os
import shutil
import h5py
import pandas as pd

# =============================================
# Configuration
# =============================================
base_dir = r"D:\zebfish"
output_base_dir = r"D:\zebfish_organized"

os.makedirs(output_base_dir, exist_ok=True)

def verify_hdf5_and_rename(file_path, new_path):
    """Verify file is HDF5 and copy with .h5 extension"""
    try:
        with h5py.File(file_path, 'r') as hf:
            num_keys = len(hf.keys())
            print(f"    ✓ Valid HDF5 with {num_keys} entries")
            
            # Show first few keys
            keys = list(hf.keys())[:3]
            for key in keys:
                shape = hf[key].shape
                print(f"      Sample: {key} -> shape {shape}")
            
            # Copy to new location with .h5 extension
            shutil.copy2(file_path, new_path)
            print(f"    ✅ Copied to: {os.path.basename(new_path)}")
            return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

def process_folder(folder_path, folder_name):
    """Process each folder - copy HDF5 and TSV to organized structure"""
    
    print(f"\n📂 Processing: {folder_name}")
    print("-" * 50)
    
    # Create output folder for this species
    output_folder = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Find the HDF5 file (no extension file with HDF5 magic number)
    hdf5_file = None
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and '.' not in file:
            # Check if it's HDF5 by reading magic number
            try:
                with open(file_path, 'rb') as f:
                    magic = f.read(8)
                    if magic[:4] == b'\x89HDF':
                        hdf5_file = file_path
                        print(f"  📊 Found HDF5 file: {file}")
                        break
            except:
                continue
    
    if hdf5_file is None:
        print(f"  ❌ No HDF5 file found")
        return False
    
    # Find TSV file
    tsv_file = None
    for file in os.listdir(folder_path):
        if file.endswith('.tsv'):
            tsv_file = os.path.join(folder_path, file)
            print(f"  📄 Found TSV file: {file}")
            break
    
    if tsv_file is None:
        print(f"  ⚠️ No TSV file found, but continuing with HDF5 only")
    
    # Copy HDF5 with .h5 extension
    h5_output = os.path.join(output_folder, "embeddings.h5")
    if verify_hdf5_and_rename(hdf5_file, h5_output):
        # Copy TSV if exists
        if tsv_file:
            tsv_output = os.path.join(output_folder, f"{folder_name}_annotations.tsv")
            shutil.copy2(tsv_file, tsv_output)
            print(f"  ✅ Copied TSV to: {os.path.basename(tsv_output)}")
        
        # Also save a metadata file about the dataset
        with h5py.File(h5_output, 'r') as hf:
            keys = list(hf.keys())
            embedding_dim = hf[keys[0]].shape[0] if keys else 0
            
            metadata = {
                'Species': folder_name,
                'Num_Proteins': len(keys),
                'Embedding_Dim': embedding_dim,
                'HDF5_Path': h5_output,
                'TSV_Path': tsv_output if tsv_file else 'None'
            }
            
            # Count enzymes if TSV available
            if tsv_file:
                df_tsv = pd.read_csv(tsv_file, sep='\t')
                if 'EC number' in df_tsv.columns:
                    enzyme_count = df_tsv['EC number'].notna().sum()
                    metadata['Enzymes_in_TSV'] = enzyme_count
                    metadata['Non_Enzymes_in_TSV'] = len(df_tsv) - enzyme_count
                
                # Check intersection
                ids_in_h5 = set(keys)
                ids_in_tsv = set(df_tsv['Entry'].values)
                intersection = ids_in_h5 & ids_in_tsv
                metadata['Common_Proteins'] = len(intersection)
            
            # Save metadata
            metadata_df = pd.DataFrame([metadata])
            metadata_path = os.path.join(output_folder, "metadata.csv")
            metadata_df.to_csv(metadata_path, index=False)
            print(f"  📊 Saved metadata: {metadata_path}")
            
            return True
    
    return False

def create_master_summary():
    """Create a master CSV with all converted datasets"""
    
    summaries = []
    
    for folder in os.listdir(output_base_dir):
        folder_path = os.path.join(output_base_dir, folder)
        metadata_path = os.path.join(folder_path, "metadata.csv")
        
        if os.path.exists(metadata_path):
            df_meta = pd.read_csv(metadata_path)
            summaries.append(df_meta)
    
    if summaries:
        master_df = pd.concat(summaries, ignore_index=True)
        master_path = os.path.join(output_base_dir, "master_summary.csv")
        master_df.to_csv(master_path, index=False)
        print(f"\n✅ Master summary saved: {master_path}")
        return master_df
    
    return None

def main():
    print("=" * 70)
    print("ORGANIZE FISH HDF5 AND TSV FILES")
    print("=" * 70)
    
    if not os.path.exists(base_dir):
        print(f"❌ ERROR: Directory not found: {base_dir}")
        return
    
    # Get all folders (skip 'codes' and 'new_codes')
    folders = [f for f in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, f)) 
               and f not in ['codes', 'new_codes']]
    
    print(f"\n📁 Found {len(folders)} folders to process:")
    for f in folders:
        print(f"   - {f}")
    print("=" * 70)
    
    successful = []
    failed = []
    
    for folder_name in folders:
        folder_path = os.path.join(base_dir, folder_name)
        
        if process_folder(folder_path, folder_name):
            successful.append(folder_name)
        else:
            failed.append(folder_name)
    
    # Create master summary
    master_df = create_master_summary()
    
    # Final summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {len(successful)} folders")
    for f in successful:
        print(f"   - {f}")
    
    if failed:
        print(f"❌ Failed: {len(failed)} folders")
        for f in failed:
            print(f"   - {f}")
    
    if master_df is not None:
        print("\n📊 MASTER SUMMARY:")
        print(master_df.to_string(index=False))
    
    print(f"\n📁 Organized data saved to: {output_base_dir}")
    print("\n✅ To use with your original code, UPDATE the path:")
    print(f'   base_data_dir = r"{output_base_dir}"')
    print("=" * 70)

if __name__ == "__main__":
    main()