# -*- coding: utf-8 -*-
"""
Unified UniProt Data Processor for Fish Species
Processes HDF5 embedding files for 12 fish species,
combines embeddings with enzyme classification, and provides detailed tracking statistics.
"""

import h5py
import numpy as np
import pandas as pd
import os

# =============================================
# 1. Configuration and Path Setup
# =============================================
base_data_dir = r"D:\zebfish_organized"
output_base_dir = r"D:\zebfish_processed_results"
os.makedirs(output_base_dir, exist_ok=True)

# Create subdirectories for organized output
processed_dir = os.path.join(output_base_dir, "processed_embeddings")
combined_dir = os.path.join(output_base_dir, "combined_data")
stats_dir = os.path.join(output_base_dir, "statistics")
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(combined_dir, exist_ok=True)
os.makedirs(stats_dir, exist_ok=True)

# =============================================
# 2. Helper Functions
# =============================================

def find_files(folder_path, folder_name):
    """Find HDF5 and TSV files with flexible naming"""
    print(f"  Searching for files in {folder_name}...")
    
    # List all files in the folder
    all_files = os.listdir(folder_path)
    print(f"  All files in folder: {all_files}")
    
    # Look for HDF5 files and TSV files
    hdf5_files = []
    tsv_files = []
    
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            # Check if it's a TSV file
            if file.lower().endswith('.tsv'):
                tsv_files.append(file_path)
                print(f"    Found TSV file: {file}")
            # Check if it's a gzip file (skip)
            elif file.lower().endswith('.gz'):
                continue
            # For all other files, test if they are HDF5
            else:
                if is_hdf5_file(file_path):
                    hdf5_files.append(file_path)
                    print(f"    Found HDF5 file: {file}")
                else:
                    print(f"    Skipping non-HDF5 file: {file}")
    
    print(f"  Final HDF5 files: {[os.path.basename(f) for f in hdf5_files]}")
    print(f"  Final TSV files: {[os.path.basename(f) for f in tsv_files]}")
    
    return hdf5_files, tsv_files

def is_hdf5_file(file_path):
    """Check if file is actually an HDF5 file"""
    try:
        with h5py.File(file_path, 'r') as f:
            # Try to read some basic info to verify it's HDF5
            keys = list(f.keys())
            if len(keys) > 0:
                print(f"    ✓ Valid HDF5: {os.path.basename(file_path)} with {len(keys)} keys")
                return True
            return False
    except Exception as e:
        return False

def process_hdf5_embeddings(hdf5_path, folder_name):
    """Process HDF5 embeddings file and return structured data"""
    print(f"  Processing HDF5 embeddings for {folder_name}...")
    
    uniprot_ids = []
    embeddings = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            protein_ids = list(f.keys())
            print(f"    Found {len(protein_ids)} protein entries")
            
            # Show first few protein IDs as samples
            for i, pid in enumerate(protein_ids[:5]):
                embedding_shape = f[pid][:].shape
                print(f"      Sample {i+1}: {pid} -> embedding shape: {embedding_shape}")
            
            for pid in protein_ids:
                uniprot_ids.append(pid)
                embeddings.append(f[pid][:])
        
        # Convert to arrays
        uniprot_ids = np.array(uniprot_ids)
        embeddings = np.vstack(embeddings).astype(np.float32)
        
        print(f"    Successfully processed {len(uniprot_ids)} embeddings")
        print(f"    Final embedding matrix shape: {embeddings.shape}")
        
        return uniprot_ids, embeddings, len(uniprot_ids)
    
    except Exception as e:
        print(f"  ERROR processing HDF5 for {folder_name}: {str(e)}")
        return None, None, 0

def classify_enzymes_from_tsv(tsv_path, folder_name):
    """Classify proteins as enzymes based on EC numbers from TSV file"""
    print(f"  Classifying enzymes for {folder_name}...")
    
    try:
        # Read TSV file
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"    TSV shape: {df.shape}")
        print(f"    Available columns: {df.columns.tolist()}")
        
        # Check for Entry column (UniProt ID)
        if 'Entry' not in df.columns:
            print(f"    ERROR: 'Entry' column not found")
            return None, 0, 0
        
        print(f"    Sample Entry IDs: {df['Entry'].head(3).tolist()}")
        
        # Find EC number column
        ec_column = None
        for col in df.columns:
            if 'EC number' in col or 'ec' in col.lower():
                ec_column = col
                break
        
        if ec_column is None:
            print(f"    WARNING: No EC column found, using 'EC number' as default")
            ec_column = 'EC number'
        
        print(f"    Using EC column: {ec_column}")
        
        # Function to determine if a protein is an enzyme
        def is_enzyme(ec_text):
            if pd.isna(ec_text):
                return "Non-enzyme"
            ec_str = str(ec_text).strip()
            # Check for EC number pattern
            if ec_str and ec_str[0].isdigit() and '.' in ec_str:
                return "Enzyme"
            return "Non-enzyme"
        
        # Function to extract EC class (first digit)
        def get_ec_class(ec_text):
            if pd.isna(ec_text):
                return 0
            ec_str = str(ec_text).strip()
            if ec_str and ec_str[0].isdigit():
                try:
                    ec_class = int(ec_str[0])
                    if 1 <= ec_class <= 7:
                        return ec_class
                except:
                    pass
            return 0
        
        # Add classification columns
        df['Enzyme_Classification'] = df[ec_column].apply(is_enzyme)
        df['EC_Class'] = df[ec_column].apply(get_ec_class)
        
        enzyme_count = (df['Enzyme_Classification'] == 'Enzyme').sum()
        non_enzyme_count = (df['Enzyme_Classification'] == 'Non-enzyme').sum()
        
        print(f"    Classification results: {enzyme_count} enzymes, {non_enzyme_count} non-enzymes")
        
        # Show EC class distribution
        if enzyme_count > 0:
            print(f"    EC Class distribution (enzymes only):")
            class_names = {
                1: 'Oxidoreductases', 2: 'Transferases', 3: 'Hydrolases',
                4: 'Lyases', 5: 'Isomerases', 6: 'Ligases', 7: 'Translocases'
            }
            for ec_class in range(1, 8):
                class_count = (df['EC_Class'] == ec_class).sum()
                if class_count > 0:
                    print(f"      Class {ec_class} ({class_names[ec_class]}): {class_count}")
        
        return df, enzyme_count, non_enzyme_count
        
    except Exception as e:
        print(f"  ERROR classifying enzymes for {folder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0, 0

def merge_embeddings_with_labels(embeddings_df, enzyme_df, folder_name):
    """Merge embeddings with enzyme classification labels"""
    print(f"  Merging embeddings with labels for {folder_name}...")
    
    try:
        # Get the ID column names
        enzyme_id_col = 'Entry'
        embeddings_id_col = 'UniProt_ID'
        
        print(f"    Enzyme ID column: {enzyme_id_col}")
        print(f"    Embeddings ID column: {embeddings_id_col}")
        
        # Create mapping from ID to label
        id_to_label = dict(zip(enzyme_df[enzyme_id_col], enzyme_df['Enzyme_Classification']))
        id_to_ec_class = dict(zip(enzyme_df[enzyme_id_col], enzyme_df['EC_Class']))
        
        # Add labels to embeddings dataframe
        embeddings_df['Enzyme_Classification'] = embeddings_df[embeddings_id_col].map(id_to_label)
        embeddings_df['EC_Class'] = embeddings_df[embeddings_id_col].map(id_to_ec_class)
        
        # Count matching results
        total_genes = len(embeddings_df)
        matched_genes = embeddings_df['Enzyme_Classification'].notna().sum()
        missing_genes = total_genes - matched_genes
        
        print(f"    Matching results: {matched_genes}/{total_genes} genes matched with labels")
        
        if missing_genes > 0:
            print(f"    WARNING: {missing_genes} genes without labels")
            # Show some unmatched genes for debugging
            unmatched = embeddings_df[embeddings_df['Enzyme_Classification'].isna()][embeddings_id_col].head(5)
            print(f"    Sample unmatched genes: {list(unmatched)}")
        
        # Fill missing labels with 'Unknown'
        embeddings_df['Enzyme_Classification'] = embeddings_df['Enzyme_Classification'].fillna('Unknown')
        embeddings_df['EC_Class'] = embeddings_df['EC_Class'].fillna(0)
        
        # Add folder name for tracking
        embeddings_df['Data_Source'] = folder_name
        
        return embeddings_df, total_genes, matched_genes, missing_genes
        
    except Exception as e:
        print(f"  ERROR merging data for {folder_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0, 0, 0

# =============================================
# 3. Main Processing Pipeline
# =============================================

def main():
    print("=" * 70)
    print("FISH PROTEOMES DATA PROCESSOR")
    print("Processing reviewed proteins from 12 fish species")
    print("=" * 70)
    
    # Statistics tracking
    statistics = {
        'Folder': [],
        'HDF5_Genes': [],
        'TSV_Genes': [],
        'Enzymes_Count': [],
        'Non_Enzymes_Count': [],
        'Matched_Genes': [],
        'Missing_Labels': [],
        'Final_Genes': []
    }
    
    all_combined_data = []
    
    # Get all folders in the data directory (excluding 'new_codes' if present)
    all_folders = [f for f in os.listdir(base_data_dir) 
                   if os.path.isdir(os.path.join(base_data_dir, f))]
    
    # Filter out non-fish folders like 'new_codes'
    fish_folders = [f for f in all_folders if f != 'new_codes']
    
    print(f"\n📁 Found {len(fish_folders)} fish folders to process: {fish_folders}")
    print("=" * 70)
    
    for folder_name in fish_folders:
        print(f"\n{'='*50}")
        print(f"Processing folder: {folder_name}")
        print('='*50)
        
        folder_path = os.path.join(base_data_dir, folder_name)
        
        # Find files
        hdf5_files, tsv_files = find_files(folder_path, folder_name)
        
        if not hdf5_files:
            print(f"  ❌ No HDF5 files found in {folder_name}")
            continue
        
        if not tsv_files:
            print(f"  ❌ No TSV files found in {folder_name}")
            continue
        
        hdf5_path = hdf5_files[0]
        tsv_path = tsv_files[0]
        
        print(f"\n  📊 Using HDF5 file: {os.path.basename(hdf5_path)}")
        print(f"  📄 Using TSV file: {os.path.basename(tsv_path)}")
        
        # Step 1: Process HDF5 embeddings
        uniprot_ids, embeddings, hdf5_gene_count = process_hdf5_embeddings(hdf5_path, folder_name)
        if uniprot_ids is None:
            continue
            
        # Step 2: Save processed embeddings for individual folder
        folder_output_dir = os.path.join(processed_dir, folder_name)
        os.makedirs(folder_output_dir, exist_ok=True)
        
        # Save embeddings as numpy arrays
        np.save(os.path.join(folder_output_dir, "uniprot_ids.npy"), uniprot_ids)
        np.save(os.path.join(folder_output_dir, "embeddings.npy"), embeddings)
        
        # Create and save embeddings DataFrame
        embeddings_df = pd.DataFrame(embeddings, index=uniprot_ids)
        embeddings_df.reset_index(inplace=True)
        embeddings_df.columns = ['UniProt_ID'] + [f'Embedding_{i}' for i in range(embeddings.shape[1])]
        
        embeddings_csv_path = os.path.join(folder_output_dir, "embeddings.csv")
        embeddings_df.to_csv(embeddings_csv_path, index=False)
        print(f"\n  💾 Saved embeddings to: {embeddings_csv_path}")
        
        # Step 3: Classify enzymes from TSV
        enzyme_df, enzyme_count, non_enzyme_count = classify_enzymes_from_tsv(tsv_path, folder_name)
        if enzyme_df is None:
            continue
        
        # Step 4: Merge embeddings with labels
        merged_df, total_genes, matched_genes, missing_genes = merge_embeddings_with_labels(
            embeddings_df, enzyme_df, folder_name
        )
        
        if merged_df is not None:
            # Save individual folder results
            individual_output_path = os.path.join(folder_output_dir, "labeled_embeddings.csv")
            merged_df.to_csv(individual_output_path, index=False)
            print(f"\n  💾 Saved labeled embeddings to: {individual_output_path}")
            
            # Add to combined data (only keep labeled proteins, not 'Unknown')
            labeled_df = merged_df[merged_df['Enzyme_Classification'] != 'Unknown']
            all_combined_data.append(labeled_df)
            
            # Update statistics
            statistics['Folder'].append(folder_name)
            statistics['HDF5_Genes'].append(hdf5_gene_count)
            statistics['TSV_Genes'].append(len(enzyme_df))
            statistics['Enzymes_Count'].append(enzyme_count)
            statistics['Non_Enzymes_Count'].append(non_enzyme_count)
            statistics['Matched_Genes'].append(matched_genes)
            statistics['Missing_Labels'].append(missing_genes)
            statistics['Final_Genes'].append(len(labeled_df))
            
            print(f"\n  ✅ Successfully processed {folder_name}:")
            print(f"     - Total proteins in HDF5: {hdf5_gene_count}")
            print(f"     - Matched with labels: {matched_genes}")
            print(f"     - Final labeled dataset: {len(labeled_df)} proteins")
            print(f"       (Enzymes: {(labeled_df['Enzyme_Classification'] == 'Enzyme').sum()}, "
                  f"Non-enzymes: {(labeled_df['Enzyme_Classification'] == 'Non-enzyme').sum()})")
        
        print("-" * 50)
    
    # =============================================
    # 4. Combine All Data and Save Results
    # =============================================
    
    if all_combined_data:
        print(f"\n{'='*70}")
        print("COMBINING DATA FROM ALL FISH SPECIES")
        print('='*70)
        
        # Combine all data
        combined_df = pd.concat(all_combined_data, ignore_index=True)
        
        # Save combined data
        combined_output_path = os.path.join(combined_dir, "all_fish_embeddings_combined.csv")
        combined_df.to_csv(combined_output_path, index=False)
        
        print(f"\n✅ Combined data saved to: {combined_output_path}")
        print(f"   Total proteins in combined dataset: {len(combined_df)}")
        
        # Save detailed statistics
        stats_df = pd.DataFrame(statistics)
        stats_output_path = os.path.join(stats_dir, "processing_statistics.csv")
        stats_df.to_csv(stats_output_path, index=False)
        
        print(f"✅ Statistics saved to: {stats_output_path}")
        
        # Print detailed summary
        print("\n" + "=" * 70)
        print("PROCESSING SUMMARY")
        print("=" * 70)
        
        for i, folder in enumerate(statistics['Folder']):
            print(f"\n📁 {folder.upper()}:")
            print(f"   HDF5 proteins:     {statistics['HDF5_Genes'][i]:>6}")
            print(f"   TSV entries:       {statistics['TSV_Genes'][i]:>6}")
            print(f"   Enzymes in TSV:    {statistics['Enzymes_Count'][i]:>6}")
            print(f"   Non-enzymes:       {statistics['Non_Enzymes_Count'][i]:>6}")
            print(f"   Matched genes:     {statistics['Matched_Genes'][i]:>6}")
            print(f"   Final labeled:     {statistics['Final_Genes'][i]:>6}")
        
        # Overall statistics
        print("\n" + "=" * 70)
        print("OVERALL TOTALS")
        print("=" * 70)
        print(f"\n  Total folders processed:     {len(statistics['Folder'])}")
        print(f"  Total proteins in HDF5:     {sum(statistics['HDF5_Genes']):>6}")
        print(f"  Total matched & labeled:    {sum(statistics['Final_Genes']):>6}")
        
        print(f"\n  Distribution by species (labeled dataset):")
        folder_distribution = combined_df['Data_Source'].value_counts()
        for folder, count in folder_distribution.items():
            percentage = (count / len(combined_df)) * 100
            print(f"    {folder:15s}: {count:>5} proteins ({percentage:.1f}%)")
        
        # Filter out 'Unknown' for enzyme statistics
        labeled_only = combined_df[combined_df['Enzyme_Classification'] != 'Unknown']
        
        print(f"\n  Enzyme classification distribution (labeled only):")
        enzyme_distribution = labeled_only['Enzyme_Classification'].value_counts()
        for classification, count in enzyme_distribution.items():
            percentage = (count / len(labeled_only)) * 100
            print(f"    {classification:15s}: {count:>5} ({percentage:.1f}%)")
        
        # EC class distribution
        enzymes_only = labeled_only[labeled_only['Enzyme_Classification'] == 'Enzyme']
        if len(enzymes_only) > 0:
            print(f"\n  EC Class distribution (enzymes only, n={len(enzymes_only)}):")
            class_names = {
                1: 'Oxidoreductases', 2: 'Transferases', 3: 'Hydrolases',
                4: 'Lyases', 5: 'Isomerases', 6: 'Ligases', 7: 'Translocases'
            }
            for ec_class in range(1, 8):
                class_count = (enzymes_only['EC_Class'] == ec_class).sum()
                if class_count > 0:
                    percentage = (class_count / len(enzymes_only)) * 100
                    print(f"    Class {ec_class} ({class_names[ec_class]:15s}): {class_count:>4} ({percentage:.1f}%)")
        
        # Save a clean dataset for deep learning (only Enzyme/Non-enzyme, no Unknown)
        clean_output_path = os.path.join(combined_dir, "clean_fish_dataset_for_dl.csv")
        labeled_only.to_csv(clean_output_path, index=False)
        print(f"\n✅ Clean dataset for deep learning saved to: {clean_output_path}")
        print(f"   Shape: {labeled_only.shape}")
        print(f"   Columns: {labeled_only.columns.tolist()[:5]}... (plus embedding columns)")
        
    else:
        print("\n❌ No data was successfully processed!")
    
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"All results saved in: {output_base_dir}")
    print('='*70)

# =============================================
# 5. Run the Main Function
# =============================================

if __name__ == "__main__":
    main()