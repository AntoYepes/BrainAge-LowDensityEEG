import os
import argparse
import numpy as np
import mne
import pywt
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# ==========================================
# CONFIGURACIÓN POR DEFECTO
# ==========================================
DEFAULT_CHANNELS = ['FP1', 'FP2', 'O1', 'O2', 'C3', 'C4', 'P7', 'P8']
SPECIAL_CHANNELS = ['FP1', 'FP2', 'O1', 'O2', 'C3', 'C4', 'T5', 'T6'] # T5=P7, T6=P8
FS = 200
SEGMENT_LEN = 2400
FREQ_BAND = (1, 30)
NUM_SCALES = 30
WAVELET = 'cmor1.5-1.5'

def parse_args():
    parser = argparse.ArgumentParser(description="Convert preprocessed EEG (.fif) to Scalogram Tensors (.h5)")
    parser.add_argument('--input_dir', type=Path, required=True, help="Folder containing .fif files")
    parser.add_argument('--labels_file', type=Path, required=True, help="Path to .npy file with subject ages")
    parser.add_argument('--output_file', type=Path, default='dataset_scalograms.h5', help="Output H5 file name")
    return parser.parse_args()

def calculate_scales(frequency_band, num_scales, fs):
    fmin, fmax = frequency_band
    scales_min, scales_max = pywt.frequency2scale(WAVELET, np.array([fmin, fmax]) / fs)
    return np.linspace(scales_min, scales_max, num=num_scales)

def compute_cwt_power(signal, scales):
    coef, _ = pywt.cwt(signal, scales, WAVELET)
    return np.abs(coef) ** 2

def get_multichannel_tensor(segment, ch_indices, scales):
    """Generates 3D tensor: (Channels, Freqs, Time)"""
    scalograms = []
    for ch_idx in ch_indices:
        cwt_power = compute_cwt_power(segment[ch_idx], scales)
        # Individual channel normalization
        norm_cwt = (cwt_power - np.min(cwt_power)) / (np.max(cwt_power) - np.min(cwt_power) + 1e-8)
        scalograms.append(norm_cwt)
    return np.stack(scalograms, axis=0)

def extend_signal(data, target_length=24000):
    current_length = data.shape[1]
    if current_length >= target_length:
        return data[:, :target_length]
    repeat_factor = int(np.ceil(target_length / current_length))
    return np.tile(data, (1, repeat_factor))[:, :target_length]

def main():
    args = parse_args()
    
    # 1. Load Labels
    print(f"📂 Loading labels from: {args.labels_file}")
    if not args.labels_file.exists():
        raise FileNotFoundError("Labels file not found.")
        
    subject_ages_raw = np.load(args.labels_file, allow_pickle=True).item()
    subject_ages = {k: v for k, v in subject_ages_raw.items() if 20 <= v <= 70}
    
    # Normalize Ages
    real_ages = np.array(list(subject_ages.values())).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(real_ages)
    
    # Pre-calculate scales
    scales = calculate_scales(FREQ_BAND, NUM_SCALES, FS)
    
    scalogram_tensors = []
    ages_real_list = []
    ages_norm_list = []
    names_list = []
    
    files = list(args.input_dir.glob("*.fif"))
    print(f"🚀 Processing {len(files)} files from {args.input_dir}...")

    for file_path in tqdm(files):
        name_no_ext = file_path.stem
        
        # Check if subject has age label
        if name_no_ext not in subject_ages:
            continue

        try:
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            raw.pick("eeg")
            
            # Channel selection logic
            # This logic mimics your original script for specific dataset prefixes
            base_prefix = '_'.join(name_no_ext.split('_')[:2])
            target_chs = SPECIAL_CHANNELS if base_prefix in ['BD_011', 'BD_005'] else DEFAULT_CHANNELS
            
            # Find indices
            ch_indices = []
            available_chs = raw.ch_names
            missing = False
            for ch in target_chs:
                if ch in available_chs:
                    ch_indices.append(available_chs.index(ch))
                else:
                    missing = True
                    break
            
            if missing: 
                continue

            # Process Data
            data = raw.get_data()
            if data.shape[1] < 24000:
                data = extend_signal(data)
            else:
                data = data[:, :24000]
                
            # Z-score normalization
            data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

            # Segmentation loop
            for start in range(0, 24000, SEGMENT_LEN):
                segment = data[:, start:start + SEGMENT_LEN]
                if segment.shape[1] == SEGMENT_LEN:
                    tensor = get_multichannel_tensor(segment, ch_indices, scales)
                    
                    scalogram_tensors.append(torch.from_numpy(tensor).float())
                    ages_real_list.append(subject_ages[name_no_ext])
                    
                    norm_age = scaler.transform([[subject_ages[name_no_ext]]])[0][0]
                    ages_norm_list.append(norm_age)
                    names_list.append(name_no_ext)
                    
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

    # Save to H5
    print(f"💾 Saving {len(scalogram_tensors)} tensors to {args.output_file}...")
    
    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("images", data=torch.stack(scalogram_tensors).numpy())
        hf.create_dataset("ages_real", data=np.array(ages_real_list))
        hf.create_dataset("ages_norm", data=np.array(ages_norm_list))
        hf.create_dataset("names", data=np.array(names_list, dtype='S'))

    print("✅ Done!")

if __name__ == "__main__":
    main()