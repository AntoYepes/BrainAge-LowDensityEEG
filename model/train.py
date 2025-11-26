import os
import argparse
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from architecture import build_improved_resnet_model 

def parse_args():
    parser = argparse.ArgumentParser(description="Train Brain Age Model (Stratified & Full Metrics)")
    parser.add_argument('--data_path', type=Path, required=True, help="Path to .h5 dataset")
    parser.add_argument('--output_dir', type=Path, default='results', help="Directory to save models and logs")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()

def load_data_stratified(file_path):
    """
    Carga datos, decodifica nombres y realiza splits estratificados por edad.
    """
    print(f"📂 Loading H5 data from {file_path}...")
    with h5py.File(file_path, 'r') as f:
        X = f['images'][:]
        y_norm = f['ages_norm'][:]
        y_real = f['ages_real'][:]
        names = f['names'][:]
        # Decodificar nombres si están en bytes
        names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in names]

    # Detectar parámetros de normalización (para poder desnormalizar después)
    # Ajustamos una regresión simple para obtener mean y std exactos de la transformación original
    reg = LinearRegression()
    reg.fit(y_norm.reshape(-1, 1), y_real)
    y_std = reg.coef_[0]
    y_mean = reg.intercept_
    
    print(f"📊 Stats: Real Ages [{y_real.min():.1f}, {y_real.max():.1f}], Norm Params (Mean={y_mean:.2f}, Std={y_std:.2f})")

    # Split 1: Train+Val (80%) / Test (20%) - ESTRATIFICADO
    X_temp, X_test, y_temp, y_test, yr_temp, yr_test, n_temp, n_test = train_test_split(
        X, y_norm, y_real, names, test_size=0.2, random_state=42, stratify=y_real.astype(int)
    )

    # Split 2: Train (64%) / Val (16%) - ESTRATIFICADO
    # (0.2 del total es 0.25 del restante 0.8)
    X_train, X_val, y_train, y_val, yr_train, yr_val, n_train, n_val = train_test_split(
        X_temp, y_temp, yr_temp, n_temp, test_size=0.25, random_state=42, stratify=yr_temp.astype(int)
    )

    data = {
        'train': (X_train, y_train, yr_train, n_train),
        'val':   (X_val, y_val, yr_val, n_val),
        'test':  (X_test, y_test, yr_test, n_test)
    }
    
    return data, (y_mean, y_std)

def denormalize(y_norm_pred, y_mean, y_std):
    """Convierte predicciones normalizadas a edad real"""
    return y_norm_pred * y_std + y_mean

def evaluate_set(model, X, y_norm_true, y_real_true, names, set_name, norm_params):
    """Evalúa un conjunto específico y calcula métricas completas"""
    y_mean, y_std = norm_params
    
    # 1. Predicción (Normalizada)
    y_pred_norm = model.predict(X, verbose=0).flatten()
    
    # 2. Desnormalización (Para métricas reales)
    y_pred_real = denormalize(y_pred_norm, y_mean, y_std)
    
    # 3. Métricas
    mae = mean_absolute_error(y_real_true, y_pred_real)
    mse = mean_squared_error(y_real_true, y_pred_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_real_true, y_pred_real)
    
    # 4. Preparar DataFrame de resultados individuales
    df_results = pd.DataFrame({
        'Subject_ID': names,
        'Age_Real': y_real_true,
        'Age_Pred': y_pred_real,
        'BAG': y_pred_real - y_real_true,
        'Dataset': set_name
    })
    
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2, 
        'df': df_results
    }

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass

    # 1. Cargar Datos Estratificados
    data_splits, norm_params = load_data_stratified(args.data_path)
    X_train, y_train, _, _ = data_splits['train']
    X_val, y_val, _, _ = data_splits['val']
    
    input_shape = X_train.shape[1:] # (8, 30, 2400)
    print(f"🏗️ Building Model for input {input_shape}...")
    
    # 2. Construir y Entrenar Modelo
    model = build_improved_resnet_model(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss='huber', metrics=['mae'])
    
    callbacks = [
        ModelCheckpoint(filepath=str(args.output_dir / 'best_model.h5'), 
                        monitor='val_loss', save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
    ]
    
    print("🚀 Starting Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Guardar historial
    pd.DataFrame(history.history).to_csv(args.output_dir / 'training_history.csv', index=False)
    
    # 3. Evaluación Completa (Train/Val/Test)
    print("\n⚖️ Calculating Full Metrics (MAE, RMSE, R²)...")
    
    # Cargar el mejor modelo guardado para evaluación justa
    model.load_weights(str(args.output_dir / 'best_model.h5'))
    
    all_results_df = []
    metrics_summary = []

    for set_name in ['train', 'val', 'test']:
        X, y_n, y_r, names = data_splits[set_name]
        
        eval_res = evaluate_set(model, X, y_n, y_r, names, set_name, norm_params)
        
        all_results_df.append(eval_res['df'])
        metrics_summary.append({
            'Dataset': set_name.upper(),
            'MAE': eval_res['mae'],
            'RMSE': eval_res['rmse'],
            'R2': eval_res['r2']
        })
        
        print(f" > {set_name.upper()}: MAE={eval_res['mae']:.2f} | RMSE={eval_res['rmse']:.2f} | R²={eval_res['r2']:.3f}")

    # 4. Guardar Resultados Consolidados
    final_df = pd.concat(all_results_df, ignore_index=True)
    final_df.to_csv(args.output_dir / 'predictions_consolidated.csv', index=False)
    
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(args.output_dir / 'metrics_summary.csv', index=False)
    
    print(f"\n✅ Done! Results saved in '{args.output_dir}':")
    print("   - predictions_consolidated.csv (Individual predictions with IDs)")
    print("   - metrics_summary.csv (Overall performance)")
    print("   - best_model.h5")

if __name__ == "__main__":
    main()