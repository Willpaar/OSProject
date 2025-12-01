import os
import argparse
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import feature_extraction
from model_cnn import CNN_DDOS_Detector

# ---------------------------
# DATA LOADING
# ---------------------------
def load_raw_csv(filepath='processed_data/csv/RAW.csv'):
    df = pd.read_csv(filepath)

    # Encode categorical columns
    for col in ['Source', 'Destination', 'Protocol']:
        df[col] = LabelEncoder().fit_transform(df[col])

    numeric_features = [
        'Source Port', 'Destination Port', 'Sequence Number', 'Acknowledgment Number',
        'Window Size', 'Payload Length', 'TSval', 'TSecr', 'NodeWeight', 'EdgeWeight'
    ]

    feature_cols = numeric_features + ['Source', 'Destination', 'Protocol']
    X = df[feature_cols].astype(float).values
    y = np.where(df['Classification'].str.lower() == 'internal', 0, 1)

    return train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# CLASS BALANCING
# ---------------------------
def balance_classes(X_train, y_train, method='oversample'):
    normal_mask = (y_train == 0)
    attack_mask = (y_train == 1)

    X_normal = X_train[normal_mask]
    y_normal = y_train[normal_mask]
    X_attack = X_train[attack_mask]
    y_attack = y_train[attack_mask]

    print(f"\nClass Distribution Before Balancing:")
    print(f"  Normal: {len(X_normal)} | Attack: {len(X_attack)}")

    if method == 'oversample':
        n_samples = len(X_attack)
        X_normal_resampled = resample(X_normal, n_samples=n_samples, replace=True, random_state=42)
        y_normal_resampled = resample(y_normal, n_samples=n_samples, replace=True, random_state=42)
        X_balanced = np.vstack([X_normal_resampled, X_attack])
        y_balanced = np.hstack([y_normal_resampled, y_attack])
    else:
        n_samples = len(X_normal)
        X_attack_resampled = resample(X_attack, n_samples=n_samples, replace=False, random_state=42)
        y_attack_resampled = resample(y_attack, n_samples=n_samples, replace=False, random_state=42)
        X_balanced = np.vstack([X_normal, X_attack_resampled])
        y_balanced = np.hstack([y_normal, y_attack_resampled])

    print(f"Class Distribution After Balancing ({method}):")
    print(f"  Normal: {np.sum(y_balanced==0)} | Attack: {np.sum(y_balanced==1)}")
    return X_balanced, y_balanced

# ---------------------------
# PLOTTING
# ---------------------------
def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, save_path='outputs'):
    epochs = list(range(len(train_losses)))
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy')
    plt.legend(); plt.grid(True, alpha=0.3)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/training_curves_cnn.png', dpi=150)
    plt.show()
    print(f"Training curves saved to {save_path}/training_curves_cnn.png")

# ---------------------------
# TRAINING
# ---------------------------
def train_model(X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.001, batch_size=32,
                model_path='models/cnn_ddos_model.pkl', balance_classes_flag=True,
                feature_columns_path='models/feature_columns.pkl'):
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if balance_classes_flag:
        X_train, y_train = balance_classes(X_train, y_train)

    # Save feature columns
    feature_names = [
        'Source Port', 'Destination Port', 'Sequence Number', 'Acknowledgment Number',
        'Window Size', 'Payload Length', 'TSval', 'TSecr', 'NodeWeight', 'EdgeWeight',
        'Source', 'Destination', 'Protocol'
    ]
    os.makedirs(os.path.dirname(feature_columns_path), exist_ok=True)
    with open(feature_columns_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Saved feature columns to {feature_columns_path}")

    # Reshape for CNN
    X_train = feature_extraction.reshape_for_cnn(X_train)
    X_val = feature_extraction.reshape_for_cnn(X_val)

    model = CNN_DDOS_Detector(n_features=128, n_classes=2)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    n_batches = len(X_train)//batch_size
    best_val_acc = 0

    for epoch in range(num_epochs):
        indices = np.random.permutation(len(X_train))
        X_shuf, y_shuf = X_train[indices], y_train[indices]

        epoch_loss, epoch_correct, epoch_samples = 0, 0, 0
        for b in range(n_batches):
            start, end = b*batch_size, (b+1)*batch_size
            X_batch, y_batch = X_shuf[start:end], y_shuf[start:end]

            loss = model.train_step(X_batch, y_batch, learning_rate)
            epoch_loss += loss
            y_pred = model.predict(X_batch)
            epoch_correct += np.sum(y_pred==y_batch)
            epoch_samples += len(y_batch)

        avg_loss = epoch_loss/n_batches
        train_losses.append(avg_loss)
        train_acc = (epoch_correct/epoch_samples)*100
        train_accuracies.append(train_acc)

        val_probs,_ = model.forward(X_val)
        val_loss = model.compute_loss(val_probs, y_val)
        val_losses.append(val_loss)
        val_acc = model.evaluate(X_val, y_val)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_model(model_path)

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")
    return model, train_losses, val_losses, train_accuracies, val_accuracies

# ---------------------------
# MAIN
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN model for DDOS detection')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--use_spliced', action='store_true')
    parser.add_argument('--spliced_dir', type=str, default='data/spliced')
    parser.add_argument('--max_chunks', type=int, default=None)
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--no_balance', action='store_true')
    args = parser.parse_args()

    model_path = 'models/cnn_ddos_model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Deleted old model at {model_path}")

    try:
        train_data = preprocess.preprocess_training_data(
            filepath='processed_data/csv/RAW.csv',
            val_split=0.2,
            save_processed=True,
            use_spliced=args.use_spliced,
            spliced_dir=args.spliced_dir,
            max_chunks=args.max_chunks
        )
        X_train = train_data['X_train']
        y_train = train_data['y_train']
        X_val = train_data['X_val']
        y_val = train_data['y_val']
    except:
        print("Preprocessing failed. Loading CSV directly...")
        X_train, X_val, y_train, y_val = load_raw_csv('processed_data/csv/RAW.csv')

    if args.sample_size and args.sample_size < len(X_train):
        indices = np.random.choice(len(X_train), args.sample_size, replace=False)
        X_train, y_train = X_train[indices], y_train[indices]

    use_balancing = not args.no_balance or args.balance

    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        X_train, y_train, X_val, y_val,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        model_path=model_path,
        balance_classes_flag=use_balancing
    )

    print("✓ Training complete! Model saved to models/cnn_ddos_model.pkl")
    plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies)
