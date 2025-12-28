import copy
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

OUTPUT_DIR = r"models/weights"

# (Subjective!)
COST_MATRIX = np.array([
    [0, 1, 2, 3],  # True: Benign
    [2, 0, 1, 2],  # True: Low
    [5, 3, 0, 1],  # True: Medium
    [10, 5, 2, 0]  # True: High
])

def map_labels_to_severity(y_series):
    severity_map = {
        'BENIGN': 0,
        'FTP-Patator': 1, 'SSH-Patator': 1, 'PortScan': 1,
        'DoS slowloris': 2, 'DoS Slowhttptest': 2, 'DoS Hulk': 2, 'DoS GoldenEye': 2,
        'WebAttackBruteForce': 2, 'WebAttackXSS': 2, 'WebAttackSqlInjection': 2,
        'Bot': 3, 'DDoS': 3, 'Heartbleed': 3, 'Infiltration': 3
    }
    # Map, filling unknown with -1 (or handling appropriately)
    return y_series.map(severity_map)

def get_weights_from_matrix(cost_matrix):
    # Sum the row costs to get a scalar "Severity Weight" for each class
    return np.sum(cost_matrix, axis=1)

class WeightedLR:
    def __init__(self):
        self.weights = get_weights_from_matrix(COST_MATRIX)
        self.weight_dict = {i: w for i, w in enumerate(self.weights)}
        
        self.model = LogisticRegression(
            class_weight=self.weight_dict, 
            # multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=2000,
            C=0.1  # <--- CHANGED: Stronger Regularization
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        print(f"Training Logistic Regression with weights: {self.weight_dict}...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
class XGBoost:
    def __init__(self):
        self.weights = get_weights_from_matrix(COST_MATRIX)
        self.weight_dict = {i: w for i, w in enumerate(self.weights)}
        
        self.model = xgb.XGBClassifier(
            objective='multi:softprob', 
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            reg_alpha=0.5,
            reg_lambda=1.0
        )

    def fit(self, X, y):
        print(f"Training XGBoost with weights: {self.weight_dict}...")
        sample_weights = np.array([self.weight_dict[val] for val in y])
        self.model.fit(X, y, sample_weight=sample_weights)

    def predict(self, X):
        return self.model.predict(X)

def clean_dataset(X):
    """
    1. Sanitizes Infinity/NaN.
    2. Applies Log-Transform to skewed columns (compression).
    """
    # 1. Sanitize
    if hasattr(X, 'replace'): # Pandas
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        # Convert to numpy for log math
        X_val = X.values
    else: # Numpy
        X_val = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Log-Transform Skewed Features
    # Network data (Bytes, Duration, Packets) is highly skewed.
    # We apply log1p (log(x+1)) to compress the range from [0, 10^9] to [0, 20].
    # This allows the NN to see the difference between "Small" and "Medium" flows.
    # Note: We apply it to absolute values to handle negatives if any exist (though rare in counts)
    X_log = np.sign(X_val) * np.log1p(np.abs(X_val))
    
    return X_log

class OrdinalNN(nn.Module):
    def __init__(self, input_dim, output_dim=4, device='cpu'):
        super(OrdinalNN, self).__init__()
        self.device = device
        
        # 1. MANDATORY SCALER
        self.scaler = StandardScaler()
        
        # 2. Deeper, wider architecture
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, 512),       # Wider layer
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01), # Leaky ReLU prevents dead neurons
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            
            nn.Linear(128, output_dim)
        ).to(device)

        self.cost_tensor = torch.tensor(COST_MATRIX, dtype=torch.float32).to(device)
        
        # 3. Optimizer with Weight Decay
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 4. Scheduler: Lowers LR if loss stops improving
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

    def ordinal_cost_loss(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        batch_costs = self.cost_tensor[targets] 
        loss = torch.sum(probs * batch_costs, dim=1).mean()
        return loss

    def fit(self, X, y, epochs=50, batch_size=2048, patience=10):
        print(f"Training Ordinal NN on {self.device}...")
        
        # --- CRITICAL: SCALE DATA ---
        # If we don't scale, gradients explode.
        X_clean = clean_dataset(X)
        X_scaled = self.scaler.fit_transform(X_clean)
        
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long).to(self.device)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.train()
        
        # Early Stopping Variables
        best_loss = float('inf')
        patience_counter = 0
        best_model_wts = copy.deepcopy(self.state_dict())
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.layer_stack(batch_X)
                loss = self.ordinal_cost_loss(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            
            # Step the scheduler
            self.scheduler.step(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.5f}")
            
            # --- EARLY STOPPING CHECK ---
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Best Loss: {best_loss:.5f}")
                    break
        
        # Load best weights
        self.load_state_dict(best_model_wts)
        torch.save(best_model_wts, OUTPUT_DIR)

    def predict(self, X):
        self.eval()
        X_clean = clean_dataset(X)
        X_scaled = self.scaler.transform(X_clean)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.layer_stack(X_t)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()