import numpy as np
import pickle
import os


class CNN_DDOS_Detector:
    """
    Convolutional Neural Network for DDOS Attack Detection.
    Uses 1D CNN architecture for network flow feature analysis.
    """
    
    def __init__(self, n_features, n_classes=2):
        """
        Initialize the CNN model for DDOS detection.
        
        :param n_features: Number of input features
        :param n_classes: Number of output classes (2 for binary: Normal vs Attack)
        """
        self.n_features = n_features
        self.n_classes = n_classes
        
        # Initialize CNN layers
        self.conv1_filters = 32
        self.conv2_filters = 64
        self.kernel_size = 3
        self.dense_units = 128
        
        # Initialize weights
        self.initialize_weights()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def initialize_weights(self):
        """
        Initialize weights for the CNN layers.
        Using He initialization for better convergence.
        """
        # Conv layer 1: (kernel_size, 1, conv1_filters)
        self.W_conv1 = np.random.randn(self.kernel_size, 1, self.conv1_filters) * np.sqrt(2.0 / self.kernel_size)
        self.b_conv1 = np.zeros(self.conv1_filters)
        
        # Conv layer 2: (kernel_size, conv1_filters, conv2_filters)
        self.W_conv2 = np.random.randn(self.kernel_size, self.conv1_filters, self.conv2_filters) * np.sqrt(2.0 / (self.kernel_size * self.conv1_filters))
        self.b_conv2 = np.zeros(self.conv2_filters)
        
        # Dense layer: needs to be initialized after first forward pass to know flattened size
        self.W_dense = None
        self.b_dense = np.zeros(self.dense_units)
        
        # Output layer
        self.W_out = np.random.randn(self.dense_units, self.n_classes) * np.sqrt(2.0 / self.dense_units)
        self.b_out = np.zeros(self.n_classes)
        
        # For momentum
        self.velocity = {}
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation for output layer."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def conv1d(self, X, W, b, stride=1):
        """
        1D Convolution operation.
        
        :param X: Input (batch_size, length, channels)
        :param W: Weights (kernel_size, in_channels, out_channels)
        :param b: Bias (out_channels,)
        :param stride: Stride for convolution
        :return: Convolution output
        """
        batch_size, length, in_channels = X.shape
        kernel_size, _, out_channels = W.shape
        
        out_length = (length - kernel_size) // stride + 1
        output = np.zeros((batch_size, out_length, out_channels))
        
        for i in range(out_length):
            start = i * stride
            end = start + kernel_size
            receptive_field = X[:, start:end, :]  # (batch, kernel_size, in_channels)
            
            for j in range(out_channels):
                # Element-wise multiply and sum
                conv = np.sum(receptive_field * W[:, :, j], axis=(1, 2))
                output[:, i, j] = conv + b[j]
        
        return output
    
    def max_pool1d(self, X, pool_size=2):
        """
        1D Max pooling.
        
        :param X: Input (batch_size, length, channels)
        :param pool_size: Size of pooling window
        :return: Pooled output
        """
        batch_size, length, channels = X.shape
        out_length = length // pool_size
        output = np.zeros((batch_size, out_length, channels))
        
        for i in range(out_length):
            start = i * pool_size
            end = start + pool_size
            output[:, i, :] = np.max(X[:, start:end, :], axis=1)
        
        return output
    
    def forward(self, X):
        """
        Forward pass through the CNN.
        
        :param X: Input features (batch_size, n_features, 1)
        :return: Output probabilities and cache for backward pass
        """
        cache = {}
        
        # Conv layer 1 + ReLU
        cache['input'] = X
        conv1 = self.conv1d(X, self.W_conv1, self.b_conv1)
        cache['conv1'] = conv1
        relu1 = self.relu(conv1)
        cache['relu1'] = relu1
        
        # Max pooling 1
        pool1 = self.max_pool1d(relu1, pool_size=2)
        cache['pool1'] = pool1
        
        # Conv layer 2 + ReLU
        conv2 = self.conv1d(pool1, self.W_conv2, self.b_conv2)
        cache['conv2'] = conv2
        relu2 = self.relu(conv2)
        cache['relu2'] = relu2
        
        # Max pooling 2
        pool2 = self.max_pool1d(relu2, pool_size=2)
        cache['pool2'] = pool2
        
        # Flatten
        batch_size = X.shape[0]
        flattened = pool2.reshape(batch_size, -1)
        cache['flattened'] = flattened
        
        # Initialize dense layer weights if not done yet
        if self.W_dense is None:
            flattened_size = flattened.shape[1]
            self.W_dense = np.random.randn(flattened_size, self.dense_units) * np.sqrt(2.0 / flattened_size)
        
        # Dense layer + ReLU
        dense = np.dot(flattened, self.W_dense) + self.b_dense
        cache['dense'] = dense
        relu_dense = self.relu(dense)
        cache['relu_dense'] = relu_dense
        
        # Output layer + Softmax
        output = np.dot(relu_dense, self.W_out) + self.b_out
        cache['output'] = output
        probs = self.softmax(output)
        cache['probs'] = probs
        
        return probs, cache
    
    def compute_loss(self, probs, y_true):
        """
        Compute cross-entropy loss.
        
        :param probs: Predicted probabilities
        :param y_true: True labels (batch_size,)
        :return: Loss value
        """
        batch_size = len(y_true)
        # Convert labels to one-hot
        y_one_hot = np.zeros((batch_size, self.n_classes))
        y_one_hot[np.arange(batch_size), y_true] = 1
        
        # Cross-entropy loss
        loss = -np.sum(y_one_hot * np.log(probs + 1e-8)) / batch_size
        return loss
    
    def backward_simple(self, cache, y_true, learning_rate):
        """
        Simplified backward pass and weight update.
        Uses gradient descent on the output and dense layers.
        
        :param cache: Forward pass cache
        :param y_true: True labels
        :param learning_rate: Learning rate for updates
        """
        batch_size = len(y_true)
        
        # Convert labels to one-hot
        y_one_hot = np.zeros((batch_size, self.n_classes))
        y_one_hot[np.arange(batch_size), y_true] = 1
        
        # Gradient of loss w.r.t. output
        d_output = (cache['probs'] - y_one_hot) / batch_size
        
        # Gradients for output layer
        d_W_out = np.dot(cache['relu_dense'].T, d_output)
        d_b_out = np.sum(d_output, axis=0)
        
        # Backprop through dense layer
        d_relu_dense = np.dot(d_output, self.W_out.T)
        d_dense = d_relu_dense * self.relu_derivative(cache['dense'])
        
        # Gradients for dense layer
        d_W_dense = np.dot(cache['flattened'].T, d_dense)
        d_b_dense = np.sum(d_dense, axis=0)
        
        # Update weights with momentum
        beta = 0.9
        
        if 'W_out' not in self.velocity:
            self.velocity['W_out'] = np.zeros_like(self.W_out)
            self.velocity['b_out'] = np.zeros_like(self.b_out)
            self.velocity['W_dense'] = np.zeros_like(self.W_dense)
            self.velocity['b_dense'] = np.zeros_like(self.b_dense)
        
        self.velocity['W_out'] = beta * self.velocity['W_out'] - learning_rate * d_W_out
        self.velocity['b_out'] = beta * self.velocity['b_out'] - learning_rate * d_b_out
        self.velocity['W_dense'] = beta * self.velocity['W_dense'] - learning_rate * d_W_dense
        self.velocity['b_dense'] = beta * self.velocity['b_dense'] - learning_rate * d_b_dense
        
        self.W_out += self.velocity['W_out']
        self.b_out += self.velocity['b_out']
        self.W_dense += self.velocity['W_dense']
        self.b_dense += self.velocity['b_dense']
    
    def train_step(self, X_batch, y_batch, learning_rate):
        """
        Perform one training step.
        
        :param X_batch: Input batch
        :param y_batch: Label batch
        :param learning_rate: Learning rate
        :return: Loss value
        """
        # Forward pass
        probs, cache = self.forward(X_batch)
        
        # Compute loss
        loss = self.compute_loss(probs, y_batch)
        
        # Backward pass and update
        self.backward_simple(cache, y_batch, learning_rate)
        
        return loss
    
    def predict(self, X):
        """
        Make predictions.
        
        :param X: Input features
        :return: Predicted class labels
        """
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def evaluate(self, X, y_true):
        """
        Evaluate model accuracy.
        
        :param X: Input features
        :param y_true: True labels
        :return: Accuracy percentage
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y_true) * 100
        return accuracy
    
    def save_model(self, filepath):
        """
        Save model weights to file.
        
        :param filepath: Path to save the model
        """
        if not filepath.endswith('.pkl'):
            filepath = filepath + '.pkl'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'W_conv1': self.W_conv1,
            'b_conv1': self.b_conv1,
            'W_conv2': self.W_conv2,
            'b_conv2': self.b_conv2,
            'W_dense': self.W_dense,
            'b_dense': self.b_dense,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'history': self.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model weights from file.
        
        :param filepath: Path to the saved model
        """
        if not filepath.endswith('.pkl'):
            filepath = filepath + '.pkl'
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_features = model_data['n_features']
        self.n_classes = model_data['n_classes']
        self.W_conv1 = model_data['W_conv1']
        self.b_conv1 = model_data['b_conv1']
        self.W_conv2 = model_data['W_conv2']
        self.b_conv2 = model_data['b_conv2']
        self.W_dense = model_data['W_dense']
        self.b_dense = model_data['b_dense']
        self.W_out = model_data['W_out']
        self.b_out = model_data['b_out']
        self.history = model_data.get('history', {})
        
        print(f"Model loaded from {filepath}")


if __name__ == '__main__':
    print("CNN DDOS Detector model loaded.")
    print("\nModel architecture:")
    print("  1. Conv1D (32 filters, kernel=3) + ReLU + MaxPool")
    print("  2. Conv1D (64 filters, kernel=3) + ReLU + MaxPool")
    print("  3. Dense (128 units) + ReLU")
    print("  4. Output (2 classes) + Softmax")
