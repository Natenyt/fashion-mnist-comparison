# Fashion MNIST Classification - Assignment 7

A comparative study of CNN architectures for fashion item classification using the Fashion MNIST dataset, demonstrating the impact of model complexity on accuracy.

## 📋 Assignment Requirements

- Use Fashion MNIST dataset (10 clothing categories)
- Train two models: one with **< 93% accuracy** and one with **>= 93% accuracy**
- Compare model performance
- Display wrong predictions and error analysis

## 🎯 Results

| Model | Accuracy | Parameters | Training Time |
|-------|----------|------------|---------------|
| Simple Model | ~87-90% | ~50K | ~2 min |
| Advanced Model | ~93-94% | ~1.2M | ~8-10 min |

## 👕 Fashion MNIST Classes

The dataset contains 10 fashion categories:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## 🏗️ Model Architectures

### Simple Model (< 93% accuracy)
```
Conv2D(16) → MaxPooling2D → Flatten → Dense(64) → Dense(10)
```

**Characteristics:**
- Minimal layers
- No dropout or batch normalization
- Fewer filters
- Faster training but lower accuracy

### Advanced Model (>= 93% accuracy)
```
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25) →
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25) →
Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Dense(10)
```

**Characteristics:**
- Deeper architecture with 4 convolutional layers
- Batch normalization for stable training
- Dropout for regularization
- More parameters but better generalization

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip

### Install Dependencies
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Run the Training Script
```bash
python assignment7.py
```

### What Happens:
1. Downloads Fashion MNIST dataset (60,000 train + 10,000 test images)
2. Trains **Simple Model** (5 epochs)
3. Trains **Advanced Model** (15 epochs)
4. Compares both models
5. Analyzes errors and generates visualizations
6. Creates confusion matrix

### Output Files
- `fashion_wrong_predictions.png` - Examples of misclassified items
- `fashion_correct_predictions.png` - Examples of correct predictions
- `fashion_training_comparison.png` - Training history for both models
- `fashion_confusion_matrix.png` - Confusion matrix showing class-wise performance

## 📊 Sample Output

```
======================================
MODEL COMPARISON
======================================
Simple Model Accuracy:   88.42%
Advanced Model Accuracy: 93.15%
Improvement: 4.73%

======================================
PREDICTIONS & ERROR ANALYSIS
======================================
Total test images: 10000
Correct predictions: 9315
Wrong predictions: 685
Error rate: 6.85%
```

## 📁 Project Structure

```
fashion-mnist-assignment/
│
├── assignment7.py                      # Main training script
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── fashion_wrong_predictions.png      # Generated output
├── fashion_correct_predictions.png    # Generated output
├── fashion_training_comparison.png    # Generated output
└── fashion_confusion_matrix.png       # Generated output
```

## 🧪 Dataset

**Fashion MNIST Database**
- Training samples: 60,000
- Test samples: 10,000
- Image size: 28x28 pixels (grayscale)
- Classes: 10 (clothing items)
- Created by: Zalando Research

Source: [Fashion MNIST on GitHub](https://github.com/zalandoresearch/fashion-mnist)

## 🔧 Hyperparameters

### Simple Model
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Batch Size | 128 |
| Epochs | 5 |
| Validation Split | 10% |

### Advanced Model
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Batch Size | 64 |
| Epochs | 15 |
| Validation Split | 10% |
| Dropout Rates | 0.25, 0.5 |

## 📈 Key Findings

### Why Advanced Model Performs Better:

1. **Deeper Architecture** - More layers = more feature extraction
2. **Batch Normalization** - Stabilizes training and allows higher learning rates
3. **Dropout Regularization** - Prevents overfitting
4. **More Filters** - Captures more complex patterns
5. **Longer Training** - 15 epochs vs 5 epochs

### Common Misclassifications:

Fashion MNIST is harder than digit MNIST because:
- Similar items: Shirt vs T-shirt, Pullover vs Coat
- Viewpoint variations
- Style diversity within classes

Most confused pairs:
- Shirt ↔ T-shirt/top
- Pullover ↔ Coat ↔ Shirt
- Sneaker ↔ Ankle boot

## 🎓 Learning Outcomes

This assignment demonstrates:
- Impact of model architecture on performance
- Trade-offs between model complexity and accuracy
- Importance of regularization techniques
- Fashion item classification challenges
- Error analysis and visualization techniques

## 🛠️ Improvements & Extensions

Possible enhancements:
- Data augmentation (rotation, flipping, zooming)
- Transfer learning from pre-trained models
- Ensemble methods
- Attention mechanisms
- Advanced architectures (ResNet, EfficientNet)

## 📝 Requirements.txt

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
seaborn>=0.11.0
```

## 👨‍💻 Author

**Davlat Eshnazarov**
- Course: Artificial Intelligence
- Assignment: 7 - Fashion MNIST Comparison

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Fashion MNIST dataset by Zalando Research
- TensorFlow/Keras framework
- Course instructor and materials

---


**Note:** The exact accuracy values may vary slightly between runs due to random initialization, but the pattern (Simple < 93%, Advanced >= 93%) remains consistent.
