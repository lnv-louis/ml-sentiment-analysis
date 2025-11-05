# Sentiment Analysis: Bidirectional LSTM vs LinearSVC

Comparative analysis of Bidirectional LSTM and LinearSVC for sentiment classification on financial news headlines.

## Research Question

How does the performance of a Bidirectional LSTM neural network compare to that of a LinearSVC in terms of classification accuracy, precision, recall, and F1 Score when analyzing sentiments in financial news?

## Results

- **LinearSVC**: F1 Score of 0.7376 (73.76%), Classification Accuracy of ~75%
- **Bidirectional LSTM**: F1 Score of 0.6843 (68.43%), Classification Accuracy of ~73% (final epoch)
- LinearSVC demonstrated more consistent and stable performance across all metrics
- LinearSVC achieved better precision and recall consistently across all 6 training epochs

## Dataset

- **Source**: FinancialPhraseBank (Kaggle)
- **Size**: 4,846 financial news headlines
- **Labels**: Positive, Neutral, Negative
- **Split**: 70% training, 30% testing

## Implementation

### Technologies
- Python 3.10
- TensorFlow/Keras (Bidirectional LSTM)
- Scikit-learn (LinearSVC)
- NLTK (text preprocessing)
- Pandas, NumPy, Matplotlib, Seaborn

### Preprocessing Pipeline
1. Remove punctuation marks
2. Remove stop-words using NLTK
3. Apply stemming using PorterStemmer
4. Remove most frequently occurring words
5. Apply lemmatization using WordNet Lemmatizer
6. Convert text to sequences (LSTM) or TF-IDF vectors (LinearSVC)

### Model Architectures

**Bidirectional LSTM**
- Embedding layer (max_tokens=10,000, output_dim=128)
- Bidirectional LSTM layers (64 units)
- Dropout (0.5) for regularization
- Dense layers (32 units + output layer)
- Trained for 6 epochs
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy

**LinearSVC**
- TF-IDF vectorization
- Linear Support Vector Classification
- Trained for 6 iterations (epochs)
- Regularization parameter: C=1.0

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| LinearSVC | ~75% | 0.75 | 0.75 | 0.7376 |
| Bidirectional LSTM | ~73% | 0.73 | 0.73 | 0.6843 |

### Detailed Classification Metrics

**LSTM Model:**
- True Positives: 1,023 (Negative: 38, Neutral: 765, Positive: 220)
- True Negatives: 2,477
- False Positives: 431
- False Negatives: 431

**LinearSVC Model:**
- True Positives: 1,094 (Negative: 89, Neutral: 807, Positive: 198)
- True Negatives: 2,548
- False Positives: 360
- False Negatives: 360

## Key Findings

1. **LinearSVC outperformed LSTM** with a 7.8% higher F1 score (0.7376 vs 0.6843), demonstrating better generalization on the financial news dataset.
2. **LSTM exhibited overfitting** with loss increasing from ~0.73 to ~1.11 between epochs 2-6, despite training accuracy improvements.
3. **LinearSVC showed superior balance** with equal false positive and false negative counts (360 each), indicating balanced classification across all sentiment classes.
4. **Both models benefited from comprehensive text preprocessing**, including stopword removal, stemming, and lemmatization.
5. **Simpler architecture advantage**: LinearSVC's linear model proved more effective than deep learning for this specific NLP task, likely due to the dataset size and feature characteristics.

## Files

- `model.ipynb` - Complete implementation notebook
- `sentimentDatabase.csv` - Dataset (4,846 financial news headlines)

## Usage

1. Clone the repository
2. Create and activate a virtual environment (Python 3.10):
   ```bash
   python3.10 -m venv myenv
   source myenv/bin/activate  # On macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
5. Run `model.ipynb` in Jupyter Notebook

## Author

**Vu Nguyen Le (Louis Le)**

BSc Computer Science @ University College London (UCL)

- Email: lelouis.lnv@gmail.com
- GitHub: [@lnv-louis](https://github.com/lnv-louis)
- Website: [lenguyenvu.com](https://lenguyenvu.com)

## Related Projects

- [AML Subtypes Classification](https://github.com/lnv-louis/AML-Subtypes-Classification-ML) - ML research on cancer subtype classification
- [Grounded - AI Fact-Checking](https://github.com/lnv-louis/grounded-fact-checking) - Full-stack AI fact-checking platform

## License

This research was conducted as part of academic work exploring machine learning applications in natural language processing.
