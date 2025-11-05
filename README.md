# Sentiment Analysis: Bidirectional LSTM vs LinearSVC

Comparative analysis of Bidirectional LSTM and LinearSVC for sentiment classification on financial news headlines.

## Research Question

How does the performance of a Bidirectional LSTM neural network compare to that of a LinearSVC in terms of classification accuracy, precision, recall, and F1 Score when analyzing sentiments in financial news?

## Results

- **LinearSVC**: F1 Score of 0.74, Classification Accuracy of 75.52%
- **Bidirectional LSTM**: F1 Score of ~0.72, Classification Accuracy of 70.36%
- LSTM showed overfitting after epoch 3
- LinearSVC demonstrated more consistent and stable performance across all metrics

## Dataset

- **Source**: FinancialPhraseBank (Kaggle)
- **Size**: 4,846 financial news headlines
- **Labels**: Positive, Neutral, Negative
- **Split**: 70% training, 30% testing

## Implementation

### Technologies
- Python 3.12
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
- Trained for 8 epochs with early stopping

**LinearSVC**
- TF-IDF vectorization
- Linear Support Vector Classification
- Trained for 8 iterations

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| LinearSVC | 75.52% | ~0.75 | ~0.76 | 0.74 |
| Bidirectional LSTM | 70.36% | ~0.73 | ~0.70 | ~0.72 |

## Key Findings

1. LinearSVC outperformed LSTM in this task, likely due to better generalization on the dataset and simpler architecture reducing overfitting risk.
2. LSTM showed overfitting after epoch 3 with declining validation accuracy despite improving training accuracy.
3. Both models benefited from comprehensive text preprocessing.

## Files

- `model.ipynb` - Complete implementation notebook
- `sentimentDatabase.csv` - Dataset (4,846 financial news headlines)

## Usage

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install numpy pandas seaborn nltk scikit-learn tensorflow matplotlib
   ```
3. Download NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
4. Run `model.ipynb` in Jupyter Notebook

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
