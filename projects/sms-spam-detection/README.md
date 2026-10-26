# SMS Spam Detection

A basic Natural Language Processing (NLP) example that classifies short messages as `spam` or `ham` (not spam).

## Run it

From the repository root:

```bash
pip install -r requirements.txt
python projects/sms-spam-detection/spam_detector.py
```

## What it demonstrates
- Converting text into TF-IDF features
- Training a Naive Bayes classifier
- Creating a train/test split
- Evaluating classification results
- Predicting labels for sample messages

**Important limitation:** the embedded dataset is small and hand-written for practice. Its score does not represent real-world performance.
