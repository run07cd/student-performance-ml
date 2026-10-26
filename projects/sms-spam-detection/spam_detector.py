"""Simple text classification demo with a small toy SMS dataset.

The small embedded dataset is only for learning; it is not suitable for real-world spam filtering.
"""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

MESSAGES = [
    ("ham", "Hey, are we still meeting for lunch today?"),
    ("ham", "Please call me when you reach home."),
    ("ham", "I sent you the notes from class."),
    ("ham", "Can you bring the book tomorrow?"),
    ("ham", "Your appointment is confirmed for 10 AM."),
    ("ham", "Thanks for helping me with the assignment."),
    ("ham", "I will be five minutes late."),
    ("ham", "The train is delayed, I will update you."),
    ("ham", "Do you want tea or coffee?"),
    ("ham", "Please check the email I sent."),
    ("ham", "Happy birthday! Have a great day."),
    ("ham", "I am outside the library."),
    ("ham", "The class starts at nine tomorrow."),
    ("ham", "I have reached the bus stop."),
    ("ham", "Can we discuss the project this evening?"),
    ("ham", "Your parcel has been delivered to reception."),
    ("spam", "Congratulations! You won a free prize. Claim now."),
    ("spam", "Claim your cash reward now by clicking this link."),
    ("spam", "Win a gift card today, limited offer."),
    ("spam", "Urgent! You have been selected for a free vacation."),
    ("spam", "Earn money fast from home. Sign up now."),
    ("spam", "You are a lucky winner, claim your bonus today."),
    ("spam", "Exclusive offer! Buy now and save big."),
    ("spam", "Free entry in our prize draw. Reply to claim."),
    ("spam", "Your reward is waiting. Click to collect now."),
    ("spam", "Get a special discount today only."),
    ("spam", "You have won a cash prize. Claim your reward."),
    ("spam", "Limited time deal, subscribe now for a bonus."),
    ("spam", "Congratulations winner! Claim your free gift."),
    ("spam", "Act now to receive your exclusive reward."),
    ("spam", "Free offer available today. Click here now."),
    ("spam", "Final notice: claim your prize before it expires."),
]

def main():
    texts = [message for _, message in MESSAGES]
    labels = [label for label, _ in MESSAGES]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("SMS Spam Detection (toy example)")
    print(f"Accuracy on a tiny held-out sample: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, zero_division=0))
    examples = [
        "Congratulations, claim your free reward now",
        "Are we meeting at the library after class?",
    ]
    for message, prediction in zip(examples, model.predict(examples)):
        print(f"\nMessage: {message}\nPrediction: {prediction.upper()}")
    print("\nNote: this tiny toy dataset is for demonstration only, not reliable spam filtering.")

if __name__ == "__main__":
    main()
