from transformers import pipeline

def classify_news(text, candidate_labels=["politics", "sports", "technology", "entertainment", "business"]):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(text, candidate_labels)
    return result

if __name__ == "__main__":
    article = input("Enter the news article text: ")
    topics = classify_news(article)
    print("Classification:", topics)
