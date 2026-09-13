from RAG import retrieve

test_questions = [
    "What is overfitting?",
    "How does k-nearest neighbors make predictions?",
    "What is the difference between L1 and L2 regularization?",
    "What is cross-validation and why is it useful?",
    "How does a decision tree make predictions?",
    "What is the purpose of regularization?",
    "What is the difference between training and test data?",
    "What is feature selection?",
    "What is linear regression?",
    "What is the purpose of a validation set?"
]

for question in test_questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    results = retrieve(question, k=5)

    for i, result in enumerate(results):

        print(f"\n--- RESULT {i + 1} ---")
        print(f"Distance: {result['distance']:.4f}")
        print(
            f"Pages: {result['page_start']} - "
            f"{result['page_end']}"
        )

        print(result["text"][:500])