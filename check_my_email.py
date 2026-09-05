"""
check_my_email.py
-------------------
Apne khud ke email ko check karne ke liye ye script use karo.

Kaise use karein:
1. Neeche "MY_EMAIL_TEXT" variable ke andar apna email ka text paste karo
   (subject + body, jo bhi text hai, seedha triple quotes ke andar)
2. Terminal mein ye command chalao:
       python check_my_email.py
3. Result terminal mein dikhega: "phishing" ya "safe", saath mein confidence %
"""

from phishing_detector import main, predict_email

# 👇 Paste Your Mail Here
MY_EMAIL_TEXT = """
Enter Your Email/Paste
"""

if __name__ == "__main__":
    print("Training the model, please wait...\n")
    vectorizer, clf, eng_cols = main()

    print("\n" + "=" * 60)
    print("Result of your email:")
    print("=" * 60)

    label, probabilities = predict_email(MY_EMAIL_TEXT, vectorizer, clf, eng_cols)

    if label == "phishing":
        print(f"\n⚠️  PHISHING EMAIL!")
    else:
        print(f"\n✅  SAFE EMAIL.")

    print(f"\nConfidence:")
    for cls, prob in probabilities.items():
        print(f"  {cls}: {prob * 100:.1f}%")
