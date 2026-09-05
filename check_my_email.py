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
Dear Students,

YBI Foundation / Skills4India is urgently hiring for the following position:

Technical Content Creator & Student Community Executive

The role involves technical content creation, teaching assistance, student mentoring, blogs, cheat sheets and S4I Club management.

Who can apply?

2025 and 2026 B.Tech/B.E. graduates

CSE, IT, AI/ML, Data Science, ECE or related branches

Freshers—no previous work experience required

Candidates currently residing in Delhi

Job Details

Work from office: West Delhi – 110018

Timings: 10:00 AM–7:00 PM

CTC: ₹2.70–₹3.60 LPA

Immediate joining preferred

Apply by: 19 September 2026

Interested candidates should immediately email their updated resume to support@ybifoundation.com with the subject:

Application – Technical Content Creator & Student Community Executive

YBI hiring for Delhi.png
For queries, call or WhatsApp 9667987711.

Apply now if you are interested in building your career in EdTech, technical content and student mentoring.

Regards,


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
