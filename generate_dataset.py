"""
generate_dataset.py
--------------------
Generates a synthetic but realistic dataset of phishing and legitimate
emails for training the Phishing Email Detection Model.

In a real project you would replace this with a genuine labeled dataset
(e.g. the Kaggle "Phishing Email Dataset" or the Enron + Nazario phishing
corpus). This generator exists so the whole pipeline runs end-to-end
out of the box.
"""

import random
import csv

random.seed(42)

# ---------------------------------------------------------------------
# Building blocks for PHISHING emails
# ---------------------------------------------------------------------
phishing_subjects = [
    "Urgent: Verify Your Account Now",
    "Your Account Has Been Suspended",
    "Action Required: Unusual Login Detected",
    "Final Notice: Payment Failed",
    "You've Won a $1000 Gift Card!",
    "Security Alert: Update Your Password Immediately",
    "Your Package Could Not Be Delivered",
    "Confirm Your Identity to Avoid Account Closure",
    "IRS Tax Refund Pending - Action Needed",
    "Your PayPal Account Has Been Limited",
]

phishing_bodies = [
    "Dear Customer, we detected unusual activity on your account. Click here {url} "
    "to verify your identity immediately or your account will be suspended within 24 hours.",
    "Your account has been temporarily locked. Please login at {url} to restore access. "
    "Failure to act within 24 hours will result in permanent suspension.",
    "Congratulations! You have won a prize. Click {url} now and claim your reward "
    "before this limited time offer expires.",
    "We were unable to process your recent payment. Update your billing details at {url} "
    "immediately to avoid service interruption.",
    "This is an urgent security notice. Someone tried to access your account from an "
    "unrecognized device. Verify now at {url} or your account will be closed.",
    "Your package delivery failed due to an incomplete address. Confirm your details at {url} "
    "within 48 hours or the package will be returned.",
    "Dear valued customer, our records show your information needs updating. "
    "Click {url} and enter your login credentials to avoid losing access.",
    "URGENT: Your mailbox has exceeded its storage limit. Click {url} immediately "
    "to prevent your account from being deactivated.",
    "You have a pending refund of $284.50. Verify your bank details at {url} "
    "to receive your payment today.",
    "Act now! Your subscription will be cancelled unless you confirm your payment "
    "information at {url} within 24 hours.",
]

phishing_urls = [
    "http://secure-verify-account.com/login",
    "http://paypal-security-alert.net/update",
    "http://bank-login-verify.info/auth",
    "http://amaz0n-support.com/confirm",
    "http://account-update-now.biz/login",
    "http://192.168.44.21/verify",
    "http://appIe-id-locked.com/restore",
    "http://irs-gov-refund.net/claim",
]

# ---------------------------------------------------------------------
# Building blocks for LEGITIMATE emails
# ---------------------------------------------------------------------
legit_subjects = [
    "Weekly Team Meeting Notes",
    "Your Invoice from Acme Consulting",
    "Reminder: Dentist Appointment Tomorrow",
    "Project Timeline Update",
    "Newsletter: This Month's Highlights",
    "Lunch Plans for Friday?",
    "Your Order Has Shipped",
    "Q3 Budget Review Attached",
    "Welcome to the Team!",
    "Flight Confirmation for Your Upcoming Trip",
]

legit_bodies = [
    "Hi team, attached are the notes from this week's meeting. Let me know if I missed anything. Thanks!",
    "Hi, please find attached your invoice for services rendered in August. Payment is due within 30 days as usual.",
    "Hi there, just a reminder that your dentist appointment is scheduled for tomorrow at 10am. See you then.",
    "Hello everyone, wanted to share an update on the project timeline. We are on track to finish by the end of the month.",
    "Hi all, here is our monthly newsletter with updates from around the company. Enjoy the read.",
    "Hey, are you free for lunch on Friday? Thinking about trying that new place downtown.",
    "Hi, your order #48213 has shipped and is expected to arrive in 3-5 business days. Thanks for shopping with us.",
    "Hi team, attached is the Q3 budget review for your reference ahead of tomorrow's meeting.",
    "Welcome aboard! We're excited to have you join the team. Your manager will reach out with onboarding details.",
    "Hello, this confirms your flight from SFO to JFK departing next Tuesday at 7:45am. Have a great trip.",
    "Hi, just following up on our conversation from last week. Let me know if you have any questions.",
    "Hi, the report you requested is attached. Let me know if you'd like me to walk you through it.",
    "Good morning, reminder that the office will be closed on Monday for the holiday.",
    "Hi, thanks for your feedback on the draft. I've made the changes you suggested.",
    "Hello, your subscription renewal receipt is attached for your records. No action is needed.",
]


def make_phishing_email():
    subject = random.choice(phishing_subjects)
    body_template = random.choice(phishing_bodies)
    url = random.choice(phishing_urls)
    body = body_template.format(url=url)
    return subject, body


def make_legit_email():
    subject = random.choice(legit_subjects)
    body = random.choice(legit_bodies)
    return subject, body


def generate_dataset(n_per_class=180):
    rows = []
    for _ in range(n_per_class):
        subject, body = make_phishing_email()
        rows.append({"subject": subject, "body": body, "label": "phishing"})
    for _ in range(n_per_class):
        subject, body = make_legit_email()
        rows.append({"subject": subject, "body": body, "label": "safe"})
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = generate_dataset(n_per_class=180)
    out_path = "data/emails_dataset.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "body", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} emails -> {out_path}")
