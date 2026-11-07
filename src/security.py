import re

def inspect_risk(message):
    """Simple, explainable rules; separate from the ML classification."""
    text = message.lower()
    flags = []

    if re.search(r"https?://|www\.", text):
        flags.append("Contains a URL")
    if any(term in text for term in ["password", "one-time password", "otp", "verify your account", "login details"]):
        flags.append("Mentions credentials or account verification")
    if any(term in text for term in ["urgent", "immediately", "act now", "account suspended", "expires today"]):
        flags.append("Uses urgency or pressure language")
    if any(term in text for term in ["you have won", "winner", "claim your prize", "free gift"]):
        flags.append("Contains a prize or free-gift claim")
    if any(term in text for term in ["bank account", "credit card", "payment details"]):
        flags.append("Mentions financial information")

    if len(flags) >= 3:
        level = "High"
    elif len(flags) >= 1:
        level = "Review"
    else:
        level = "No configured flags"
    return level, flags
