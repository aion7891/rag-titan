"""Texts with planted topic changes, for testing TextTiling."""

import random

TOPICS = [words.split() for words in [
    "laptop charger dock software repair screen keyboard",
    "travel train hotel flight receipt booking portal",
    "leave holiday vacation sick note doctor absence",
    "salary payroll bonus tax payslip bank pension",
    "office desk chair badge parking kitchen floor",
    "security password phishing account login token",
]]
COMMON = "employee team manager company week month request policy".split()
# stopwords, which TextTiling must ignore
GLUE = "the a of to and is for with in on".split()

THREE_TOPICS = (
    "The new laptop arrives on your first day. The laptop comes with a charger and a dock. "
    "IT installs the laptop software before you start. A broken laptop goes back to IT for "
    "repair. Travel is booked through the travel portal. Train travel is preferred for short "
    "trips. Hotel costs for travel are refunded up to 150 euros. Travel receipts go to finance "
    "within a month. Leave days are requested in the leave tool. Every employee has twenty "
    "leave days. Unused leave days carry over to March. Sick leave needs a note from the third "
    "day.")


def sentence(rng, topic):
    words = ([rng.choice(TOPICS[topic]) for _ in range(rng.randint(2, 5))]
             + [rng.choice(COMMON) for _ in range(rng.randint(1, 3))]
             + [rng.choice(GLUE) for _ in range(rng.randint(0, 4))])
    rng.shuffle(words)
    text = " ".join(words)
    return text[0].upper() + text[1:] + "."


def planted_text(seed):
    """
    A text of 2 to 4 topics with 3 to 8 sentences each, sometimes in paragraphs.

    Returns:
        (text, changes): changes are the sentence numbers where a new topic starts.
    """
    rng = random.Random(seed)
    sentences, changes = [], []
    for topic in rng.sample(range(len(TOPICS)), rng.randint(2, 4)):
        if sentences:
            changes.append(len(sentences))
        sentences += [sentence(rng, topic) for _ in range(rng.randint(3, 8))]
    text = ""
    for number, line in enumerate(sentences):
        gap = "\n\n" if number + 1 in changes and rng.random() < 0.5 else " "
        text += line + (gap if number + 1 < len(sentences) else "")
    return text, changes
