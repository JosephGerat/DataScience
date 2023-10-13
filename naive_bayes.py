import math
from collections import defaultdict
from typing import Set, NamedTuple, Dict, Iterable, Tuple, List
import re


def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9']+", text)
    return set(all_words)


out = tokenize("Data Science is science")

assert out == {"data", "science", "is"}


class Message(NamedTuple):
    text: str
    is_spam: bool


class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5):
        self.k = k

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # Increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # Increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """
            P(x1|y=spam)
            P(x1|y=ham)
        :param token:
        :return:
        """
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k) #
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)

        return prob_if_spam / (prob_if_spam + prob_if_ham)


def test1():

    """
    Naive Bayes
    P(S|B) = [P(B|S)*P(S)]/[P(B|S)*P(S) + P(B|not S)*P(not S)]

    """

    messages = [
        Message("spam rules", is_spam=True),
        Message("ham rules", is_spam=False),
        Message("hello ham", is_spam=False),
    ]

    c = NaiveBayesClassifier()
    c.train(messages)

    assert c.tokens == {"spam", "ham", "rules", "hello"}
    assert c.spam_messages == 1
    assert c.ham_messages == 2
    assert c.token_spam_counts == {"spam": 1, "rules": 1}
    assert c.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

    out = c.predict("hello spam")

    print(out)

    # P(S|B) = [P(B|S)*P(S)]/[P(B|S)*P(S) + P(B|not S)*P(not S)]
    # Bernoulli distribution :
    # p - probability of success

    # if x = 0      q = 1 - p
    # if x = 1      p
    probs_if_spam = [
        (1 + 0.5) / (1 + 2 * 0.5), # "spam" (present)
        1 - (0+0.5)/(1+2*0.5), # "ham" (not present)
        1 - (1+0.5)/(1+2*0.5), # "rules" (not present)
        (0+0.5)/(1+2*0.5) # "hello" (present)
    ]

    probs_if_ham = [
        (0 + 0.5) / (2 + 2 * 0.5), # "spam" (present)
        1 - (2+0.5)/(2+2*0.5), # "ham" (not present)
        1 - (1+0.5)/(2+2*0.5), # "rules" (not present)
        (1+0.5)/(2+2*0.5) # "hello" (present)
    ]

    p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
    p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

    assert c.predict("hello spam") == p_if_spam / (p_if_spam + p_if_ham)

def test2():
    messages = [
        Message("19 tdi", is_spam=True),
        Message("nafta", is_spam=True),
        Message("benzinak", is_spam=False),
        Message("turbo diesel", is_spam=True),
        Message("110kw tdi", is_spam=True),
        Message("TSI 81 kw", is_spam=False),
        Message("14 mpi", is_spam=False),
        Message("14 tsi", is_spam=False),
        Message("20 tdi rs", is_spam=True),
        Message("benzin", is_spam=False),
        Message("hladky benzin", is_spam=False),
    ]
    #5/11

    c = NaiveBayesClassifier()
    c.train(messages)

    # je jedno kolko slov tam ide, zalezi na  ulozenych tokenoch
    out = c.predict("bla")

    print(out)

def test_real_data():
    import glob, re

    path = 'spam_data/*/*'
    data: List[Message] = []

    for filename in glob.glob(path):
        is_spam = 'ham' not in filename

        with open(filename, errors='ignore') as email_file:
            for line in email_file:
                if line.startswith("Subject:"):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(subject, is_spam))
                    break

    import random
    from machine_learning import split_data

    random.seed(0)
    train_messages, test_messages = split_data(data, 0.55)

    model = NaiveBayesClassifier()
    model.train(train_messages)

    from collections import Counter

    o = []
    for message in test_messages:
        output = model.predict(message.text)
        o.append((message, output))

    confusion_matrix = Counter((message.is_spam, spam_probability > 0.5) for message, spam_probability in o)
    print(confusion_matrix)


if __name__ == "__main__": x()