import math
from collections import defaultdict
from typing import Set, NamedTuple, Dict, Iterable, Tuple
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

        self.token_spam_counts = {"spam": 1, "rules": 1}
        self.token_ham_counts = {"ham": 2, "rules": 1, "hello": 1}
        self.k = 1
        self.spam_messages = 1
        self.ham_messages = 2

        ---
        token = "spam"
        p_token_spam = (1 + 1) / (1 + 2) = 0.66
        p_token_ham = (0 + 1) / (2 + 2) = 0.25
        ---
        token = "ham"
        p_token_spam = (0 + 1) / (1 + 2) = 0.33
        p_token_ham = (2 + 1) / (2 + 2) = ~0.75


        :param token:
        :return: P(token | spam) and P(token | not spam)
        """

        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # iterate through each word in our vocabulary
        for token in self.tokens: # self.tokens = {"spam", "ham", "rules", "hello"}
            prob_if_spam, prob_if_ham = self._probabilities(token) # P(X=x|spam), P(X=x| not spam)
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)

        return prob_if_spam / (prob_if_spam + prob_if_ham) # P(X=x|spam) / [ P(X=x|spam), P(X=x| not spam) ]


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

text = "hello spam"
out = c.predict(text)

print(out)

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

a = c.predict(text=text)
b = p_if_spam / (p_if_spam + p_if_ham)

assert a == b

