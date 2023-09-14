from collections import Counter
from typing import List, NamedTuple

from linear_algebra import Vector, distance


def raw_majority_vote(labels: List[str]) -> str:
    """Assumes that labels are ordered from nearest to farthest."""
    votes = Counter(labels)
    winner, winner_count = votes.most_common(1)[0]
    num_winners = len([count for count in votes.values() if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return raw_majority_vote(labels[:-1])

assert raw_majority_vote(['a', 'b', 'c', 'a', 'b']) == 'a'

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int,
                 labeled_points: List[LabeledPoint],
                 new_point: Vector
                 ):
    # distance = sqrt( sum[(x - y)^2])
    by_distance = sorted(labeled_points, key=lambda lp:distance(lp.point, new_point))
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    return raw_majority_vote(k_nearest_labels)
