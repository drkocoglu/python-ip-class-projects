from cardclassifier import dataset


def test_parse_truth_single():
    assert dataset.parse_truth("Card_1_Diamond_Q.tif") == [("Q", "Diamonds")]


def test_parse_truth_multi_and_typo():
    pairs = dataset.parse_truth("Cards_5_Club_Q_Diamand_J.tif")
    assert pairs == [("Q", "Clubs"), ("J", "Diamonds")]


def test_parse_truth_unlabeled_is_empty():
    assert dataset.parse_truth("Cards_1.tif") == []


def test_optimal_assignments_prefers_suit_match():
    preds = [("K", "Spades"), ("3", "Diamonds")]
    truth = [("3", "Diamonds"), ("K", "Spades")]
    assignments = dataset.optimal_assignments(preds, truth)
    assert assignments == [(1, 0)]  # card0 -> K Spades, card1 -> 3 Diamonds


def test_optimal_assignments_empty():
    assert dataset.optimal_assignments([], [("K", "Spades")]) == []


def test_forced_labels_unique():
    preds = [("K", "Spades"), ("3", "Diamonds")]
    truth = [("3", "Diamonds"), ("K", "Spades")]
    labels = dataset.forced_labels(preds, truth)
    assert labels[0] == ("K", "Spades")
    assert labels[1] == ("3", "Diamonds")


def test_forced_labels_ambiguous_rank_is_none():
    # Two identical suits and unhelpful rank predictions: rank stays undecided,
    # suit is forced because both candidates agree on it.
    preds = [(None, "Spades"), (None, "Spades")]
    truth = [("K", "Spades"), ("J", "Spades")]
    labels = dataset.forced_labels(preds, truth)
    assert labels[0] == (None, "Spades")
    assert labels[1] == (None, "Spades")
