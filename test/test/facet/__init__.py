from typing import Any, List, Mapping, Optional, Sequence

import pytest

from facet.selection import LearnerEvaluation
from sklearndf import TransformerDF
from sklearndf.transformation import (
    ColumnTransformerDF,
    OneHotEncoderDF,
    SimpleImputerDF,
)

STEP_IMPUTE = "impute"
STEP_ONE_HOT_ENCODE = "one-hot-encode"


def make_simple_transformer(
    impute_median_columns: Sequence[str] = None,
    one_hot_encode_columns: Sequence[str] = None,
) -> TransformerDF:
    column_transforms = []

    if impute_median_columns is not None and len(impute_median_columns) > 0:
        column_transforms.append(
            (STEP_IMPUTE, SimpleImputerDF(strategy="median"), impute_median_columns)
        )

    if one_hot_encode_columns is not None and len(one_hot_encode_columns) > 0:
        column_transforms.append(
            (
                STEP_ONE_HOT_ENCODE,
                OneHotEncoderDF(sparse=False, handle_unknown="ignore"),
                one_hot_encode_columns,
            )
        )

    return ColumnTransformerDF(transformers=column_transforms)


def check_ranking(
    ranking: List[LearnerEvaluation],
    expected_scores: Sequence[float],
    expected_learners: Optional[Sequence[type]],
    expected_parameters: Optional[Mapping[int, Mapping[str, Any]]],
) -> None:
    """
    Test helper to check rankings produced by learner rankers

    :param ranking: a list of LearnerEvaluations
    :param expected_scores: expected ranking scores, rounded to 3 decimal places
    :param expected_learners: expected learner classes
    :param expected_parameters: expected learner parameters
    :return: None
    """

    if expected_learners is None:
        expected_learners = [None] * len(ranking)

    for rank, (learner_eval, score_expected, learner_expected) in enumerate(
        zip(ranking, expected_scores, expected_learners)
    ):
        score_actual = round(learner_eval.ranking_score, 3)
        assert score_actual == pytest.approx(score_expected, abs=0.1), (
            f"unexpected score for learner at rank #{rank + 1}: "
            f"got {score_actual} but expected {score_expected}"
        )
        if learner_expected is not None:
            learner_actual = learner_eval.pipeline.final_estimator
            assert type(learner_actual) == learner_expected, (
                f"unexpected class for learner at rank #{rank}: "
                f"got {type(learner_actual)} but expected {learner_expected}"
            )

    if expected_parameters is not None:
        for rank, parameters_expected in expected_parameters.items():
            parameters_actual = ranking[rank].parameters
            assert parameters_actual == parameters_expected, (
                f"unexpected parameters for learner at rank #{rank}: "
                f"got {parameters_actual} but expected {parameters_expected}"
            )
