import hashlib
from typing import List, Sequence

import pytest

from gamma.ml.selection import _T_LearnerPipelineDF, LearnerEvaluation
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.transformation import (
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
    ranking: List[LearnerEvaluation[_T_LearnerPipelineDF]],
    checksum_scores: float,
    checksum_learners: str,
    first_n_learners: int = 10,
) -> None:
    """
    Test helper to check rankings produced by gamma.ml rankers

    :param ranking: a list of LearnerEvaluations
    :param checksum_scores: the expected checksum of learner scores
    :param checksum_learners: the expected checksum of learners string reprs
    :param first_n_learners: the number of learners to check
    :return: None
    """

    assert sum(
        [learner_eval.ranking_score for learner_eval in ranking[:first_n_learners]]
    ) == pytest.approx(checksum_scores)

    assert (
        hashlib.md5(
            "".join(
                [
                    str(learner_eval.pipeline.final_estimator)
                    for learner_eval in ranking[:first_n_learners]
                ]
            ).encode("UTF-8")
        ).hexdigest()
        == checksum_learners
    )
