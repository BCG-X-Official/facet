"""
Core implementation of :mod:`gamma.ml.inspection`
"""
import logging
from typing import *

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from gamma.common import AllTracker
from gamma.common.fit import FittableMixin, T_Self
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.inspection._shap_decomposition import (
    ShapInteractionValueDecomposer,
    ShapValueDecomposer,
)
from gamma.sklearndf import BaseLearnerDF, ClassifierDF, RegressorDF
from gamma.sklearndf.pipeline import BaseLearnerPipelineDF
from gamma.viz.dendrogram import LinkageTree
from ._explainer import TreeExplainerFactory
from ._shap import (
    ClassifierShapInteractionValuesCalculator,
    ClassifierShapValuesCalculator,
    ExplainerFactory,
    RegressorShapInteractionValuesCalculator,
    RegressorShapValuesCalculator,
    ShapCalculator,
    ShapInteractionValuesCalculator,
)

log = logging.getLogger(__name__)

__all__ = ["LearnerInspector"]

#
# Type variables
#

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=BaseLearnerPipelineDF)


#
# Class definitions
#

__tracker = AllTracker(globals())


class LearnerInspector(
    FittableMixin[Sample], ParallelizableMixin, Generic[T_LearnerPipelineDF]
):
    """
    Inspect feature interactions in a learner pipeline through SHAP values.
    """

    COL_IMPORTANCE = "importance"
    COL_IMPORTANCE_MARGINAL = "marginal importance"
    COL_SPLIT = ShapCalculator.COL_SPLIT

    #: The default explainer factory used by this inspector.
    #: This is a tree explainer using the tree_path_dependent method for
    #: feature perturbation, so we can calculate SHAP interaction values
    DEFAULT_EXPLAINER_FACTORY = TreeExplainerFactory(
        feature_perturbation="tree_path_dependent"
    )

    def __init__(
        self,
        *,
        explainer_factory: Optional[ExplainerFactory] = None,
        shap_interaction: bool = True,
        min_direct_synergy: Optional[float] = None,
        random_state: Union[int, RandomState, None] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param explainer_factory: optional function that creates a shap Explainer \
            (default: :func:``.tree_explainer_factory``)
        :param shap_interaction: if ``True``, calculate SHAP interaction values, else \
            only calculate SHAP contribution values.\
            SHAP interaction values are needed to determine feature synergy and \
            redundancy; otherwise only SHAP association can be calculated.\
            (default: ``True``)
        :param min_direct_synergy: minimum direct synergy to consider a feature pair \
            for calculation of indirect synergy. \
            Only relevant if parameter ``shap_interaction`` is ``True``. \
            (default: <DEFAULT_MIN_DIRECT_SYNERGY>)
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        if explainer_factory:
            if not explainer_factory.explains_raw_output:
                raise ValueError(
                    "arg explainer_factory is not configured to explain raw output"
                )
        else:
            explainer_factory = self.DEFAULT_EXPLAINER_FACTORY
            assert explainer_factory.explains_raw_output

        if shap_interaction and not explainer_factory.supports_shap_interaction_values:
            log.warning(
                "ignoring arg shap_interaction=True: "
                "explainers made by arg explainer_factory do not support "
                "SHAP interaction values"
            )
            shap_interaction = False

        self._explainer_factory = explainer_factory
        self._shap_interaction = shap_interaction
        self._min_direct_synergy = min_direct_synergy
        self._random_state = random_state

        self._crossfit: Optional[LearnerCrossfit[T_LearnerPipelineDF]] = None
        self._shap_calculator: Optional[ShapCalculator] = None
        self._shap_decomposer: Optional[ShapValueDecomposer] = None

    # dynamically complete the __init__ docstring
    # noinspection PyTypeChecker
    __init__.__doc__ = (
        __init__.__doc__.replace(
            "<DEFAULT_MIN_DIRECT_SYNERGY>",
            str(ShapInteractionValueDecomposer.DEFAULT_MIN_DIRECT_SYNERGY),
        )
        + ParallelizableMixin.__init__.__doc__
    )

    def fit(self: T_Self, crossfit: LearnerCrossfit, **fit_params) -> T_Self:
        """
        Fit the inspector with the given sample.

        :param crossfit: the model crossfit to be explained during model inspection
        :param fit_params: additional keyword arguments (ignored)
        :return: ``self``
        """

        self: LearnerInspector  # support type hinting in PyCharm

        if not crossfit.is_fitted:
            raise ValueError("crossfit in arg pipeline is not fitted")

        learner: BaseLearnerDF = crossfit.pipeline.final_estimator

        if isinstance(learner, ClassifierDF):
            if len(crossfit.training_sample.target_columns) != 1:
                raise ValueError(
                    "only single-output classifiers (binary or multi-class) are "
                    "supported, but the classifier in the given crossfit has been "
                    "fitted on multiple columns "
                    f"{crossfit.training_sample.target_columns}"
                )

            is_classifier = True

        elif isinstance(learner, RegressorDF):
            is_classifier = False

        else:
            raise TypeError(
                "learner in given crossfit must be a classifier or a regressor,"
                f"but is a {type(learner).__name__}"
            )

        if self._shap_interaction:
            shap_calculator_type = (
                ClassifierShapInteractionValuesCalculator
                if is_classifier
                else RegressorShapInteractionValuesCalculator
            )
            shap_calculator = shap_calculator_type(
                explain_full_sample=False,
                explainer_factory=self._explainer_factory,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )
            shap_decomposer = ShapInteractionValueDecomposer(
                min_direct_synergy=self._min_direct_synergy
            )

        else:
            shap_calculator_type = (
                ClassifierShapValuesCalculator
                if is_classifier
                else RegressorShapValuesCalculator
            )
            shap_calculator = shap_calculator_type(
                explain_full_sample=False,
                explainer_factory=self._explainer_factory,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )
            shap_decomposer = ShapValueDecomposer()

        shap_calculator.fit(crossfit=crossfit)
        shap_decomposer.fit(shap_calculator=shap_calculator)

        self._shap_calculator = shap_calculator
        self._shap_decomposer = shap_decomposer
        self._crossfit = crossfit

        return self

    @property
    def is_fitted(self) -> bool:
        """``True`` if this inspector is fitted, else ``False``"""
        return self._crossfit is not None

    @property
    def crossfit(self) -> LearnerCrossfit[T_LearnerPipelineDF]:
        """
        CV fit of the pipeline being examined by this inspector.
        """
        self._ensure_fitted()
        return self._crossfit

    @property
    def training_sample(self) -> Sample:
        """
        The training sample used for model inspection.
        """
        self._ensure_fitted()
        return self.crossfit.training_sample

    @property
    def outputs(self) -> List[str]:
        """
        The names of the outputs explained by this inspector.

        For regressors, this corresponds to the number of targets.

        For binary classifiers, this is a single class, since the SHAP values of the
        second class can be trivially derived as the negation of SHAP values of the
        first class.

        For multi-class classifiers, this is the list of all classes.
        """

        self._ensure_fitted()

        estimator_df: BaseLearnerDF = self.crossfit.pipeline.final_estimator

        if isinstance(estimator_df, ClassifierDF):
            try:
                # noinspection PyUnresolvedReferences
                classes = estimator_df.classes_
                # for binary classifiers, we will only produce SHAP values for the first
                # class (since they would only be mirrored by the second class)
                return classes[:1] if len(classes) == 2 else classes
            except AttributeError as cause:
                raise TypeError(
                    f"underlying {type(estimator_df.__name__)} "
                    "does not implement 'classes_' attribute: "
                ) from cause

        else:
            return self.crossfit.training_sample.target_columns

    @property
    def features(self) -> List[str]:
        """
        The names of the features used to fit the learner explained by this inspector.
        """
        return self.crossfit.pipeline.features_out.to_list()

    def shap_values(self, consolidate: Optional[str] = "mean") -> pd.DataFrame:
        """
        Calculate the SHAP values for all splits.

        Each row is an observation in a specific test split, and each column is a
        feature. Values are the SHAP values per observation, calculated as the mean
        SHAP value across all splits that contain the observation.

        :param consolidate: consolidate SHAP values across splits; \
            permissible values are ``"mean"`` (calculate the mean), ``"std"`` \
            (calculate the standard deviation), or ``None`` to prevent consolidation \
            (default: ``"mean"``)
        :return: shap values as a data frame
        """
        self._ensure_fitted()
        return self._shap_calculator.get_shap_values(consolidate=consolidate)

    def shap_interaction_values(
        self, consolidate: Optional[str] = "mean"
    ) -> pd.DataFrame:
        """
        Calculate the SHAP interaction values for all splits.

        Each row is an observation in a specific test split, and each column is a
        combination of two features. Values are the SHAP interaction values per
        observation, calculated as the mean SHAP interaction value across all splits
        that contain the observation.

        :param consolidate: consolidate SHAP values across splits; \
            permissible values are ``"mean"`` (calculate the mean), ``"std"`` \
            (calculate the standard deviation), or ``None`` to prevent consolidation \
            (default: ``"mean"``)
        :return: SHAP interaction values as a data frame
        """
        self._ensure_fitted()
        return self._shap_interaction_values_calculator.get_shap_interaction_values(
            consolidate=consolidate
        )

    def feature_importance(
        self, *, method: str = "rms"
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Feature importance computed using relative absolute shap contributions across
        all observations.

        :param method: method for calculating feature importance. Supported methods \
            are ``rms`` (root of mean squares, default) and ``mav`` (mean absolute values)
        :return: importance of each feature as its mean absolute SHAP contribution, \
          normalised to a total 100%. Returned as a series of length n_features for \
          single-output models, and as a data frame of shape (n_features, n_outputs) \
          for multi-output models
        """

        methods = ["rms", "mav"]
        if method not in methods:
            raise ValueError(
                f'arg method="{method}" must be one of {{{", ".join(methods)}}}'
            )

        shap_matrix = self.shap_values()
        abs_importance: pd.Series
        if method == "rms":
            abs_importance = shap_matrix.pow(2).mean().pow(0.5)
        elif method == "mav":
            abs_importance = shap_matrix.abs().mean()
        else:
            raise ValueError(f"unknown method: {method}")

        total_importance: float = abs_importance.sum()

        feature_importance_sr: pd.Series = abs_importance.divide(
            total_importance
        ).rename(LearnerInspector.COL_IMPORTANCE)

        if len(self.outputs) > 1:
            assert (
                abs_importance.index.nlevels == 2
            ), "2 index levels in place for multi-output models"

            feature_importance_sr: pd.DataFrame = abs_importance.unstack(level=0)

        return feature_importance_sr

    def feature_association_matrix(self) -> pd.DataFrame:
        """
        Calculate the Pearson correlation matrix of the shap values.

        :return: data frame with column and index given by the feature names,
          and values as the Pearson correlations of the shap values of features
        """
        self._ensure_fitted()

        return self._shap_decomposer.association

    def feature_association_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Calculate the :class:`.LinkageTree` based on the
        :meth:`.feature_association_matrix`.

        :return: linkage tree for the shap clustering dendrogram; \
            list of linkage trees if the base estimator is a multi-output model
        """
        self._ensure_fitted()
        return self._linkages_from_affinity_matrices(
            feature_affinity_matrix=self._shap_decomposer.association_rel_
        )

    def feature_synergy_matrix(self) -> pd.DataFrame:
        """
        For each pairing of features, calculate the relative share of their synergistic
        contribution to the model prediction.

        The synergistic contribution of a pair of features ranges between 0.0
        (no synergy - both features contribute fully autonomously) and 1.0
        (full synergy - both features combine all of their information into a joint
        contribution).

        :return: feature synergy matrix as a data frame of shape \
            (n_features, n_outputs * n_features)
        """
        self._ensure_fitted()
        return self._shap_interaction_decomposer.synergy

    def feature_redundancy_matrix(self) -> pd.DataFrame:
        """
        For each pairing of features, calculate the relative share of their redundant
        contribution to the model prediction.

        The redundant contribution of a pair of features ranges between 0.0
        (no redundancy - both features contribute fully independently) and 1.0
        (full redundancy - the information used by either feature is fully redundant).

        :return: feature redundancy matrix as a data frame of shape \
            (n_features, n_outputs * n_features)
        """
        self._ensure_fitted()
        return self._shap_interaction_decomposer.redundancy

    def feature_redundancy_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Calculate the :class:`.LinkageTree` based on the
        :meth:`.feature_redundancy_matrix`.

        :return: linkage tree for the shap clustering dendrogram; \
            list of linkage trees if the base estimator is a multi-output model
        """
        self._ensure_fitted()
        return self._linkages_from_affinity_matrices(
            feature_affinity_matrix=self._shap_interaction_decomposer.redundancy_rel_
        )

    def feature_interaction_matrix(self) -> pd.DataFrame:
        """
        Calculate relative shap interaction values for all feature pairings.

        Shap interactions quantify direct interactions between pairs of features.
        For a quantification of overall interaction (including indirect interactions
        among more than two features), see :meth:`.feature_synergy_matrix`.

        The relative values are normalised to add up to 1.0, and each value ranges
        between 0.0 and 1.0.

        For features :math:`f_i` and :math:`f_j`, relative feature interaction
        :math:`I` is calculated as

        .. math::
            I_{ij} = \\frac
                {\\sigma(\\vec{\\phi}_{ij})}
                {\\sum_{a=1}^n \\sum_{b=1}^n \\sigma(\\vec{\\phi}_{ab})}

        where :math:`\\sigma(\\vec v)` is the standard deviation of all elements of
        vector :math:`\\vec v`.

        The total average interaction of features
        :math:`f_i` and :math:`f_j` is
        :math:`I_{ij} \
            + I_{ji} \
            = 2 I_{ij}`.

        :math:`I_{ii}` is the residual, non-synergistic contribution
        of feature :math:`f_i`

        The matrix returned by this method is a diagonal matrix

        .. math::

            \\newcommand\\fi[1]{I_{#1}}
            \\newcommand\\nan{\\mathit{nan}}
            \\fi{} = \\begin{pmatrix}
                \\fi{11} & \\nan & \\nan & \\dots & \\nan \\\\
                2\\fi{21} & \\fi{22} & \\nan & \\dots & \\nan \\\\
                2\\fi{31} & 2\\fi{32} & \\fi{33} & \\dots & \\nan \\\\
                \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
                2\\fi{n1} & 2\\fi{n2} & 2\\fi{n3} & \\dots & \\fi{nn} \\\\
            \\end{pmatrix}

        with :math:`\\sum_{a=1}^n \\sum_{b=a}^n I_{ab} = 1`

        :return: average shap interaction values as a data frame of shape \
            (n_features, n_outputs * n_features)
        """

        n_features = len(self.features)
        n_outputs = len(self.outputs)

        # get a feature interaction array with shape
        # (n_observations, n_outputs, n_features, n_features)
        # where the innermost feature x feature arrays are symmetrical
        im_matrix_per_observation_and_output = (
            self.shap_interaction_values(consolidate=None)
            .values.reshape((-1, n_features, n_outputs, n_features))
            .swapaxes(1, 2)
        )

        # calculate the average interactions for each output and feature/feature
        # interaction, based on the standard deviation assuming a mean of 0.0.
        # The resulting matrix has shape (n_outputs, n_features, n_features)
        interaction_matrix = np.sqrt(
            (
                im_matrix_per_observation_and_output
                * im_matrix_per_observation_and_output
            ).mean(axis=0)
        )
        assert interaction_matrix.shape == (n_outputs, n_features, n_features)

        # we normalise the synergy matrix for each output to a total of 1.0
        interaction_matrix /= interaction_matrix.sum()

        # the total interaction effect for features i and j is the total of matrix
        # cells (i,j) and (j,i); theoretically both should be the same but to minimize
        # numerical errors we total both in the lower matrix triangle (but excluding the
        # matrix diagonal, hence k=1)
        interaction_matrix += np.triu(interaction_matrix, k=1).swapaxes(1, 2)

        # discard the upper matrix triangle by setting it to nan
        interaction_matrix += np.triu(
            np.full(shape=(n_features, n_features), fill_value=np.nan), k=1
        )[np.newaxis, :, :]

        # create a data frame from the feature matrix
        return self._feature_matrix_to_df(interaction_matrix)

    def _feature_matrix_to_df(self, matrix: np.ndarray) -> pd.DataFrame:
        # transform a matrix of shape (n_outputs, n_features, n_features)
        # to a data frame

        n_features = len(self.features)
        n_outputs = len(self.outputs)

        assert matrix.shape == (n_outputs, n_features, n_features)

        # transform to 2D shape (n_features, n_outputs * n_features)
        matrix_2d = matrix.swapaxes(0, 1).reshape((n_features, n_outputs * n_features))

        # convert array to data frame with appropriate indices
        matrix_df = pd.DataFrame(
            data=matrix_2d,
            columns=self.shap_values().columns,
            index=self.crossfit.pipeline.features_out.rename(Sample.COL_FEATURE),
        )

        assert matrix_df.shape == (n_features, n_outputs * n_features)

        return matrix_df

    def _linkages_from_affinity_matrices(
        self, feature_affinity_matrix: np.ndarray
    ) -> Union[LinkageTree, List[LinkageTree]]:
        # calculate the linkage trees for all outputs in a feature distance matrix;
        # matrix has shape (n_outputs, n_features, n_features) with values ranging from
        # (1 = closest, 0 = most distant)
        # return a linkage tree if there is only one output, else return a list of
        # linkage trees

        feature_importance = self.feature_importance(method="rms")

        if len(feature_affinity_matrix) == 1:
            # we have only a single output
            # feature importance is already a series
            return self._linkage_from_affinity_matrix_for_output(
                feature_affinity_matrix[0], feature_importance
            )

        else:
            return [
                self._linkage_from_affinity_matrix_for_output(
                    feature_affinity_for_output, feature_importance_for_output
                )
                for feature_affinity_for_output, (
                    _,
                    feature_importance_for_output,
                ) in zip(feature_affinity_matrix, feature_importance.iteritems())
            ]

    @staticmethod
    def _linkage_from_affinity_matrix_for_output(
        feature_affinity_matrix: np.ndarray, feature_importance: pd.Series
    ) -> LinkageTree:
        # calculate the linkage tree from the a given output in a feature distance
        # matrix;
        # matrix has shape (n_features, n_features) with values ranging from
        # (1 = closest, 0 = most distant)

        # compress the distance matrix (required by SciPy)
        compressed_distance_vector = squareform(1 - abs(feature_affinity_matrix))

        # calculate the linkage matrix
        linkage_matrix = linkage(y=compressed_distance_vector, method="single")

        # Feature labels and weights will be used as the leaves of the linkage tree.
        # Select only the features that appear in the distance matrix, and in the
        # correct order

        # build and return the linkage tree
        return LinkageTree(
            scipy_linkage_matrix=linkage_matrix,
            leaf_labels=feature_importance.index,
            leaf_weights=feature_importance.values,
            max_distance=1.0,
        )

    def _ensure_shap_interaction(self) -> None:
        if not self._shap_interaction:
            raise RuntimeError(
                "SHAP interaction values have not been calculated. "
                "Create an inspector with parameter 'shap_interaction=True' to "
                "enable calculations involving SHAP interaction values."
            )

    @property
    def _shap_interaction_values_calculator(self) -> ShapInteractionValuesCalculator:
        self._ensure_shap_interaction()
        return cast(ShapInteractionValuesCalculator, self._shap_calculator)

    @property
    def _shap_interaction_decomposer(self) -> ShapInteractionValueDecomposer:
        self._ensure_shap_interaction()
        return cast(ShapInteractionValueDecomposer, self._shap_decomposer)


__tracker.validate()
