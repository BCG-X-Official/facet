"""
Core implementation of :mod:`facet.inspection`
"""
import logging
from typing import Generic, List, NamedTuple, Optional, Sequence, TypeVar, Union, cast

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin
from pytools.parallelization import ParallelizableMixin
from pytools.viz.dendrogram import LinkageTree
from sklearndf import ClassifierDF, LearnerDF, RegressorDF
from sklearndf.pipeline import LearnerPipelineDF

from ._explainer import ExplainerFactory, TreeExplainerFactory
from ._shap import (
    ClassifierShapInteractionValuesCalculator,
    ClassifierShapValuesCalculator,
    RegressorShapInteractionValuesCalculator,
    RegressorShapValuesCalculator,
    ShapCalculator,
    ShapInteractionValuesCalculator,
)
from facet import Sample
from facet.crossfit import LearnerCrossfit
from facet.inspection._shap_decomposition import (
    ShapInteractionValueDecomposer,
    ShapValueDecomposer,
)

log = logging.getLogger(__name__)

__all__ = ["ShapPlotData", "LearnerInspector"]
#
# Type variables
#

T = TypeVar("T")
T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=LearnerPipelineDF)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class ShapPlotData(NamedTuple):
    """
    Data for use in SHAP plots provided by the SHAP package (:module:`shap`)
    """

    #: Matrix of SHAP values (# observations x # features)
    #: or list of shap value matrices for multi-output models
    shap_values: Union[np.ndarray, List[np.ndarray]]

    #: Matrix of feature values (# observations x #features)
    features: pd.DataFrame

    #: Series of target values (# observations)
    #: or matrix of target values for multi-output models (# observations x # outputs)
    target: Union[pd.Series, pd.DataFrame]


@inheritdoc(match="[see superclass]")
class LearnerInspector(
    FittableMixin[Sample], ParallelizableMixin, Generic[T_LearnerPipelineDF]
):
    """
    Explain regressors and classifiers based on SHAP values.

    Focus is on explaining the overall model as well as individual observations.
    Given that SHAP values are estimations, this inspector operates based on crossfits
    to enable estimations of the uncertainty of SHAP values.

    Available inspection methods are:

    - SHAP values (mean or standard deviation across crossfits)
    - SHAP interaction values (mean or standard deviation across crossfits)
    - feature importance derived from SHAP values (either as mean absolute values \
      or as the root of mean squares)
    - (pairwise) feature interaction matrix (direct feature interaction quantified by \
      SHAP interaction values)
    - (pairwise) feature redundancy matrix (requires availability of SHAP interaction \
      values)
    - (pairwise) feature synergy matrix (requires availability of SHAP interaction \
      values)
    - (pairwise) feature association matrix (combined effect of redundancy and \
      synergy, if no SHAP interaction values are available)
    - feature redundancy linkage (to visualize clusters of redundant features in a \
      dendrogram)
    - feature synergy linkage (to visualize clusters of synergistic features in a \
      dendrogram)
    - feature association linkage (to visualize clusters of associated features in a \
      dendrogram)

    All inspections that aggregate across observations will respect sample weights, if
    specified in the underlying training sample.
    """

    #: name for "feature importance" series or column
    COL_IMPORTANCE = "importance"

    #: The default explainer factory used by this inspector.
    #: This is a tree explainer using the tree_path_dependent method for
    #: feature perturbation, so we can calculate SHAP interaction values
    DEFAULT_EXPLAINER_FACTORY = TreeExplainerFactory(
        feature_perturbation="tree_path_dependent", use_background_dataset=False
    )

    def __init__(
        self,
        *,
        explainer_factory: Optional[ExplainerFactory] = None,
        shap_interaction: bool = True,
        min_direct_synergy: Optional[float] = None,
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
            redundancy. \
            (default: ``True``)
        :param min_direct_synergy: minimum direct synergy to consider a feature pair \
            for calculation of indirect synergy, \
            only relevant if parameter ``shap_interaction`` is ``True``. \
            This effectively acts as a noise threshold given the approximate nature of \
            SHAP calculations. \
            Consider increasing this value if you see warnings about contravariant \
            indirect synergy when fitting this inspector. \
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

    def fit(self: T, crossfit: LearnerCrossfit, **fit_params) -> T:
        """
        Fit the inspector with the given crossfit.

        This will calculate SHAP values and, if enabled in the underlying SHAP
        explainer, also SHAP interaction values.

        :param crossfit: the model crossfit to be explained by this model inspector
        :param fit_params: additional keyword arguments (ignored; accepted for \
            compatibility with :class:`.FittableMixin`)
        :return: ``self``
        """

        self: LearnerInspector  # support type hinting in PyCharm

        if not crossfit.is_fitted:
            raise ValueError("crossfit in arg pipeline is not fitted")

        learner: LearnerDF = crossfit.pipeline.final_estimator

        if isinstance(learner, ClassifierDF):
            if len(crossfit.sample.target_columns) != 1:
                raise ValueError(
                    "only single-output classifiers (binary or multi-class) are "
                    "supported, but the classifier in the given crossfit has been "
                    "fitted on multiple columns "
                    f"{crossfit.sample.target_columns}"
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
        """[see superclass]"""
        return self._crossfit is not None

    @property
    def crossfit(self) -> LearnerCrossfit[T_LearnerPipelineDF]:
        """
        The crossfit with which this inspector was fitted.
        """
        self._ensure_fitted()
        return self._crossfit

    @property
    def training_sample(self) -> Sample:
        """
        The training sample of the crossfit with which this inspector was fitted.
        """
        self._ensure_fitted()
        return self.crossfit.sample

    @property
    def output_names(self) -> Sequence[str]:
        """
        The names of the outputs explained by this inspector.

        For regressors, these are the names of the target columns.

        For binary classifiers, this is a list of length 1 with the name of a single
        class, since the SHAP values of the second class can be trivially derived as
        the negation of the SHAP values of the first class.

        For non-binary classifiers, this is the list of all classes.
        """

        self._ensure_fitted()
        return self._shap_calculator.output_names_

    @property
    def features(self) -> List[str]:
        """
        The names of the features used to fit the learner pipeline explained by this
        inspector.
        """
        return self.crossfit.pipeline.features_out_.to_list()

    def shap_values(self, consolidate: Optional[str] = "mean") -> pd.DataFrame:
        """
        Calculate the SHAP values for all observations and features.

        Returns a data frame of SHAP values where each row corresponds to an
        observation, and each column corresponds to a feature.

        By default, one SHAP value is returned for each observation and feature; this
        value is calculated as the mean SHAP value across all crossfits.

        The ``consolidate`` argument can be used to disable or change the consolidation
        of SHAP values:

        - passing ``consolidate=None`` will disable SHAP value consolidation, \
          generating one row for every crossfit and observation (identified by
          a hierarchical index with two levels)
        - passing ``consolidate="std"`` will calculate the standard deviation of SHAP \
          values across all crossfits, as the basis for determining the uncertainty \
          of SHAP calculations

        :param consolidate: consolidate SHAP values across splits; \
            permissible values are ``"mean"`` (calculate the mean), ``"std"`` \
            (calculate the standard deviation), or ``None`` to prevent consolidation \
            (default: ``"mean"``)
        :return: a data frame with SHAP values
        """
        self._ensure_fitted()
        return self._shap_calculator.get_shap_values(consolidate=consolidate)

    def shap_interaction_values(
        self, consolidate: Optional[str] = "mean"
    ) -> pd.DataFrame:
        """
        Calculate the SHAP interaction values for all observations and pairs of
        features.

        Returns a data frame of SHAP interaction values where each row corresponds to an
        observation and a feature (identified by a hierarchical index with two levels),
        and each column corresponds to a feature.

        By default, one SHAP interaction value is returned for each observation and
        feature pairing; this value is calculated as the mean SHAP interaction value
        across all crossfits.

        The ``consolidate`` argument can be used to disable or change the consolidation
        of SHAP interaction values:

        - passing ``consolidate=None`` will disable SHAP interaction value \
          consolidation, generating one row for every crossfit, observation and \
          feature (identified by a hierarchical index with three levels)
        - passing ``consolidate="std"`` will calculate the standard deviation of SHAP \
          interaction values across all crossfits, as the basis for determining the \
          uncertainty of SHAP calculations

        :param consolidate: consolidate SHAP interaction values across splits; \
            permissible values are ``"mean"`` (calculate the mean), ``"std"`` \
            (calculate the standard deviation), or ``None`` to prevent consolidation \
            (default: ``"mean"``)
        :return: a data frame with SHAP interaction values
        """
        self._ensure_fitted()
        return self._shap_interaction_values_calculator.get_shap_interaction_values(
            consolidate=consolidate
        )

    def feature_importance(
        self, *, method: str = "rms"
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate the relative importance of each feature based on SHAP values.

        The importance values of all features always add up to ``1.0``.

        The calculation applies sample weights if specified in the underlying
        :attr:`.sample`.

        :param method: method for calculating feature importance. Supported methods \
            are ``rms`` (root of mean squares, default), ``mav`` (mean absolute \
            values)
        :return: a series of length `n_features` for single-output models, or a \
            data frame of shape (n_features, n_outputs) for multi-output models
        """

        methods = ["rms", "mav"]
        if method not in methods:
            raise ValueError(
                f'arg method="{method}" must be one of {{{", ".join(methods)}}}'
            )

        shap_matrix = self.shap_values(consolidate="mean")
        weight = self.training_sample.weight

        abs_importance: pd.Series
        if method == "rms":
            if weight is None:
                abs_importance = shap_matrix.pow(2).mean().pow(0.5)
            else:
                abs_importance = shap_matrix.pow(2).mul(weight, axis=0).mean().pow(0.5)
        elif method == "mav":
            if weight is None:
                abs_importance = shap_matrix.abs().mean()
            else:
                abs_importance = shap_matrix.abs().mul(weight, axis=0).mean()
        else:
            raise ValueError(f"unknown method: {method}")

        total_importance: float = abs_importance.sum()

        feature_importance: Union[pd.Series, pd.DataFrame]

        feature_importance = abs_importance.divide(total_importance).rename(
            LearnerInspector.COL_IMPORTANCE
        )

        if len(self.output_names) > 1:
            assert (
                abs_importance.index.nlevels == 2
            ), "2 index levels in place for multi-output models"

            feature_importance: pd.DataFrame = abs_importance.unstack(level=0)

        return feature_importance

    def feature_association_matrix(self) -> pd.DataFrame:
        """
        Calculate the feature association matrix.

        This yields a symmetrical matrix where each row and column represents one
        feature, and the values at the intersections are the pairwise feature
        associations, ranging from `0.0` (no association) to `1.0` (full association).

        The association of a feature with itself is defined as `1.0`.

        Feature association is the combined effect of feature redundancy and feature
        synergy (see methods :meth:`.feature_redundancy_matrix` and
        :meth:`feature_synergy_matrix`).

        While it is preferable to assess redundancy and synergy separately, association
        can be calculated using only SHAP values, and thus can be used as a fallback
        if no SHAP interaction values are available.

        In the case of multi-target regression and non-binary classification, returns
        a data frame with one matrix per output, stacked horizontally, and with a
        hierarchical column index (target/class name on level 1, and feature name on
        level 2).

        :return: feature association matrix as a data frame of shape \
            `(n_features, n_outputs * n_features)`
        """
        self._ensure_fitted()

        return self._shap_decomposer.association

    def feature_association_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Calculate a linkage tree based on the :meth:`.feature_association_matrix`.

        The linkage tree can be used to render a dendrogram indicating clusters of
        associated features.

        In the case of multi-target regression and non-binary classification, returns
        a list of linkage trees per target or class.

        :return: linkage tree of feature associations; list of linkage trees \
            for multi-target regressors or non-binary classifiers
        """
        self._ensure_fitted()
        return self._linkages_from_affinity_matrices(
            feature_affinity_matrix=self._shap_decomposer.association_rel_
        )

    def feature_synergy_matrix(self, symmetrical: bool = True) -> pd.DataFrame:
        """
        Calculate the feature synergy matrix.

        This yields a symmetrical matrix where each row and column represents one
        feature, and the values at the intersections are the pairwise feature synergies,
        ranging from `0.0` (no synergy - both features contribute to predictions fully
        autonomously of each other) to `1.0` (full synergy, both features rely on
        combining all of their information to achieve any contribution to predictions).

        The synergy of a feature with itself is defined as `1.0`.

        Feature synergy calculations require SHAP interaction values; if only SHAP
        values are available consider calculating feature associations instead
        (see :meth:`.feature_association_matrix`).

        In the case of multi-target regression and non-binary classification, returns
        a data frame with one matrix per output, stacked horizontally, and with a
        hierarchical column index (target/class name on level 1, and feature name on
        level 2).

        :param symmetrical: if ``True``, return a symmetrical matrix quantifying \
            mutual synergy; if ``False``, return an asymmetrical matrix quantifying \
            unilateral redundancy of the features represented by rows with the \
            features represented by columns
        :return: feature synergy matrix as a data frame of shape \
            `(n_features, n_outputs * n_features)`
        """
        self._ensure_fitted()
        return self._shap_interaction_decomposer.synergy(symmetrical=symmetrical)

    def feature_redundancy_matrix(self, symmetrical: bool = True) -> pd.DataFrame:
        """
        Calculate the feature redundancy matrix.

        This yields a symmetrical matrix where each row and column represents one
        feature, and the values at the intersections are the pairwise feature
        redundancies, ranging from `0.0` (no redundancy - both features contribute to
        predictions fully independently of each other) to `1.0` (full redundancy, either
        feature can replace the other feature without loss of predictive power).

        The redundancy of a feature with itself is defined as `1.0`.

        Feature redundancy calculations require SHAP interaction values; if only SHAP
        values are available consider calculating feature associations instead
        (see :meth:`.feature_association_matrix`).

        In the case of multi-target regression and non-binary classification, returns
        a data frame with one matrix per output, stacked horizontally, and with a
        hierarchical column index (target/class name on level 1, and feature name on
        level 2).

        :param symmetrical: if ``True``, return a symmetrical matrix quantifying \
            mutual redundancy; if ``False``, return an asymmetrical matrix quantifying \
            unilateral redundancy of the features represented by rows with the \
            features represented by columns
        :return: feature redundancy matrix as a data frame of shape \
            `(n_features, n_outputs * n_features)`
        """
        self._ensure_fitted()
        return self._shap_interaction_decomposer.redundancy(symmetrical=symmetrical)

    def feature_redundancy_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Calculate a linkage tree based on the :meth:`.feature_redundancy_matrix`.

        The linkage tree can be used to render a dendrogram indicating clusters of
        redundant features.

        In the case of multi-target regression and non-binary classification, returns
        a list of linkage trees per target or class.

        :return: linkage tree of feature redundancies; list of linkage trees \
            for multi-target regressors or non-binary classifiers
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
        across more than two features), see :meth:`.feature_synergy_matrix`.

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

            \\newcommand\\nan{\\mathit{nan}}
            I_{} = \\begin{pmatrix}
                I_{11} & \\nan & \\nan & \\dots & \\nan \\\\
                2I_{21} & I_{22} & \\nan & \\dots & \\nan \\\\
                2I_{31} & 2I_{32} & I_{33} & \\dots & \\nan \\\\
                \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
                2I_{n1} & 2I_{n2} & 2I_{n3} & \\dots & I_{nn} \\\\
            \\end{pmatrix}

        with :math:`\\sum_{a=1}^n \\sum_{b=a}^n I_{ab} = 1`

        In the case of multi-target regression and non-binary classification, returns
        a data frame with one matrix per output, stacked horizontally, and with a
        hierarchical column index (target/class name on level 1, and feature name on
        level 2).

        :return: relative shap interaction values as a data frame of shape \
            (n_features, n_outputs * n_features)
        """

        n_features = len(self.features)
        n_outputs = len(self.output_names)

        # get a feature interaction array with shape
        # (n_observations, n_outputs, n_features, n_features)
        # where the innermost feature x feature arrays are symmetrical
        im_matrix_per_observation_and_output = (
            self.shap_interaction_values(consolidate=None)
            .values.reshape((-1, n_features, n_outputs, n_features))
            .swapaxes(1, 2)
        )

        # get the observation weights with shape
        # (n_observations, n_outputs, n_features, n_features)
        weight: Optional[np.ndarray]
        _weight_sr = self.training_sample.weight
        if _weight_sr is not None:
            # if sample weights are defined, convert them to an array
            # and align the array with the dimensions of the feature interaction array
            weight = _weight_sr.values.reshape((-1, 1, 1, 1))
        else:
            weight = None

        # calculate the average interactions for each output and feature/feature
        # interaction, based on the standard deviation assuming a mean of 0.0.
        # The resulting matrix has shape (n_outputs, n_features, n_features)
        _interaction_squared = im_matrix_per_observation_and_output ** 2
        if weight is not None:
            _interaction_squared *= weight
        interaction_matrix = np.sqrt(_interaction_squared.mean(axis=0))
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

    def shap_plot_data(self) -> ShapPlotData:
        """
        Consolidate SHAP values and corresponding feature values from this inspector
        for use in SHAP plots offered by the :module:`shap` package.

        The _shap_ package provides functions for creating various SHAP plots.
        Most of these functions require

        - one or more SHAP value matrices as a single _numpy_ array, or a list of \
            _numpy_ arrays of shape _(n_observations, n_features)_
        - a feature matrix of shape _(n_observations, n_features)_, which can be \
            provided as a data frame to preserve feature names

        This method provides this data inside a :class:`.ShapPlotData` object, plus

        - the names of all outputs (i.e., the target names in case of regression, \
            or the class names in case of classification)
        - corresponding target values as a series, or as a data frame in the case of \
            multiple targets

        This method also ensures that the rows of all arrays, frames, and series are
        aligned, even if only a subset of the observations in the original sample was
        used to calculate SHAP values.

        Calculates mean shap values for each observation and feature, across all
        splits for which SHAP values were calculated.

        :return: consolidated SHAP and feature values for use shap plots
        """

        shap_values: pd.DataFrame = self.shap_values(consolidate="mean")
        sample: Sample = self.crossfit.sample.subsample(loc=shap_values.index)

        output_names = self.output_names

        if len(output_names) > 1:
            shap_values_numpy = [
                shap_values.xs(output_name, axis=1).values
                for output_name in output_names
            ]
        else:
            shap_values_numpy = shap_values.values

        return ShapPlotData(
            shap_values=shap_values_numpy,
            features=sample.features,
            target=sample.target,
        )

    def _feature_matrix_to_df(self, matrix: np.ndarray) -> pd.DataFrame:
        # transform a matrix of shape (n_outputs, n_features, n_features)
        # to a data frame

        n_features = len(self.features)
        n_outputs = len(self.output_names)

        assert matrix.shape == (n_outputs, n_features, n_features)

        # transform to 2D shape (n_features, n_outputs * n_features)
        matrix_2d = matrix.swapaxes(0, 1).reshape((n_features, n_outputs * n_features))

        # convert array to data frame with appropriate indices
        matrix_df = pd.DataFrame(
            data=matrix_2d,
            columns=self.shap_values().columns,
            index=self.crossfit.pipeline.features_out_.rename(Sample.IDX_FEATURE),
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
