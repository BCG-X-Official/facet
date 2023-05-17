"""
Implementation of :class:`.ModelInspector`.
"""
import logging
from abc import ABCMeta, abstractmethod
from types import MethodType
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial import distance
from typing_extensions import TypeAlias

from pytools.api import AllTracker
from pytools.data import LinkageTree, Matrix
from pytools.fit import FittableMixin, fitted_only
from pytools.parallelization import ParallelizableMixin

from facet.data import Sample
from facet.inspection._inspection import ShapPlotData
from facet.inspection._shap_projection import (
    ShapInteractionVectorProjector,
    ShapProjector,
    ShapVectorProjector,
)
from facet.inspection.shap import ShapCalculator

log = logging.getLogger(__name__)

__all__ = [
    "ModelInspector",
]


#
# Type aliases
#

FloatArray: TypeAlias = npt.NDArray[np.float_]
FloatMatrix: TypeAlias = Matrix[np.float_]


#
# Type variables
#

T_Model = TypeVar("T_Model")
T_ModelInspector = TypeVar("T_ModelInspector", bound="ModelInspector[Any]")
T_Number = TypeVar("T_Number", bound="np.number[Any]")
T_SeriesOrDataFrame = TypeVar("T_SeriesOrDataFrame", pd.Series, pd.DataFrame)


#
# Constants
#

ASSERTION__INSPECTOR_IS_FITTED = "inspector is fitted"
ASSERTION__SHAP_INTERACTION_SUPPORTED = "SHAP interaction values are supported"


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class ModelInspector(
    ParallelizableMixin, FittableMixin[Sample], Generic[T_Model], metaclass=ABCMeta
):
    """
    Explain a model based on SHAP values.

    .. note::
        This is an abstract base class for inspectors explaining different kinds of
        models based on SHAP values.

    Focus is on explaining the overall model, but the inspector also delivers
    SHAP explanations of the individual observations.

    Available inspection methods are:

    - SHAP values
    - SHAP interaction values
    - feature importance derived from SHAP values (either as mean absolute values
      or as the root of mean squares)
    - pairwise feature redundancy matrix (requires availability of SHAP interaction
      values)
    - pairwise feature synergy matrix (requires availability of SHAP interaction
      values)
    - pairwise feature association matrix (upper bound for redundancy but can be
      inflated by synergy; available if SHAP interaction values are unknown)
    - pairwise feature interaction matrix (direct feature interaction quantified by
      SHAP interaction values)
    - feature redundancy linkage (to visualize clusters of redundant features in a
      dendrogram; requires availability of SHAP interaction values)
    - feature synergy linkage (to visualize clusters of synergistic features in a
      dendrogram; requires availability of SHAP interaction values)
    - feature association linkage (to visualize clusters of associated features in a
      dendrogram)

    All inspections that aggregate across observations will respect sample weights, if
    specified in the underlying training sample.
    """

    # defined in superclass, repeated here for Sphinx
    n_jobs: Optional[int]

    # defined in superclass, repeated here for Sphinx
    shared_memory: Optional[bool]

    # defined in superclass, repeated here for Sphinx
    pre_dispatch: Optional[Union[str, int]]

    # defined in superclass, repeated here for Sphinx
    verbose: Optional[int]

    #: The model to inspect.
    model: T_Model

    #: If ``True``, calculate SHAP interaction values, else only calculate SHAP
    #: contribution values.
    #: SHAP interaction values are needed to determine feature synergy and redundancy.
    shap_interaction: bool

    #: Name for feature importance series or column.
    COL_IMPORTANCE = "importance"

    def __init__(
        self,
        model: T_Model,
        *,
        shap_interaction: bool = True,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param model: the model to inspect
        :param shap_interaction: if ``True``, calculate SHAP interaction values, else
            only calculate SHAP contribution values;
            SHAP interaction values are needed to determine feature synergy and
            redundancy
            (default: ``True``)
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        self.model = model
        self.shap_interaction = shap_interaction

        self._shap_projector: Optional[ShapProjector] = None
        self._sample: Optional[Sample] = None

    __init__.__doc__ = cast(str, __init__.__doc__) + cast(
        str, ParallelizableMixin.__init__.__doc__
    )

    def preprocess_features(
        self, features: Union[pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """
        Preprocess the features prior to calculating SHAP values.

        This method is called by :meth:`.fit` before fitting the inspector.
        By default, returns the features unchanged.

        :param features: features to preprocess
        :return: preprocessed features
        """
        return features

    def fit(
        self: T_ModelInspector, __sample: Sample, **fit_params: Any
    ) -> T_ModelInspector:
        """
        Fit the inspector with the given sample, creating global explanations including
        feature redundancy and synergy.

        This will calculate SHAP values and, if enabled in the underlying SHAP
        explainer, also SHAP interaction values.

        :param __sample: the background sample to be used for the global explanation
            of this model
        :param fit_params: additional keyword arguments (ignored; accepted for
            compatibility with :class:`.FittableMixin`)
        :return: ``self``
        """

        shap_calculator: ShapCalculator[Any] = self.shap_calculator

        shap_calculator.fit(self.preprocess_features(__sample.features))

        shap_global_projector: ShapProjector = (
            ShapInteractionVectorProjector()
            if self.shap_interaction
            else ShapVectorProjector()
        )

        shap_global_projector.fit(shap_calculator, sample_weight=__sample.weight)

        self._sample = __sample
        self._shap_projector = shap_global_projector

        return self

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self._sample is not None

    @property
    @fitted_only
    def sample_(self) -> Sample:
        """
        The background sample used to fit this inspector.
        """

        assert self._sample is not None, ASSERTION__INSPECTOR_IS_FITTED
        return self._sample

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """
        The feature names of the model being inspected.

        These names may differ from the feature names expected for the sample
        set has been preprocessed before calculating SHAP values.

        :return: the feature names
        """
        pass

    @property
    def output_names(self) -> List[str]:
        """
        The names of the outputs explained by this inspector.

        For regressors, these are the names of the target columns.

        For binary classifiers, this is a list of length 1 with the name of a single
        class, since the SHAP values of the second class can be trivially derived as
        the negation of the SHAP values of the first class.

        For non-binary classifiers, this is the list of all classes.
        """
        return self.shap_calculator.output_names

    @fitted_only
    def shap_values(self) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Calculate the SHAP values for all observations and features.

        Returns a data frame of SHAP values where each row corresponds to an
        observation, and each column corresponds to a feature.

        :return: a data frame with SHAP values
        """

        return self.__split_multi_output_df(self.shap_calculator.shap_values)

    @fitted_only
    def shap_interaction_values(self) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Calculate the SHAP interaction values for all observations and pairs of
        features.

        Returns a data frame of SHAP interaction values where each row corresponds to an
        observation and a feature (identified by a hierarchical index with two levels),
        and each column corresponds to a feature.

        :return: a data frame with SHAP interaction values
        """
        return self.__split_multi_output_df(
            self.shap_calculator.shap_interaction_values
        )

    @fitted_only
    def feature_importance(
        self, *, method: str = "rms"
    ) -> Union[pd.Series, pd.DataFrame]:
        # noinspection GrazieInspection
        """
        Calculate the relative importance of each feature based on SHAP values.

        The importance values of all features always add up to `1.0`.

        The calculation applies sample weights if specified in the underlying
        :class:`.Sample`.

        :param method: method for calculating feature importance. Supported methods
            are ``rms`` (root of mean squares, default), ``mav`` (mean absolute
            values)
        :return: a series of length `n_features` for single-output models, or a
            data frame of shape (n_features, n_outputs) for multi-output models
        """

        methods = {"rms", "mav"}
        if method not in methods:
            raise ValueError(f'arg method="{method}" must be one of {methods}')

        shap_matrix: pd.DataFrame = self.shap_calculator.shap_values
        weight: Optional[pd.Series] = self.sample_.weight

        abs_importance: pd.Series
        if method == "rms":
            if weight is None:
                abs_importance = shap_matrix.pow(2).mean().pow(0.5)
            else:
                abs_importance = shap_matrix.pow(2).mul(weight, axis=0).mean().pow(0.5)
        else:
            assert method == "mav", f"method is in {methods}"
            if weight is None:
                abs_importance = shap_matrix.abs().mean()
            else:
                abs_importance = shap_matrix.abs().mul(weight, axis=0).mean()

        def _normalize_importance(
            _importance: T_SeriesOrDataFrame,
        ) -> T_SeriesOrDataFrame:
            return _importance.divide(_importance.sum())

        if len(self.output_names) == 1:
            return _normalize_importance(abs_importance).rename(self.output_names[0])

        else:
            assert (
                abs_importance.index.nlevels == 2
            ), "2 index levels in place for multi-output models"

            return _normalize_importance(abs_importance.unstack(level=0))

    @fitted_only
    def feature_synergy_matrix(
        self,
        *,
        absolute: bool = False,
        symmetrical: bool = False,
        clustered: bool = True,
    ) -> Union[FloatMatrix, List[FloatMatrix]]:
        """
        Calculate the feature synergy matrix.

        This yields an asymmetric matrix where each row and column represents one
        feature, and the values at the intersections are the pairwise feature synergies,
        ranging from `0.0` (no synergy - both features contribute to predictions fully
        autonomously of each other) to `1.0` (full synergy, both features rely on
        combining all of their information to achieve any contribution to predictions).

        The synergy of a feature with itself is defined as `1.0`.

        Feature synergy calculations require SHAP interaction values; if only SHAP
        values are available consider calculating feature associations instead
        (see :meth:`.feature_association_matrix`).

        In the case of multi-target regression and non-binary classification, returns
        a list of data frames with one matrix per output.

        :param absolute: if ``False``, return relative synergy as a percentage of
            total feature importance;
            if ``True``, return absolute synergy as a portion of feature importance
        :param symmetrical: if ``True``, return a symmetrical matrix quantifying
            mutual synergy; if ``False``, return an asymmetrical matrix quantifying
            unilateral synergy of the features represented by rows with the
            features represented by columns (default: ``False``)
        :param clustered: if ``True``, reorder the rows and columns of the matrix
            such that synergy between adjacent rows and columns is maximised; if
            ``False``, keep rows and columns in the original features order
            (default: ``True``)
        :return: feature synergy matrix as a data frame of shape
            `(n_features, n_features)`, or a list of data frames for multiple outputs
        """

        return self.__feature_affinity_matrix(
            explainer_fn=self.__interaction_projector.synergy,
            absolute=absolute,
            symmetrical=symmetrical,
            clustered=clustered,
        )

    @fitted_only
    def feature_redundancy_matrix(
        self,
        *,
        absolute: bool = False,
        symmetrical: bool = False,
        clustered: bool = True,
    ) -> Union[FloatMatrix, List[FloatMatrix]]:
        """
        Calculate the feature redundancy matrix.

        This yields an asymmetric matrix where each row and column represents one
        feature, and the values at the intersections are the pairwise feature
        redundancies, ranging from `0.0` (no redundancy - both features contribute to
        predictions fully independently of each other) to `1.0` (full redundancy, either
        feature can replace the other feature without loss of predictive power).

        The redundancy of a feature with itself is defined as `1.0`.

        Feature redundancy calculations require SHAP interaction values; if only SHAP
        values are available consider calculating feature associations instead
        (see :meth:`.feature_association_matrix`).

        In the case of multi-target regression and non-binary classification, returns
        a list of data frames with one matrix per output.

        :param absolute: if ``False``, return relative redundancy as a percentage of
            total feature importance;
            if ``True``, return absolute redundancy as a portion of feature importance
        :param symmetrical: if ``True``, return a symmetrical matrix quantifying
            mutual redundancy; if ``False``, return an asymmetrical matrix quantifying
            unilateral redundancy of the features represented by rows with the
            features represented by columns (default: ``False``)
        :param clustered: if ``True``, reorder the rows and columns of the matrix
            such that redundancy between adjacent rows and columns is maximised; if
            ``False``, keep rows and columns in the original features order
            (default: ``True``)
        :return: feature redundancy matrix as a data frame of shape
            `(n_features, n_features)`, or a list of data frames for multiple outputs
        """

        return self.__feature_affinity_matrix(
            explainer_fn=self.__interaction_projector.redundancy,
            absolute=absolute,
            symmetrical=symmetrical,
            clustered=clustered,
        )

    @fitted_only
    def feature_association_matrix(
        self,
        *,
        absolute: bool = False,
        symmetrical: bool = False,
        clustered: bool = True,
    ) -> Union[FloatMatrix, List[FloatMatrix]]:
        """
        Calculate the feature association matrix.

        This yields an asymmetric matrix where each row and column represents one
        feature, and the values at the intersections are the pairwise feature
        associations, ranging from `0.0` (no association) to `1.0` (full association).

        The association of a feature with itself is defined as `1.0`.

        Feature association provides an upper bound for feature redundancy but might be
        inflated by feature synergy.

        While it is preferable to assess redundancy and synergy separately, association
        can be calculated using only SHAP values, and thus can be used as a fallback
        if no SHAP interaction values are available.

        In the case of multi-target regression and non-binary classification, returns
        a list of data frames with one matrix per output.

        :param absolute: if ``False``, return relative association as a percentage of
            total feature importance;
            if ``True``, return absolute association as a portion of feature importance
        :param symmetrical: if ``False``, return an asymmetrical matrix
            quantifying unilateral association of the features represented by rows
            with the features represented by columns;
            if ``True``, return a symmetrical matrix quantifying mutual association
            (default: ``False``)
        :param clustered: if ``True``, reorder the rows and columns of the matrix
            such that association between adjacent rows and columns is maximised; if
            ``False``, keep rows and columns in the original features order
            (default: ``True``)
        :return: feature association matrix as a data frame of shape
            `(n_features, n_features)`, or a list of data frames for multiple outputs
        """

        assert self._shap_projector is not None, ASSERTION__INSPECTOR_IS_FITTED
        return self.__feature_affinity_matrix(
            explainer_fn=self._shap_projector.association,
            absolute=absolute,
            symmetrical=symmetrical,
            clustered=clustered,
        )

    @fitted_only
    def feature_synergy_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Calculate a linkage tree based on the :meth:`.feature_synergy_matrix`.

        The linkage tree can be used to render a dendrogram indicating clusters of
        synergistic features.

        In the case of multi-target regression and non-binary classification, returns
        a list of linkage trees per target or class.

        :return: linkage tree of feature synergies; list of linkage trees
            for multi-target regressors or non-binary classifiers
        """

        feature_affinity_matrix = self.__interaction_projector.synergy(
            symmetrical=True, absolute=False
        )
        assert (
            feature_affinity_matrix is not None
        ), ASSERTION__SHAP_INTERACTION_SUPPORTED

        return self.__linkages_from_affinity_matrices(
            feature_affinity_matrix=feature_affinity_matrix
        )

    @fitted_only
    def feature_redundancy_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Calculate a linkage tree based on the :meth:`.feature_redundancy_matrix`.

        The linkage tree can be used to render a dendrogram indicating clusters of
        redundant features.

        In the case of multi-target regression and non-binary classification, returns
        a list of linkage trees per target or class.

        :return: linkage tree of feature redundancies; list of linkage trees
            for multi-target regressors or non-binary classifiers
        """

        feature_affinity_matrix = self.__interaction_projector.redundancy(
            symmetrical=True, absolute=False
        )
        assert (
            feature_affinity_matrix is not None
        ), ASSERTION__SHAP_INTERACTION_SUPPORTED

        return self.__linkages_from_affinity_matrices(
            feature_affinity_matrix=feature_affinity_matrix
        )

    @fitted_only
    def feature_association_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Calculate a linkage tree based on the :meth:`.feature_association_matrix`.

        The linkage tree can be used to render a dendrogram indicating clusters of
        associated features.

        In the case of multi-target regression and non-binary classification, returns
        a list of linkage trees per target or class.

        :return: linkage tree of feature associations; list of linkage trees
            for multi-target regressors or non-binary classifiers
        """

        assert self._shap_projector is not None, ASSERTION__INSPECTOR_IS_FITTED
        feature_affinity_matrix = self._shap_projector.association(
            absolute=False, symmetrical=True
        )
        assert (
            feature_affinity_matrix is not None
        ), ASSERTION__SHAP_INTERACTION_SUPPORTED

        return self.__linkages_from_affinity_matrices(
            feature_affinity_matrix=feature_affinity_matrix
        )

    @fitted_only
    def feature_interaction_matrix(self) -> Union[FloatMatrix, List[FloatMatrix]]:
        """
        Calculate relative shap interaction values for all feature pairings.

        Shap interactions quantify direct interactions between pairs of features.
        For a quantification of overall interaction (including indirect interactions
        across more than two features), see :meth:`.feature_synergy_matrix`.

        The relative values are normalised to add up to `1.0`, and each value ranges
        between `0.0` and `1.0`.

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

        The matrix returned by this method is a lower-triangular matrix

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
        a list with one matrix per output.

        :return: relative shap interaction values as a data frame of shape
            `(n_features, n_features)`; or a list of such data frames
        """

        n_features = len(self.feature_names)
        n_outputs = len(self.output_names)

        # get a feature interaction array with shape
        # (n_observations, n_outputs, n_features, n_features)
        # where the innermost feature x feature arrays are symmetrical
        im_matrix_per_observation_and_output: FloatArray = (
            # TODO missing proper handling for list of data frames
            self.shap_interaction_values()  # type: ignore
            .values.reshape((-1, n_features, n_outputs, n_features))
            .swapaxes(1, 2)
        )

        # get the observation weights with shape
        # (n_observations, n_outputs, n_features, n_features)
        weight: Optional[FloatArray]
        _weight_sr = self.sample_.weight
        if _weight_sr is not None:
            # if sample weights are defined, convert them to an array
            # and align the array with the dimensions of the feature interaction array
            weight = _weight_sr.values.reshape((-1, 1, 1, 1))
        else:
            weight = None

        # calculate the average interactions for each output and feature/feature
        # interaction, based on the standard deviation assuming a mean of 0.0.
        # The resulting matrix has shape (n_outputs, n_features, n_features)
        _interaction_squared = im_matrix_per_observation_and_output**2
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
        return self.__arrays_to_matrix(
            interaction_matrix, value_label="relative shap interaction"
        )

    @fitted_only
    def shap_plot_data(self) -> ShapPlotData:
        """
        Consolidate SHAP values and corresponding feature values from this inspector
        for use in SHAP plots offered by the
        `shap <https://shap.readthedocs.io/en/stable/>`__ package.

        The `shap` package provides functions for creating various SHAP plots.
        Most of these functions require:

        - one or more SHAP value matrices as a single `numpy` array,
          or a list of `numpy` arrays of shape `(n_observations, n_features)`
        - a feature matrix of shape `(n_observations, n_features)`, which can be
          provided as a data frame to preserve feature names

        This method provides this data inside a :class:`.ShapPlotData` object, plus:

        - the names of all outputs (i.e., the target names in case of regression,
          or the class names in case of classification)
        - corresponding target values as a series, or as a data frame in the case of
          multiple targets

        This method also ensures that the rows of all arrays, frames, and series are
        aligned, even if only a subset of the observations in the original sample was
        used to calculate SHAP values.

        Calculates mean shap values for each observation and feature, across all
        splits for which SHAP values were calculated.

        :return: consolidated SHAP and feature values for use shap plots
        """

        shap_values: Union[pd.DataFrame, List[pd.DataFrame]] = self.shap_values()

        output_names: List[str] = self.output_names
        shap_values_numpy: Union[FloatArray, List[FloatArray]]
        included_observations: pd.Index

        if len(output_names) > 1:
            shap_values_numpy = [s.values for s in shap_values]
            included_observations = shap_values[0].index
        else:
            shap_values = cast(pd.DataFrame, shap_values)
            shap_values_numpy = shap_values.values
            included_observations = shap_values.index

        sample: Sample = self.sample_.subsample(loc=included_observations)

        return ShapPlotData(
            shap_values=shap_values_numpy,
            sample=sample,
        )

    @property
    @abstractmethod
    def shap_calculator(self) -> ShapCalculator[Any]:
        """
        The SHAP calculator used by this inspector.
        """

    def __arrays_to_matrix(
        self, matrix: FloatArray, value_label: str
    ) -> Union[FloatMatrix, List[FloatMatrix]]:
        # transform a matrix of shape (n_outputs, n_features, n_features)
        # to a data frame

        feature_index = self.feature_names

        n_features = len(feature_index)
        assert matrix.shape == (len(self.output_names), n_features, n_features)

        # convert array to data frame(s) with features as row and column indices
        if len(matrix) == 1:
            return self.__array_to_matrix(
                matrix[0],
                feature_importance=self.feature_importance(),
                value_label=value_label,
            )
        else:
            return [
                self.__array_to_matrix(
                    m,
                    feature_importance=feature_importance,
                    value_label=f"{value_label} ({output_name})",
                )
                for m, (_, feature_importance), output_name in zip(
                    matrix, self.feature_importance().items(), self.output_names
                )
            ]

    def __feature_affinity_matrix(
        self,
        *,
        explainer_fn: Callable[..., FloatArray],
        absolute: bool,
        symmetrical: bool,
        clustered: bool,
    ) -> Union[FloatMatrix, List[FloatMatrix]]:
        affinity_matrices = explainer_fn(symmetrical=symmetrical, absolute=absolute)

        explainer: ShapProjector = cast(
            ShapProjector, cast(MethodType, explainer_fn).__self__
        )
        affinity_matrices_df: List[pd.DataFrame] = explainer.to_frames(
            affinity_matrices
        )

        if clustered:
            affinity_symmetrical = explainer_fn(symmetrical=True, absolute=False)
            assert (
                affinity_symmetrical is not None
            ), ASSERTION__SHAP_INTERACTION_SUPPORTED

            affinity_matrices_df = self.__sort_affinity_matrices(
                affinity_matrices=affinity_matrices_df,
                symmetrical_affinity_matrices=affinity_symmetrical,
            )

        return self.__isolate_single_frame(
            affinity_matrices_df, affinity_metric=explainer_fn.__name__
        )

    @staticmethod
    def __sort_affinity_matrices(
        affinity_matrices: List[pd.DataFrame],
        symmetrical_affinity_matrices: FloatArray,
    ) -> List[pd.DataFrame]:
        # abbreviate a very long function name to stay within the permitted line length
        fn_linkage = ModelInspector.__linkage_matrix_from_affinity_matrix_for_output

        return [
            (lambda feature_order: affinity_matrix.iloc[feature_order, feature_order])(
                feature_order=hierarchy.leaves_list(
                    Z=fn_linkage(feature_affinity_matrix=symmetrical_affinity_matrix)
                )
            )
            for affinity_matrix, symmetrical_affinity_matrix in zip(
                affinity_matrices, symmetrical_affinity_matrices
            )
        ]

    @staticmethod
    def __split_multi_output_df(
        multi_output_df: pd.DataFrame,
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        # Split a multi-output data frame into a list of single-output data frames.
        # Return single-output data frames as is.
        # Multi-output data frames are grouped by level 0 in the column index.
        if multi_output_df.columns.nlevels == 1:
            return multi_output_df
        else:
            return [
                multi_output_df.xs(key=output_name, axis=1, level=0, drop_level=True)
                for output_name in (
                    cast(pd.MultiIndex, multi_output_df.columns).levels[0]
                )
            ]

    def __linkages_from_affinity_matrices(
        self, feature_affinity_matrix: FloatArray
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
            return self.__linkage_tree_from_affinity_matrix_for_output(
                feature_affinity_matrix[0], feature_importance
            )

        else:
            feature_importance_iter: (
                Iterable[Tuple[Any, pd.Series]]
            ) = feature_importance.items()

            return [
                self.__linkage_tree_from_affinity_matrix_for_output(
                    feature_affinity_for_output,
                    feature_importance_for_output,
                )
                for feature_affinity_for_output, (
                    _,
                    feature_importance_for_output,
                ) in zip(feature_affinity_matrix, feature_importance_iter)
            ]

    @staticmethod
    def __linkage_tree_from_affinity_matrix_for_output(
        feature_affinity_matrix: FloatArray, feature_importance: pd.Series
    ) -> LinkageTree:
        # calculate the linkage tree from the given output in a feature distance
        # matrix;
        # matrix has shape (n_features, n_features) with values ranging from
        # (1 = closest, 0 = most distant)

        linkage_matrix: FloatArray = (
            ModelInspector.__linkage_matrix_from_affinity_matrix_for_output(
                feature_affinity_matrix
            )
        )

        # Feature labels and weights will be used as the leaves of the linkage tree.
        # Select only the features that appear in the distance matrix, and in the
        # correct order.

        # build and return the linkage tree
        return LinkageTree(
            scipy_linkage_matrix=linkage_matrix,
            leaf_names=feature_importance.index,
            leaf_weights=feature_importance.values,
            max_distance=1.0,
            distance_label="feature distance",
            leaf_label="feature",
            weight_label="feature importance",
        )

    @staticmethod
    def __linkage_matrix_from_affinity_matrix_for_output(
        feature_affinity_matrix: FloatArray,
    ) -> FloatArray:
        # calculate the linkage matrix from the given output in a feature distance
        # matrix;
        # matrix has shape (n_features, n_features) with values ranging from
        # (1 = closest, 0 = most distant)

        # compress the distance matrix (required by SciPy)
        distance_matrix = 1.0 - abs(feature_affinity_matrix)
        np.fill_diagonal(distance_matrix, 0.0)
        compressed_distance_matrix: FloatArray = distance.squareform(distance_matrix)

        # calculate the linkage matrix
        leaf_ordering: FloatArray = hierarchy.optimal_leaf_ordering(
            Z=hierarchy.linkage(y=compressed_distance_matrix, method="single"),
            y=compressed_distance_matrix,
        )

        # reverse the leaf ordering, so that larger values tend to end up on top
        leaf_ordering[:, [1, 0]] = leaf_ordering[:, [0, 1]]

        return leaf_ordering

    def _ensure_shap_interaction(self) -> None:
        if not self.shap_interaction:
            raise RuntimeError(
                "SHAP interaction values have not been calculated. "
                "Create an inspector with parameter 'shap_interaction=True' to "
                "enable calculations involving SHAP interaction values."
            )

    def __isolate_single_frame(
        self,
        frames: List[pd.DataFrame],
        affinity_metric: str,
    ) -> Union[FloatMatrix, List[FloatMatrix]]:
        feature_importance = self.feature_importance()

        if len(frames) == 1:
            assert isinstance(feature_importance, pd.Series)
            return self.__frame_to_matrix(
                frames[0],
                affinity_metric=affinity_metric,
                feature_importance=feature_importance,
            )
        else:
            return [
                self.__frame_to_matrix(
                    frame,
                    affinity_metric=affinity_metric,
                    feature_importance=frame_importance,
                    feature_importance_category=str(frame_name),
                )
                for frame, (frame_name, frame_importance) in zip(
                    frames, feature_importance.items()
                )
            ]

    @staticmethod
    def __array_to_matrix(
        a: npt.NDArray[T_Number],
        *,
        feature_importance: pd.Series,
        value_label: str,
    ) -> Matrix[T_Number]:
        return Matrix(
            a,
            names=(feature_importance.index, feature_importance.index),
            weights=(feature_importance, feature_importance),
            value_label=value_label,
            name_labels=("feature", "feature"),
        )

    @staticmethod
    def __frame_to_matrix(
        frame: pd.DataFrame,
        *,
        affinity_metric: str,
        feature_importance: pd.Series,
        feature_importance_category: Optional[str] = None,
    ) -> FloatMatrix:
        return Matrix.from_frame(
            frame,
            weights=(
                feature_importance.reindex(frame.index),
                feature_importance.reindex(frame.columns),
            ),
            value_label=(
                f"{affinity_metric} ({feature_importance_category})"
                if feature_importance_category
                else affinity_metric
            ),
            name_labels=("primary feature", "associated feature"),
        )

    @property
    def __interaction_projector(self) -> ShapInteractionVectorProjector:
        self._ensure_shap_interaction()
        return cast(ShapInteractionVectorProjector, self._shap_projector)


__tracker.validate()
