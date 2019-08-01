"""
data frame versions of scikit-learn transformers
"""

#
# To create the DF class stubs:
#
# - generate a list of all child classes of TransformerMixin in PyCharm using the
#   hierarchy view (^H)
# - remove all abstract base classes and non-sklearn classes from the list
# - unindent all lines
# - use replace with regular expressions
#   Find: (\w+)\([^\)]+\) \(([\w\.]+)\)
#   Replace: @_df_transformer\nclass $1DF($1, TransformerDF):\n    """\n    Wraps
#            :class:`$2.$1`;\n    accepts and returns data frames.\n    """
#            \n    pass\n\n
# - clean up imports; import only the module names not the individual classes

import logging
from functools import reduce
from typing import *

import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSSVD
from sklearn.decomposition import (
    DictionaryLearning,
    FactorAnalysis,
    FastICA,
    IncrementalPCA,
    KernelPCA,
    LatentDirichletAllocation,
    MiniBatchDictionaryLearning,
    MiniBatchSparsePCA,
    NMF,
    PCA,
    SparseCoder,
    SparsePCA,
    TruncatedSVD,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.feature_selection import (
    GenericUnivariateSelect,
    RFE,
    RFECV,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.kernel_approximation import (
    AdditiveChi2Sampler,
    Nystroem,
    RBFSampler,
    SkewedChi2Sampler,
)
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    KernelCenterer,
    LabelBinarizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    MultiLabelBinarizer,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from gamma.sklearndf import TransformerDF
from gamma.sklearndf._wrapper import df_estimator, TransformerWrapperDF
from gamma.sklearndf.transformation._wrapper import (
    BaseDimensionalityReductionWrapperDF,
    BaseMultipleInputsPerOutputTransformerWrapperDF,
    ColumnPreservingTransformerWrapperDF,
    ColumnSubsetTransformerWrapperDF,
    ComponentsDimensionalityReductionWrapperDF,
    FeatureSelectionWrapperDF,
    NComponentsDimensionalityReductionWrapperDF,
)

log = logging.getLogger(__name__)

# [sym for sym in dir(transformation) if sym.endswith("DF")]
__all__ = [
    "AdditiveChi2SamplerDF",
    "BernoulliRBMDF",
    "BinarizerDF",
    "ColumnTransformerDF",
    "DictVectorizerDF",
    "DictionaryLearningDF",
    "FactorAnalysisDF",
    "FastICADF",
    "FeatureAgglomerationDF",
    "FeatureHasherDF",
    "FunctionTransformerDF",
    "GaussianRandomProjectionDF",
    "GenericUnivariateSelectDF",
    "HashingVectorizerDF",
    "IncrementalPCADF",
    "IsomapDF",
    "IterativeImputerDF",
    "KBinsDiscretizerDF",
    "KernelCentererDF",
    "KernelPCADF",
    "LabelBinarizerDF",
    "LabelEncoderDF",
    "LatentDirichletAllocationDF",
    "LinearDiscriminantAnalysisDF",
    "LocallyLinearEmbeddingDF",
    "MaxAbsScalerDF",
    "MinMaxScalerDF",
    "MiniBatchDictionaryLearningDF",
    "MiniBatchSparsePCADF",
    "MissingIndicatorDF",
    "MultiLabelBinarizerDF",
    "NMFDF",
    "NeighborhoodComponentsAnalysisDF",
    "NormalizerDF",
    "NystroemDF",
    "OneHotEncoderDF",
    "OrdinalEncoderDF",
    "PCADF",
    "PLSSVDDF",
    "PolynomialFeaturesDF",
    "PowerTransformerDF",
    "QuantileTransformerDF",
    "RBFSamplerDF",
    "RFECVDF",
    "RFEDF",
    "RobustScalerDF",
    "SelectFdrDF",
    "SelectFprDF",
    "SelectFromModelDF",
    "SelectFweDF",
    "SelectKBestDF",
    "SelectPercentileDF",
    "SimpleImputerDF",
    "SkewedChi2SamplerDF",
    "SparseCoderDF",
    "SparsePCADF",
    "SparseRandomProjectionDF",
    "StandardScalerDF",
    "TfidfTransformerDF",
    "TransformerDF",
    "TransformerWrapperDF",
    "TruncatedSVDDF",
    "VarianceThresholdDF",
]


#
# cluster
#


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class FeatureAgglomerationDF(FeatureAgglomeration, TransformerDF):
    """
    Wraps :class:`sklearn.cluster.FeatureAgglomeration`;
    accepts and returns data frames.
    """

    pass


#
# compose
#


class ColumnTransformerDF(TransformerWrapperDF[ColumnTransformer]):
    """
    Wrap :class:`sklearn.compose.ColumnTransformer` and return a DataFrame.

    Like :class:`~sklearn.compose.ColumnTransformer`, it has a ``transformers``
    parameter
    (``None`` by default) which is a list of tuple of the form (name, transformer,
    column(s)),
    but here all the transformers must be of type
    :class:`~yieldengine.df.transform.TransformerDF`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # noinspection PyTypeChecker
        column_transformer = self.delegate_estimator

        if column_transformer.remainder != "drop":
            raise ValueError(
                f"arg column_transformer with unsupported remainder attribute "
                f"({column_transformer.remainder})"
            )

        if not (
            all(
                [
                    isinstance(transformer, TransformerDF)
                    for _, transformer, _ in column_transformer.transformers
                ]
            )
        ):
            raise ValueError(
                "arg column_transformer must only contain instances of "
                f"{TransformerDF.__name__}"
            )

        self._columnTransformer = column_transformer

    @classmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> ColumnTransformer:
        return ColumnTransformer(*args, **kwargs)

    def _get_columns_original(self) -> pd.Series:
        """
        Return the series mapping output column names to original columns names.

        :return: the series with index the column names of the output dataframe and
        values the corresponding input column names.
        """
        return reduce(
            lambda x, y: x.append(y),
            (
                df_transformer.columns_original
                for df_transformer in self._inner_transformers()
            ),
        )

    def _inner_transformers(self) -> Iterable[TransformerWrapperDF]:
        return (
            df_transformer
            for _, df_transformer, columns in self.delegate_estimator.transformers_
            if len(columns) > 0
            if df_transformer != "drop"
        )


#
# cross_decomposition
#


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class PLSSVDDF(PLSSVD, TransformerDF):
    """
    Wraps :class:`sklearn.cross_decomposition.pls_.PLSSVD`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class FeatureHasherDF(FeatureHasher, TransformerDF):
    """
    Wraps :class:`sklearn.feature_extraction.FeatureHasher`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class DictVectorizerDF(DictVectorizer, TransformerDF):
    """
    Wraps :class:`sklearn.feature_extraction.dict_vectorizer.DictVectorizer`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class HashingVectorizerDF(HashingVectorizer, TransformerDF):
    """
    Wraps :class:`sklearn.feature_extraction.text.HashingVectorizer`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class TfidfTransformerDF(TfidfTransformer, TransformerDF):
    """
    Wraps :class:`sklearn.feature_extraction.text.TfidfTransformer`;
    accepts and returns data frames.
    """

    pass


#
# impute
#
class SimpleImputerDF(ColumnSubsetTransformerWrapperDF[SimpleImputer]):
    """
    Impute missing values with data frames as input and output.

    Wrap around :class:`impute.SimpleImputer`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`impute.SimpleImputer`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> SimpleImputer:
        return SimpleImputer(*args, **kwargs)

    def _get_columns_out(self) -> pd.Index:
        stats = self.delegate_estimator.statistics_
        if issubclass(stats.dtype.type, float):
            nan_mask = np.isnan(stats)
        else:
            nan_mask = [
                x is None or (isinstance(x, float) and np.isnan(x)) for x in stats
            ]
        return self.columns_in.delete(np.argwhere(nan_mask))


class MissingIndicatorDF(TransformerWrapperDF[MissingIndicator]):
    """
    Indicate missing values with dataframes as input and output.

    Wrap :class:`impute.MissingIndicatorDF`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`impute.MissingIndicator`.

    The parameters are the same as the one passed to
    :class:`impute.MissingIndicator`.
    """

    def __init__(
        self,
        missing_values=np.nan,
        features="missing-only",
        sparse="auto",
        error_on_new=True,
        **kwargs,
    ) -> None:
        super().__init__(
            missing_values=missing_values,
            features=features,
            sparse=sparse,
            error_on_new=error_on_new,
            **kwargs,
        )

    @classmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> MissingIndicator:
        return MissingIndicator(*args, **kwargs)

    def _get_columns_original(self) -> pd.Series:
        columns_original: np.ndarray = self.columns_in[
            self.delegate_estimator.features_
        ].values
        columns_out = pd.Index([f"{name}__missing" for name in columns_original])
        return pd.Series(index=columns_out, data=columns_original)


# noinspection PyProtectedMember
@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class IterativeImputerDF(IterativeImputer, TransformerDF):
    """
    Wraps :class:`sklearn.impute.IterativeImputer`;
    accepts and returns data frames.
    """

    pass


class IsomapDF(BaseDimensionalityReductionWrapperDF[Isomap]):
    """
    Wraps :class:`sklearn.manifold.Isomap`;
    accepts and returns data frames.
    """

    @property
    def _n_components(self) -> int:
        return self.delegate_estimator.embedding_.shape[1]

    @classmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> Isomap:
        return Isomap(*args, **kwargs)


#
# neighbors
#


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class NeighborhoodComponentsAnalysisDF(NeighborhoodComponentsAnalysis, TransformerDF):
    """
    Wraps :class:`sklearn.neighbors.NeighborhoodComponentsAnalysis`;
    accepts and returns data frames.
    """

    pass


#
# neural_network
#


#
# preprocessing
#
@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class MinMaxScalerDF(MinMaxScaler, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.MinMaxScaler`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class StandardScalerDF(StandardScaler, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.StandardScaler`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class MaxAbsScalerDF(MaxAbsScaler, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.MaxAbsScaler`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class RobustScalerDF(RobustScaler, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.RobustScaler`;
    accepts and returns data frames.
    """

    pass


class PolynomialFeaturesDF(
    BaseMultipleInputsPerOutputTransformerWrapperDF[PolynomialFeatures]
):
    """
    Wraps :class:`sklearn.preprocessing.PolynomialFeatures`;
    accepts and returns data frames.
    """

    def _get_columns_out(self) -> pd.Index:
        # todo: deal case when columns_in is not an iterable of strings
        return pd.Index(
            data=self.delegate_estimator.get_feature_names(
                input_features=self.columns_in
            )
        )

    @classmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> PolynomialFeatures:
        return PolynomialFeatures(*args, **kwargs)


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class NormalizerDF(Normalizer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.Normalizer`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class BinarizerDF(Binarizer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.Binarizer`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class KernelCentererDF(KernelCenterer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.KernelCenterer`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class QuantileTransformerDF(QuantileTransformer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.QuantileTransformer`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class PowerTransformerDF(PowerTransformer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.PowerTransformer`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class FunctionTransformerDF(FunctionTransformer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.FunctionTransformer`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class LabelEncoderDF(LabelEncoder, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.LabelEncoder`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class LabelBinarizerDF(LabelBinarizer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.LabelBinarizer`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class MultiLabelBinarizerDF(MultiLabelBinarizer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.MultiLabelBinarizer`;
    accepts and returns data frames.
    """

    pass


class OneHotEncoderDF(TransformerWrapperDF[OneHotEncoder]):
    """
    One-hot encoder with dataframes as input and output.

    Wrap around :class:`preprocessing.OneHotEncoder`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`preprocessing.OneHotEncoder`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.delegate_estimator.sparse:
            raise ValueError(
                "sparse matrices not supported; set OneHotEncoder.sparse to False"
            )

    @classmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> OneHotEncoder:
        return OneHotEncoder(*args, **kwargs)

    def _get_columns_original(self) -> pd.Series:
        """
        Return the series mapping output column names to original columns names.

        :return: the series with index the column names of the output dataframe and
        values the corresponding input column names.
        """
        return pd.Series(
            index=pd.Index(self.delegate_estimator.get_feature_names(self.columns_in)),
            data=[
                column_original
                for column_original, category in zip(
                    self.columns_in, self.delegate_estimator.categories_
                )
                for _ in category
            ],
        )


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class OrdinalEncoderDF(OrdinalEncoder, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.OrdinalEncoder`;
    accepts and returns data frames.
    """

    pass


class KBinsDiscretizerDF(TransformerWrapperDF[KBinsDiscretizer]):
    """
    Wrap :class:`sklearn.preprocessing.KBinsDiscretizer`;
    accepts and returns dataframes.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.delegate_estimator.encode == "onehot":
            raise AttributeError(
                'property encode="onehot" is not supported due to sparse matrices;'
                'consider using "onehot-dense" instead'
            )

    @classmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> KBinsDiscretizer:
        return KBinsDiscretizer(*args, **kwargs)

    def _get_columns_original(self) -> pd.Series:
        """
        Return the series mapping output column names to original columns names.

        :return: the series with index the column names of the output dataframe and
        values the corresponding input column names.
        """
        if self.delegate_estimator.encode == "onehot-dense":
            n_bins_per_feature = self.delegate_estimator.n_bins_
            columns_in, columns_out = zip(
                *(
                    (feature_name, f"{feature_name}_bin_{bin_index}")
                    for feature_name, n_bins in zip(self.columns_in, n_bins_per_feature)
                    for bin_index in range(n_bins)
                )
            )

            return pd.Series(index=columns_out, data=columns_in)

        elif self.delegate_estimator.encode == "ordinal":
            return pd.Series(
                index=self.columns_in.astype(str) + "_bin", data=self.columns_in
            )
        else:
            raise AttributeError(
                f"unexpected value for property encode={self.delegate_estimator.encode}"
            )


#
# Transformers which have a components_ attribute
# Implemented through ComponentsDimensionalityReductionWrapperDF
#


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class BernoulliRBMDF(BernoulliRBM, TransformerDF):
    """
    Wraps :class:`sklearn.neural_network.BernoulliRBM`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class DictionaryLearningDF(DictionaryLearning, TransformerDF):
    """
    Wraps :class:`decomposition.dict_learning.DictionaryLearning`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class FactorAnalysisDF(FactorAnalysis, TransformerDF):
    """
    Wraps :class:`decomposition.factor_analysis.FactorAnalysis`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class FastICADF(FastICA, TransformerDF):
    """
    Wraps :class:`decomposition.fastica_.FastICA`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class GaussianRandomProjectionDF(GaussianRandomProjection, TransformerDF):
    """
    Wraps :class:`sklearn.random_projection.GaussianRandomProjection`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class IncrementalPCADF(IncrementalPCA, TransformerDF):
    """
    Wraps :class:`decomposition.incremental_pca.IncrementalPCA`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class LatentDirichletAllocationDF(LatentDirichletAllocation, TransformerDF):
    """
    Wraps :class:`decomposition.online_lda.LatentDirichletAllocation`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class MiniBatchDictionaryLearningDF(MiniBatchDictionaryLearning, TransformerDF):
    """
    Wraps :class:`decomposition.dict_learning.MiniBatchDictionaryLearning`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class MiniBatchSparsePCADF(MiniBatchSparsePCA, TransformerDF):
    """
    Wraps :class:`decomposition.sparse_pca.MiniBatchSparsePCA`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class NMFDF(NMF, TransformerDF):
    """
    Wraps :class:`decomposition.NMF`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class PCADF(PCA, TransformerDF):
    """
    Wraps :class:`decomposition.pca.PCA`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class SparseCoderDF(SparseCoder, TransformerDF):
    """
    Wraps :class:`decomposition.dict_learning.SparseCoder`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class SparsePCADF(SparsePCA, TransformerDF):
    """
    Wraps :class:`decomposition.sparse_pca.SparsePCA`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class SparseRandomProjectionDF(SparseRandomProjection, TransformerDF):
    """
    Wraps :class:`sklearn.random_projection.SparseRandomProjection`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=ComponentsDimensionalityReductionWrapperDF)
class TruncatedSVDDF(TruncatedSVD, TransformerDF):
    """
    Wraps :class:`decomposition.truncated_svd.TruncatedSVD`;
    accepts and returns data frames.
    """

    pass


#
# Transformer which have an n_components attribute
# Implemented through NComponentsDimensionalityReductionWrapperDF
#


@df_estimator(df_wrapper_type=NComponentsDimensionalityReductionWrapperDF)
class AdditiveChi2SamplerDF(AdditiveChi2Sampler, TransformerDF):
    """
    Wraps :class:`sklearn.kernel_approximation.AdditiveChi2Sampler`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=NComponentsDimensionalityReductionWrapperDF)
class KernelPCADF(KernelPCA, TransformerDF):
    """
    Wraps :class:`decomposition.kernel_pca.KernelPCA`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=NComponentsDimensionalityReductionWrapperDF)
class LinearDiscriminantAnalysisDF(LinearDiscriminantAnalysis, TransformerDF):
    """
    Wraps :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=NComponentsDimensionalityReductionWrapperDF)
class LocallyLinearEmbeddingDF(LocallyLinearEmbedding, TransformerDF):
    """
    Wraps :class:`sklearn.manifold.LocallyLinearEmbedding`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=NComponentsDimensionalityReductionWrapperDF)
class NystroemDF(Nystroem, TransformerDF):
    """
    Wraps :class:`sklearn.kernel_approximation.Nystroem`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=NComponentsDimensionalityReductionWrapperDF)
class RBFSamplerDF(RBFSampler, TransformerDF):
    """
    Wraps :class:`sklearn.kernel_approximation.RBFSampler`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=NComponentsDimensionalityReductionWrapperDF)
class SkewedChi2SamplerDF(SkewedChi2Sampler, TransformerDF):
    """
    Wraps :class:`sklearn.kernel_approximation.SkewedChi2Sampler`;
    accepts and returns data frames.
    """

    pass


#
# feature_selection
#
#
# Transformer which have an get_support method
# Implemented through FeatureSelectionWrapperDF
#


@df_estimator(df_wrapper_type=FeatureSelectionWrapperDF)
class VarianceThresholdDF(VarianceThreshold, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.VarianceThreshold`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=FeatureSelectionWrapperDF)
class RFEDF(RFE, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.RFE`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=FeatureSelectionWrapperDF)
class RFECVDF(RFECV, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.RFECV`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=FeatureSelectionWrapperDF)
class SelectFromModelDF(SelectFromModel, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectFromModel`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=FeatureSelectionWrapperDF)
class SelectPercentileDF(SelectPercentile, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectPercentile`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=FeatureSelectionWrapperDF)
class SelectKBestDF(SelectKBest, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectKBest`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=FeatureSelectionWrapperDF)
class SelectFprDF(SelectFpr, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectFpr`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=FeatureSelectionWrapperDF)
class SelectFdrDF(SelectFdr, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectFdr`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=FeatureSelectionWrapperDF)
class SelectFweDF(SelectFwe, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectFwe`;
    accepts and returns data frames.
    """

    pass


@df_estimator(df_wrapper_type=FeatureSelectionWrapperDF)
class GenericUnivariateSelectDF(GenericUnivariateSelect, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.GenericUnivariateSelect`;
    accepts and returns data frames.
    """

    pass
