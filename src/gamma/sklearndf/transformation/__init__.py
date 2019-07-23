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
import sklearn.feature_extraction
from sklearn import (
    cluster,
    compose,
    cross_decomposition,
    decomposition,
    discriminant_analysis,
    feature_extraction,
    feature_selection,
    impute,
    kernel_approximation,
    manifold,
    neighbors,
    neural_network,
    pipeline,
    preprocessing,
    random_projection,
)
from sklearn.impute import _iterative

from gamma.sklearndf import T_Transformer, TransformerDF
from gamma.sklearndf._wrapper import (
    df_estimator,
    PersistentColumnTransformerWrapperDF,
    PersistentNamingTransformerWrapperDF,
    TransformerWrapperDF,
)

log = logging.getLogger(__name__)


#
# decorator for wrapping the sklearn transformer classes
#


def _df_transformer(
    delegate_transformer: Type[T_Transformer]
) -> Type[TransformerWrapperDF[T_Transformer]]:
    return cast(
        Type[TransformerWrapperDF[T_Transformer]],
        df_estimator(
            delegate_estimator=delegate_transformer,
            df_estimator_type=PersistentColumnTransformerWrapperDF,
        ),
    )


#
# cluster
#


@_df_transformer
class FeatureAgglomerationDF(cluster.FeatureAgglomeration, TransformerDF):
    """
    Wraps :class:`sklearn.cluster.FeatureAgglomeration`;
    accepts and returns data frames.
    """

    pass


#
# compose
#


class ColumnTransformerDF(TransformerWrapperDF[compose.ColumnTransformer]):
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
        column_transformer = self.base_transformer

        if column_transformer.remainder != "drop":
            raise ValueError(
                f"arg column_transformer with unsupported remainder attribute "
                f"({column_transformer.remainder})"
            )

        if not (
            all(
                [
                    isinstance(transformer, TransformerWrapperDF)
                    for _, transformer, _ in column_transformer.transformers
                ]
            )
        ):
            raise ValueError(
                "arg column_transformer must only contain instances of "
                "TransformerWrapperDF"
            )

        self._columnTransformer = column_transformer

    @classmethod
    def _make_delegate_estimator(cls, **kwargs) -> compose.ColumnTransformer:
        return compose.ColumnTransformer(**kwargs)

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
            for _, df_transformer, columns in self.base_transformer.transformers_
            if len(columns) > 0
            if df_transformer != "drop"
        )


#
# cross_decomposition
#


@_df_transformer
class PLSSVDDF(cross_decomposition.PLSSVD, TransformerDF):
    """
    Wraps :class:`sklearn.cross_decomposition.pls_.PLSSVD`;
    accepts and returns data frames.
    """

    pass


#
# decomposition
#


@_df_transformer
class LatentDirichletAllocationDF(
    decomposition.LatentDirichletAllocation, TransformerDF
):
    """
    Wraps :class:`decomposition.online_lda.LatentDirichletAllocation`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class FactorAnalysisDF(decomposition.FactorAnalysis, TransformerDF):
    """
    Wraps :class:`decomposition.factor_analysis.FactorAnalysis`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class KernelPCADF(decomposition.KernelPCA, TransformerDF):
    """
    Wraps :class:`decomposition.kernel_pca.KernelPCA`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class TruncatedSVDDF(decomposition.TruncatedSVD, TransformerDF):
    """
    Wraps :class:`decomposition.truncated_svd.TruncatedSVD`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class FastICADF(decomposition.FastICA, TransformerDF):
    """
    Wraps :class:`decomposition.fastica_.FastICA`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class SparseCoderDF(decomposition.SparseCoder, TransformerDF):
    """
    Wraps :class:`decomposition.dict_learning.SparseCoder`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class DictionaryLearningDF(decomposition.DictionaryLearning, TransformerDF):
    """
    Wraps :class:`decomposition.dict_learning.DictionaryLearning`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class MiniBatchDictionaryLearningDF(
    decomposition.MiniBatchDictionaryLearning, TransformerDF
):
    """
    Wraps :class:`decomposition.dict_learning.MiniBatchDictionaryLearning`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class IncrementalPCADF(decomposition.IncrementalPCA, TransformerDF):
    """
    Wraps :class:`decomposition.incremental_pca.IncrementalPCA`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class PCADF(decomposition.PCA, TransformerDF):
    """
    Wraps :class:`decomposition.pca.PCA`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class SparsePCADF(decomposition.SparsePCA, TransformerDF):
    """
    Wraps :class:`decomposition.sparse_pca.SparsePCA`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class MiniBatchSparsePCADF(decomposition.MiniBatchSparsePCA, TransformerDF):
    """
    Wraps :class:`decomposition.sparse_pca.MiniBatchSparsePCA`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class NMFDF(decomposition.NMF, TransformerDF):
    """
    Wraps :class:`decomposition.NMF`;
    accepts and returns data frames.
    """

    pass


#
# discriminant_analysis
#


@_df_transformer
class LinearDiscriminantAnalysisDF(
    discriminant_analysis.LinearDiscriminantAnalysis, TransformerDF
):
    """
    Wraps :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`;
    accepts and returns data frames.
    """

    pass


#
# feature_extraction
#


@_df_transformer
class FeatureHasherDF(feature_extraction.FeatureHasher, TransformerDF):
    """
    Wraps :class:`sklearn.feature_extraction.FeatureHasher`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class DictVectorizerDF(sklearn.feature_extraction.DictVectorizer, TransformerDF):
    """
    Wraps :class:`sklearn.feature_extraction.dict_vectorizer.DictVectorizer`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class HashingVectorizerDF(feature_extraction.text.HashingVectorizer, TransformerDF):
    """
    Wraps :class:`sklearn.feature_extraction.text.HashingVectorizer`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class TfidfTransformerDF(feature_extraction.text.TfidfTransformer, TransformerDF):
    """
    Wraps :class:`sklearn.feature_extraction.text.TfidfTransformer`;
    accepts and returns data frames.
    """

    pass


#
# feature_selection
#
@_df_transformer
class VarianceThresholdDF(feature_selection.VarianceThreshold, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.VarianceThreshold`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class RFEDF(feature_selection.RFE, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.RFE`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class RFECVDF(feature_selection.RFECV, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.RFECV`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class SelectFromModelDF(feature_selection.SelectFromModel, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectFromModel`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class SelectPercentileDF(feature_selection.SelectPercentile, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectPercentile`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class SelectKBestDF(feature_selection.SelectKBest, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectKBest`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class SelectFprDF(feature_selection.SelectFpr, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectFpr`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class SelectFdrDF(feature_selection.SelectFdr, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectFdr`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class SelectFweDF(feature_selection.SelectFwe, TransformerDF):
    """
    Wraps :class:`sklearn.feature_selection.SelectFwe`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class GenericUnivariateSelectDF(
    feature_selection.GenericUnivariateSelect, TransformerDF
):
    """
    Wraps :class:`sklearn.feature_selection.GenericUnivariateSelect`;
    accepts and returns data frames.
    """

    pass


#
# impute
#
class SimpleImputerDF(PersistentNamingTransformerWrapperDF[impute.SimpleImputer]):
    """
    Impute missing values with dataframes as input and output.

    Wrap around :class:`impute.SimpleImputer`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`impute.SimpleImputer`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_delegate_estimator(cls, **kwargs) -> impute.SimpleImputer:
        return impute.SimpleImputer(**kwargs)

    def _get_columns_out(self) -> pd.Index:
        stats = self.base_transformer.statistics_
        if issubclass(stats.dtype.type, float):
            nan_mask = np.isnan(stats)
        else:
            nan_mask = [
                x is None or (isinstance(x, float) and np.isnan(x)) for x in stats
            ]
        return self.columns_in.delete(np.argwhere(nan_mask))


class MissingIndicatorDF(TransformerWrapperDF[impute.MissingIndicator]):
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
    def _make_delegate_estimator(cls, **kwargs) -> impute.MissingIndicator:
        return impute.MissingIndicator(**kwargs)

    def _get_columns_original(self) -> pd.Series:
        columns_original: np.ndarray = self.columns_in[
            self.base_transformer.features_
        ].values
        columns_out = pd.Index([f"{name}__missing" for name in columns_original])
        return pd.Series(index=columns_out, data=columns_original)


# noinspection PyProtectedMember
@_df_transformer
class IterativeImputerDF(impute._iterative.IterativeImputer, TransformerDF):
    """
    Wraps :class:`sklearn.impute.IterativeImputer`;
    accepts and returns data frames.
    """

    pass


#
# kernel_approximation
#
@_df_transformer
class RBFSamplerDF(kernel_approximation.RBFSampler, TransformerDF):
    """
    Wraps :class:`sklearn.kernel_approximation.RBFSampler`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class SkewedChi2SamplerDF(kernel_approximation.SkewedChi2Sampler, TransformerDF):
    """
    Wraps :class:`sklearn.kernel_approximation.SkewedChi2Sampler`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class AdditiveChi2SamplerDF(kernel_approximation.AdditiveChi2Sampler, TransformerDF):
    """
    Wraps :class:`sklearn.kernel_approximation.AdditiveChi2Sampler`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class NystroemDF(kernel_approximation.Nystroem, TransformerDF):
    """
    Wraps :class:`sklearn.kernel_approximation.Nystroem`;
    accepts and returns data frames.
    """

    pass


#
# manifold
#


@_df_transformer
class LocallyLinearEmbeddingDF(manifold.LocallyLinearEmbedding, TransformerDF):
    """
    Wraps :class:`sklearn.manifold.LocallyLinearEmbedding`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class IsomapDF(manifold.Isomap, TransformerDF):
    """
    Wraps :class:`sklearn.manifold.Isomap`;
    accepts and returns data frames.
    """

    pass


#
# neighbors
#


@_df_transformer
class NeighborhoodComponentsAnalysisDF(
    neighbors.NeighborhoodComponentsAnalysis, TransformerDF
):
    """
    Wraps :class:`sklearn.neighbors.NeighborhoodComponentsAnalysis`;
    accepts and returns data frames.
    """

    pass


#
# neural_network
#
@_df_transformer
class BernoulliRBMDF(neural_network.BernoulliRBM, TransformerDF):
    """
    Wraps :class:`sklearn.neural_network.BernoulliRBM`;
    accepts and returns data frames.
    """

    pass


#
# pipeline
#


@_df_transformer
class FeatureUnionDF(pipeline.FeatureUnion, TransformerDF):
    """
    Wraps :class:`sklearn.pipeline.FeatureUnion`;
    accepts and returns data frames.
    """

    pass


#
# preprocessing
#
@_df_transformer
class MinMaxScalerDF(preprocessing.MinMaxScaler, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.MinMaxScaler`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class StandardScalerDF(preprocessing.StandardScaler, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.StandardScaler`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class MaxAbsScalerDF(preprocessing.MaxAbsScaler, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.MaxAbsScaler`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class RobustScalerDF(preprocessing.RobustScaler, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.RobustScaler`;
    accepts and returns data frames.
    """

    pass


class PolynomialFeaturesDF(
    PersistentNamingTransformerWrapperDF[preprocessing.PolynomialFeatures]
):
    """
    Wraps :class:`sklearn.preprocessing.PolynomialFeatures`;
    accepts and returns data frames.
    """

    def _get_columns_out(self) -> pd.Index:
        return pd.Index(
            data=self.base_transformer.get_feature_names(input_features=self.columns_in)
        )

    @classmethod
    def _make_delegate_estimator(cls, **kwargs) -> preprocessing.PolynomialFeatures:
        return preprocessing.PolynomialFeatures(**kwargs)


@_df_transformer
class NormalizerDF(preprocessing.Normalizer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.Normalizer`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class BinarizerDF(preprocessing.Binarizer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.Binarizer`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class KernelCentererDF(preprocessing.KernelCenterer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.KernelCenterer`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class QuantileTransformerDF(preprocessing.QuantileTransformer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.QuantileTransformer`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class PowerTransformerDF(preprocessing.PowerTransformer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.PowerTransformer`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class FunctionTransformerDF(preprocessing.FunctionTransformer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.FunctionTransformer`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class LabelEncoderDF(preprocessing.LabelEncoder, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.LabelEncoder`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class LabelBinarizerDF(preprocessing.LabelBinarizer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.LabelBinarizer`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class MultiLabelBinarizerDF(preprocessing.MultiLabelBinarizer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.MultiLabelBinarizer`;
    accepts and returns data frames.
    """

    pass


class OneHotEncoderDF(TransformerWrapperDF[preprocessing.OneHotEncoder]):
    """
    One-hot encoder with dataframes as input and output.

    Wrap around :class:`preprocessing.OneHotEncoder`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`preprocessing.OneHotEncoder`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.base_transformer.sparse:
            raise ValueError(
                "sparse matrices not supported; set OneHotEncoder.sparse to False"
            )

    @classmethod
    def _make_delegate_estimator(cls, **kwargs) -> preprocessing.OneHotEncoder:
        return preprocessing.OneHotEncoder(**kwargs)

    def _get_columns_original(self) -> pd.Series:
        """
        Return the series mapping output column names to original columns names.

        :return: the series with index the column names of the output dataframe and
        values the corresponding input column names.
        """
        return pd.Series(
            index=pd.Index(self.base_transformer.get_feature_names(self.columns_in)),
            data=[
                column_original
                for column_original, category in zip(
                    self.columns_in, self.base_transformer.categories_
                )
                for _ in category
            ],
        )


@_df_transformer
class OrdinalEncoderDF(preprocessing.OrdinalEncoder, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.OrdinalEncoder`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class KBinsDiscretizerDF(preprocessing.KBinsDiscretizer, TransformerDF):
    """
    Wraps :class:`sklearn.preprocessing.KBinsDiscretizer`;
    accepts and returns data frames.
    """

    pass


#
# random_projection
#
@_df_transformer
class GaussianRandomProjectionDF(
    random_projection.GaussianRandomProjection, TransformerDF
):
    """
    Wraps :class:`sklearn.random_projection.GaussianRandomProjection`;
    accepts and returns data frames.
    """

    pass


@_df_transformer
class SparseRandomProjectionDF(random_projection.SparseRandomProjection, TransformerDF):
    """
    Wraps :class:`sklearn.random_projection.SparseRandomProjection`;
    accepts and returns data frames.
    """

    pass
