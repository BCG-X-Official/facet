"""
Implementation of :class:`.LearnerInspector`.
"""
import logging
import re
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union, cast

from pytools.api import AllTracker, inheritdoc, subsdoc, to_list

from ._explainer import ExactExplainerFactory, ExplainerFactory
from ._model_inspector import ModelInspector
from ._types import ModelFunction
from .shap import FunctionShapCalculator, ShapCalculator

log = logging.getLogger(__name__)

__all__ = [
    "FunctionInspector",
]


#
# Type variables
#

T_Function = TypeVar("T_Function", bound=ModelFunction)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@subsdoc(
    pattern=(
        r"(?m)"  # match multiline
        r"(^\s+)\.\. note::\s*"  # .. note:: at start of line
        r"(?:\1.*\n)+"  # followed by one or more indented lines
        r"(?:\1?\n)*"  # followed by zero or more blank lines
    ),
    replacement="",
)
@inheritdoc(match="""[see superclass]""")
class FunctionInspector(ModelInspector[T_Function], Generic[T_Function]):
    """[see superclass]"""

    #: The default explainer factory used by this inspector.
    DEFAULT_EXPLAINER_FACTORY: ExplainerFactory[ModelFunction] = ExactExplainerFactory()

    # the feature names of the model function
    _feature_names: List[str]

    def __init__(
        self,
        model: T_Function,
        *,
        feature_names: Sequence[str],
        explainer_factory: Optional[ExplainerFactory[T_Function]] = None,
        shap_interaction: bool = True,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param model: the model function to inspect, which takes a 2D array of
            feature values as input and returns a 1D array of output values
        :param feature_names: the names of the inputs to the model function
        :param explainer_factory: optional function that creates a shap Explainer
            (default: a :class:`.KernelExplainerFactory` instance; see
            :attr:`.DEFAULT_EXPLAINER_FACTORY`)
        """
        _explainer_factory: ExplainerFactory[T_Function]

        if explainer_factory:
            if not explainer_factory.explains_raw_output:
                raise ValueError(
                    "arg explainer_factory must be configured to explain raw output"
                )
            _explainer_factory = explainer_factory
        else:
            _explainer_factory = cast(
                ExplainerFactory[T_Function], self.DEFAULT_EXPLAINER_FACTORY
            )
            assert _explainer_factory.explains_raw_output

        if shap_interaction:
            if not _explainer_factory.supports_shap_interaction_values:
                log.warning(
                    "ignoring arg shap_interaction=True: "
                    f"explainers made by {_explainer_factory!r} do not support "
                    "SHAP interaction values"
                )
                shap_interaction = False

        super().__init__(
            model=model,
            shap_interaction=shap_interaction,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        self.model = model
        self._feature_names = to_list(
            feature_names, element_type=str, arg_name="feature_names"
        )
        self.explainer_factory: ExplainerFactory[T_Function] = _explainer_factory
        self._shap_calculator: Optional[ShapCalculator[Any]] = None

    __init__.__doc__ = str(__init__.__doc__) + re.sub(
        r"(?m)^\s*:param model:\s+.*$", "", str(ModelInspector.__init__.__doc__)
    )

    @property
    def feature_names(self) -> List[str]:
        """[see superclass]"""
        return self._feature_names

    @property
    def shap_calculator(self) -> ShapCalculator[Any]:
        """[see superclass]"""

        if self._shap_calculator is not None:
            return self._shap_calculator

        model = self.model

        shap_calculator = FunctionShapCalculator(
            model=model,
            explainer_factory=self.explainer_factory,
            interaction_values=self.shap_interaction,
            n_jobs=self.n_jobs,
            shared_memory=self.shared_memory,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose,
        )

        self._shap_calculator = shap_calculator
        return shap_calculator


__tracker.validate()