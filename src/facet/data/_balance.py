from typing import Any, Dict, Union
from facet.data import Sample

__all__ = ["SampleBalancer"]


#
# Ensure all symbols introduced below are included in __all__
#
from pytools.api import AllTracker

__tracker = AllTracker(globals())


class SampleBalancer:
    """
    Balances the target distribution of a :class:`.Sample`, by either over- or
    under-sampling its observations.
    """

    def __init__(
        self,
        *,
        class_ratio: Union[float, Dict[Any, float]],
        bins: Union[str, int],
        downsample: bool = True,
        max_observations: int = None,
    ):
        """
        :param class_ratio: The desired target class ratio after balancing, either
            indicating the maximum ratio between the minority class and any other
            class as a positive scalar in range ``]0,1]`` or indicating the target ratio
            among multiple classes as a dictionary mapping class labels to scalars.
        :param bins: either ``"labels"`` to treat target variables as class labels,
            or an integer greater than 1 indicating the number of equally-sized ranges
            of target values to use as bins
        :param downsample: boolean parameter, whether majority class should be
            downsampled, or minority class should be upsampled
        :param max_observations: upper limit for the resulting sample size (optional)
        """

        if isinstance(class_ratio, float):
            if not 0 < class_ratio <= 1:
                raise ValueError(f"'class_ratio' not in range ]0,1]")
        elif isinstance(class_ratio, Dict):
            if not bins == "labels":
                raise ValueError(
                    f"If Dict passed for 'class_ratio', then bins='labels' is required."
                )
        else:
            raise TypeError(f"Unsupported type '{type(class_ratio)}' for class_ratio.")

        if max_observations is not None:
            if not isinstance(max_observations, int) or max_observations <= 1:
                raise ValueError(f"'max_observations' needs to be an integer >1 .")

        self._class_ratio = class_ratio
        self._bins = bins
        self._downsample = downsample
        self._max_observations = max_observations

    def balance(self, sample: Sample) -> Sample:
        """
        Balance the sample by over- or undersampling observations.
        :param sample: the sample to balance
        :return: the balanced sample
        """
        pass


__tracker.validate()
