"""
Partitioning for collections of categorical or numerical values.

:class:`Partitioning` partitions a set of values
into finitely many partitions (synonym of buckets).
:meth:`~Partitioning.frequencies` returns  an iterable of the number of
values in the different partitions, :meth:`~Partitioning.partitions`
returns a list of central value in each partition,
and :attr:`~Partitioning.n_partitions` is the number of partitions.

:class:`ContinuousRangePartitioning` is adapted set of floats

:class:`IntegerRangePartitioning` is adapted to sets of  integers; the bounds of the
partitions are integers

:class:`CategoryPartitioning` is adapted to categorical sets
"""
from ._partition import *

__all__ = [member for member in _partition.__all__ if not member.startswith("Base")]
