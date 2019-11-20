"""
Partitioner for collections of categorical or numerical values.

:class:`Partitioner` partitions a set of values
into finitely many partitions (synonym of buckets).
:meth:`~Partitioner.frequencies` returns  an iterable of the number of
values in the different partitions, :meth:`~Partitioner.partitions`
returns a list of central value in each partition,
and :attr:`~Partitioner.n_partitions` is the number of partitions.

:class:`ContinuousRangePartitioner` is adapted set of floats

:class:`IntegerRangePartitioner` is adapted to sets of  integers; the bounds of the
partitions are integers

:class:`CategoryPartitioner` is adapted to categorical sets
"""
from ._partition import *
