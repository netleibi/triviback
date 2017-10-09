=======================================
triviback --- the trivial backup system
=======================================

What it is
----------

`triviback` is a simple, proof-of-concept backup system built on top of `https://github.com/netleibi/seccs`_.
Its purpose is to evaluate `sec-cs` and play around with some data deduplication concepts in realistic scenarios.
Please do not use it beyond research purposes. Its code is NOT production-ready.

Triviback has been developed as part of the work [LS17]_ at CISPA, Saarland University.

Testing
-------

`triviback` uses tox for testing, so simply run:

::

   $ tox

References:
    .. [LS17] Dominik Leibenger and Christoph Sorge (2017). triviback: A Storage-Efficient Secure Backup System.
       In 42nd IEEE Conference on Local Computer Networks, LCN 2017, Singapore, October 9-12, 2017, 2017.
