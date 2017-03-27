Credibility Allocation
======================

Simulate credibility allocation using the `Google Cluster Data`_.

.. _`Google Cluster Data`: https://github.com/google/cluster-data

In this repo you will find the code to aggregate users requests, discretize
their needs, simulate the credibility allocation and run some experiments on it.


Usage
-----

Download the Google dataset following the instructions here_.

.. _here: https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md

Clone this repo::

    git clone git@github.com:hugombarreto/credibility_allocation.git

Run it for usage information::

    python credibility_allocation --help

For further help in a particular subcommand you can run::

    python credibility_allocation --help


Subcommands
-----------
Here we detail the purpose of each subcommand. In an usual workflow they are
likely used in the order they appear here (except for the
``multicore_simulate_allocation`` which is in fact an alternative to the
``simulate_allocation``).

:aggregate_user_events:
    Inspect dataset events aggregating cpu and memory requests for each user in
    a separate file for each.

:generate_user_needs:
    Aggregates all resource requests so that we have, at a given timestamp, the
    amount of resources needed by the user.

:sample_needs:
    Sample needs for each user following a period specified. Creates a single
    file with all users types for every timestamp.

:simulate_allocation:
    Simulate allocation using sampled types created with the ``sample_needs``.
    It generates a file with the allocations for every user on every iteration.

:multicore_simulate_allocation:
    **Alternative** to the ``simulate_allocation`` using multiple CPU cores.
    It monitors the memory usage so as to avoid swapping.

:experiments:
    Runs a set of experiments using the simulation output.
