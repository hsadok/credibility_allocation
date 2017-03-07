

Description of some Important Files
-----------------------------------


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

