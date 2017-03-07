import math
import heapq


def allocate(desire, reputation, output=None):
    if output is None:
        output = [0]*len(reputation)
    desire = desire[:]
    desire_sum = sum(desire)
    
    if desire_sum < 0:
        desire = [-d for d in desire]
        reputation = [-r for r in reputation]
    elif desire_sum == 0:
        output[:] = desire
        return output

    reputation_heap = [(r, i) for (i, r) in enumerate(reputation)
                       if desire[i] > 0]
    reputation_heap.append((float('inf'), -1))
    heapq.heapify(reputation_heap)

    available_resources = 0
    for i, d in enumerate(desire):
        if d < 0:
            output[i] = d
            available_resources += -d

    while available_resources > 0:
        if len(reputation_heap) < 2:
            max_reputation, max_id = reputation_heap[0]
            output[max_id] += available_resources
            available_resources = 0
            break

        max_reputation, max_id = heapq.heappop(reputation_heap)
        ref_reputation, ref_id = reputation_heap[0]

        distance = ref_reputation - max_reputation
        distance = math.ceil(distance)
        if distance == 0:
            distance = 1

        max_allocatable = min(available_resources, desire[max_id])

        allocation = int(min(max_allocatable, distance))

        output[max_id] += allocation
        available_resources -= allocation

        if max_allocatable > distance:
            max_reputation += allocation
            desire[max_id] -= allocation
            heapq.heappush(reputation_heap, (max_reputation, max_id))

    if desire_sum < 0:
        output[:] = [-o for o in output]

    return output


def check_allocation(desire, output):
    def utility(desire, output):
        if output < desire < 0:
            return False
        if output < 0 <= desire:
            return False
        return True

    if sum(output) != 0:
        return False

    if not all(utility(d, o) for (d, o) in zip(desire, output)):
        return False

    return True
