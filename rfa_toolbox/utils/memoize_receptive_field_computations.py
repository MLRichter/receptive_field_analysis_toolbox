from functools import lru_cache
from typing import List, Tuple

from rfa_toolbox.domain import Layer


def compute_receptive_field_size(
    previous_receptive_field_size: int,
    multiplicator: int,
    kernel_size: int,
    stride_size: int,
) -> Tuple[int, int]:
    return (
        previous_receptive_field_size + ((kernel_size - 1) * multiplicator),
        multiplicator * stride_size,
    )


@lru_cache(maxsize=100000)
def compute_receptive_field_size_recursive(
    sequence: Tuple[Layer], receptive_field_sizes: Tuple[int], multiplicator: int = 1
) -> Tuple[int]:
    if not sequence:
        return receptive_field_sizes
    else:
        rf, mult = compute_receptive_field_size(
            previous_receptive_field_size=1
            if not receptive_field_sizes
            else receptive_field_sizes[-1],
            multiplicator=multiplicator,
            kernel_size=sequence[0].kernel_size,
            stride_size=sequence[0].stride_size,
        )
        new_rf = list(receptive_field_sizes)
        new_rf.append(rf)
        new_rf = tuple(new_rf)
        return compute_receptive_field_size_recursive(
            sequence=tuple() if len(sequence) == 1 else sequence[1:],
            receptive_field_sizes=new_rf,
            multiplicator=mult,
        )


def compute_receptive_field_size_for_sequence(sequence: List[Layer]) -> List[int]:
    sized: Tuple[int] = tuple()
    return list(
        compute_receptive_field_size_recursive(
            sequence=tuple(sequence), receptive_field_sizes=sized, multiplicator=1
        )
    )
