import collections.abc
from itertools import chain

from . import SyclDevice, get_devices


def select_device_with_aspects(required_aspects, excluded_aspects=[]):
    """Selects the root :class:`dpctl.SyclDevice` that has the highest
    default selector score among devices that have all aspects in the
    `required_aspects` list, and do not have any aspects in `excluded_aspects`
    list.

    Supported

        :Example:
            .. code-block:: python

                import dpctl
                # select a GPU that supports double precision
                dpctl.select_device_with_aspects(['fp64', 'gpu'])
                # select non-custom device with USM shared allocations
                dpctl.select_device_with_aspects(
                    ['usm_shared_allocations'], excluded_aspects=['custom'])
    """
    if isinstance(required_aspects, str):
        required_aspects = [required_aspects]
    if isinstance(excluded_aspects, str):
        excluded_aspects = [excluded_aspects]
    seq = collections.abc.Sequence
    input_types_ok = isinstance(required_aspects, seq) and isinstance(
        excluded_aspects, seq
    )
    if not input_types_ok:
        raise TypeError(
            "Aspects are expected to be Python sequences, "
            "e.g. lists, of strings"
        )
    for asp in chain(required_aspects, excluded_aspects):
        if type(asp) != str:
            raise TypeError("The list objects must be of a string type")
        if not hasattr(SyclDevice, "has_aspect_" + asp):
            raise AttributeError(f"The {asp} aspect is not supported in dpctl")
    devs = get_devices()
    max_score = 0
    selected_dev = None

    for dev in devs:
        aspect_status = all(
            (
                getattr(dev, "has_aspect_" + asp) is True
                for asp in required_aspects
            )
        )
        aspect_status = aspect_status and not (
            any(
                (
                    getattr(dev, "has_aspect_" + asp) is True
                    for asp in excluded_aspects
                )
            )
        )
        if aspect_status and dev.default_selector_score > max_score:
            max_score = dev.default_selector_score
            selected_dev = dev

    if selected_dev is None:
        raise ValueError(
            f"Requested device is unavailable: "
            f"required_aspects={required_aspects}, "
            f"excluded_aspects={excluded_aspects}"
        )

    return selected_dev
