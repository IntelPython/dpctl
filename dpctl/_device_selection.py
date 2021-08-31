from . import SyclDevice, get_devices


def select_device_with_aspects(aspect_list, deny_list=[]):
    check_list = aspect_list + deny_list
    for asp in check_list:
        if type(asp) != str:
            raise TypeError("The list objects must be of a string type")
        if not hasattr(SyclDevice, "has_aspect_" + asp):
            raise ValueError(f"The {asp} aspect is not supported in dpctl")
    devs = get_devices()
    max_score = 0
    selected_dev = None

    for dev in devs:
        # aspect_status = True
        #     for asp in aspect_list:
        #         has_aspect = "dev.has_aspect_" + asp
        #         if not eval(has_aspect):
        #             aspect_status = False
        #     for deny in deny_list:
        #         has_aspect = "dev.has_aspect_" + deny
        #         if eval(has_aspect):
        #             aspect_status = False
        aspect_status = all(
            (getattr(dev, "has_aspect_" + asp) is True for asp in aspect_list)
        )
        aspect_status = aspect_status and not (
            any(
                (getattr(dev, "has_aspect_" + asp) is True for asp in deny_list)
            )
        )
        if aspect_status and dev.default_selector_score > max_score:
            max_score = dev.default_selector_score
            selected_dev = dev

    return selected_dev
