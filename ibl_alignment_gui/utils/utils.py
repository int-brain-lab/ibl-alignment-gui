from collections.abc import Callable
from functools import wraps
from typing import Any


def shank_loop(func: Callable) -> Callable:
    """
    Iterate over multiple shanks and configurations.

    This decorator allows a method to be called automatically for each
    combination of shank and configuration, collecting the results.

    Behavior
    --------
    - If no `shanks` are provided in `kwargs`, `self.all_shanks` is used.
    - If no `configs` are provided in `kwargs`, `self.model.configs` is used.
    - If `data_only` is False, shanks without alignment (`align_exists`) are skipped.
    - The decorated function is called with additional keyword arguments

    Parameters
    ----------
    func : Callable
        The instance method to decorate. It must accept:
        `self, items, *args, **kwargs` (with `shank` and `config` in kwargs).

    Returns
    -------
    Callable
        A wrapped method that, when called, loops over the specified (or default)
        shanks and configurations, calling the original method for each, and
        returning a list of results.
    """
    @wraps(func)
    def wrapper(controller, *args, **kwargs) -> Any:

        shanks = kwargs.pop('shanks', controller.all_shanks)
        shanks = controller.all_shanks if shanks is None else shanks
        configs = kwargs.pop('configs', controller.model.configs)
        configs = controller.model.configs if configs is None else configs
        data_only = kwargs.pop('data_only', False)

        results = []
        for config in configs:
            for shank in shanks:
                if (not controller.model.get_current_shank(shank, config).align_exists and
                        not data_only):
                    continue

                result = func(controller, controller.shank_items[shank][config], *args, **kwargs,
                              shank=shank, config=config)
                results.append(result)
        return results

    return wrapper
