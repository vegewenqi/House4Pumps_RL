import tqdm
import inspect
from contextlib import contextmanager


@contextmanager
def redirect_stdout__to_tqdm():
    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        to_print = "".join(map(repr, args))
        tqdm.tqdm.write(to_print, **kwargs)

    try:
        # Globally replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print


def tqdm_context(*args, **kwargs):
    # with redirect_stdout__to_tqdm():
    #     postfix_dict = kwargs.pop("postfix_dict", {})
    #     additional_info_flag = kwargs.pop("additional_info_flag", False)
    #     position = kwargs.pop("pos", 0)
    #     kwargs.update({"position": position})
    #     kwargs.update({"leave": False})

    #     t_main = tqdm.tqdm(*args, **kwargs)
    #     t_main.postfix_dict = postfix_dict
    #     if additional_info_flag:
    #         yield t_main
    #     for x in t_main:
    #         t_main.set_postfix(**t_main.postfix_dict)
    #         t_main.refresh()
    #         yield x
    it = args[0]
    return it
