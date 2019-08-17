import h2o

import_errors = {}


def assert_import(package_name):
    global import_errors
    if package_name in import_errors:
        msg,e = import_errors[package_name]
        print(msg)
        raise e


def record_import_error(package_name, msg, e):
    global import_errors
    import_errors[package_name] = (msg, e)


def is_h2o_frame(df):
    """
    Checks if teh dataframe is an h2o frame
    :param df:
    :return:
    """
    if type(df) == h2o.H2OFrame: # if isinstance(df, h2o.H2OFrame):
        return True
    else:
        return False


def convert_h2o_list(lst):
    """
    Converts an h2o list to a python list
    :param lst:
    :return:
    """
    return h2o.as_list(lst)
