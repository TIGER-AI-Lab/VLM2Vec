import pyarrow as pa
from abc import ABCMeta, abstractmethod
from functools import wraps
from datasets import Features, Value, Sequence


# MULTIMODAL_FEATURES = Features(**{
#     "query_text": Value(dtype='string', id=None),
#     "query_image": {'bytes': Value(dtype='null', id=None), 'path': Value(dtype='null', id=None)},
#     "pos_text": Value(dtype='string', id=None),
#     "pos_image": {'bytes': Value(dtype='null', id=None), 'path': Value(dtype='null', id=None)},
#     "neg_text": Value(dtype='string', id=None),
#     "neg_image": {'bytes': Value(dtype='null', id=None), 'path': Value(dtype='null', id=None)},
#     "global_dataset_name": Value(dtype='string', id=None),
# })

arrow_schema = pa.schema([
    pa.field("query_text", pa.string()),
    pa.field("query_image", pa.struct([
        pa.field("paths", pa.list_(pa.string())),
        pa.field("bytes", pa.list_(pa.binary())),
        pa.field("resolutions", pa.list_(pa.list_(pa.int32(), 2))),
    ])),
    pa.field("pos_text", pa.string()),
    pa.field("pos_image", pa.struct([
        pa.field("paths", pa.list_(pa.string())),
        pa.field("bytes", pa.list_(pa.binary())),
        pa.field("resolutions", pa.list_(pa.list_(pa.int32(), 2))),
    ])),
    pa.field("neg_text", pa.list_(pa.string())),
    pa.field("neg_image", pa.list_(pa.struct([
        pa.field("paths", pa.list_(pa.string())),
        pa.field("bytes", pa.list_(pa.binary())),
        pa.field("resolutions", pa.list_(pa.list_(pa.int32(), 2))),
    ]))),
    pa.field("global_dataset_name", pa.string()),
])

MULTIMODAL_FEATURES = Features.from_arrow_schema(arrow_schema)

RESOLUTION_MAPPING = {
    "high": (1344, 1344),
    "mid": (672, 672),
    "low": (128, 128),
}


class AutoPairDataset(metaclass=ABCMeta):
    # Base class for auto datasets.
    registry = {}

    def __init_subclass__(cls):
        if cls.__name__ not in AutoPairDataset.registry:
            AutoPairDataset.registry[cls.__name__] = cls
        else:
            raise RuntimeError('Subclass "{cls.__name__}" has already defined.')

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def instantiate(cls, dataset_parser, *args, **kwargs):
        try:
            return cls.registry[dataset_parser](*args, **kwargs)
        except Exception as e:
            raise e

    @classmethod
    def register(cls, dataset_name):
        def inner_wrapper(wrapped_class):
            if dataset_name in cls.registry:
                print(f"[Alert] AutoPairDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @abstractmethod
    def main(self):
        pass

def add_metainfo_hook(f):
    """
    A post-processing wrapper function that add meta information (e.g. data_type, dataset_name, loss_type) into batches
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # go through data pipeline customized to each dataset
        batch_data = f(*args, **kwargs)
        # append common metadata
        batch_size = len(batch_data['query_text'])
        global_dataset_name = kwargs.get("global_dataset_name", "None")
        batch_data['global_dataset_name'] = [global_dataset_name] * batch_size
        return batch_data

    return wrapper

def convert_neg_fields(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        assert isinstance(out, dict), f"Expected output to be a dict, but got {type(out)}"
        # helpers
        def is_nested_list_str(x):
            return isinstance(x, list) and all(isinstance(e, list) and all(isinstance(s, str) for s in e) for e in x)

        def is_list_str(x):
            return isinstance(x, list) and all(isinstance(s, str) for s in x)

        def is_nested_list_dict(x):
            return isinstance(x, list) and all(isinstance(e, list) and all(isinstance(d, dict) for d in e) for e in x)

        def is_list_dict(x):
            return isinstance(x, list) and all(isinstance(d, dict) for d in x)

        # neg_text: make list[list[str]]
        if "neg_text" in out and not is_nested_list_str(out["neg_text"]):
            if is_list_str(out["neg_text"]):
                out["neg_text"] = [[s] for s in out["neg_text"]]

        # neg_image: make list[list[dict]]
        if "neg_image" in out and not is_nested_list_dict(out["neg_image"]):
            if is_list_dict(out["neg_image"]):
                out["neg_image"] = [[d] for d in out["neg_image"]]

        return out
    return wrapper