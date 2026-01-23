import ephysatlas.regionclassifier
import ephysatlas.data
import numpy as np
import pandas as pd
from iblutil.numerical import ismember
from iblutil.util import Bunch

MODEL_NAME = 'Inference'

def ensure_model(controller):
    plugin = controller.plugins['Channel Prediction']
    if MODEL_NAME not in plugin or plugin[MODEL_NAME] is None:
        plugin[MODEL_NAME] = load_inference_model(controller)
    return plugin[MODEL_NAME]


def load_inference_model(controller):
    one = controller.model.one

    # Find the latest available model and download
    available_models = ephysatlas.data.list_available_models(one=one)
    model_name = available_models[-1]
    model_path = one.cache_dir.joinpath('ephys_atlas_features', model_name)
    model_path.mkdir(parents=True, exist_ok=True)
    _ = ephysatlas.data.download_model(one=one, model_name=model_name, local_path=model_path)
    # Load in the model
    _, model_info = ephysatlas.regionclassifier.load_model(model_path)

    return Bunch(info=model_info, path=model_path.joinpath('folds'))

def predict(controller, items):

    if not items.model.raw_data['features']['exists']:
        return

    df = items.model.raw_data['features']['df']
    model = ensure_model(controller)
    predicted_probas, _ = ephysatlas.regionclassifier.infer_regions(df, path_model=model['path'])

    cosmos_ids = np.array(model['info']['CLASSES'])[np.argmax(np.mean(predicted_probas, axis=0), axis=1)]
    depths = df['axial_um'].values

    return cosmos_ids, depths


def predict_cumulative(controller, items):

    if not items.model.raw_data['features']['exists']:
        return

    df = items.model.raw_data['features']['df']
    model = ensure_model(controller)
    predicted_probas, _ = ephysatlas.regionclassifier.infer_regions(df, path_model=model['path'])

    cprobas = np.mean(predicted_probas, axis=0).cumsum(axis=1)
    region_ids = np.array(model['info']['CLASSES']).astype(int)
    depths = df['axial_um'].values

    _, region_idxs = ismember(region_ids, controller.model.brain_atlas.regions.id)
    colours = [controller.model.brain_atlas.regions.rgb[idx] for idx in region_idxs]

    return cprobas, depths, colours, region_ids