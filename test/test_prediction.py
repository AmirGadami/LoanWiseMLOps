import pytest

from configs import config
from src.processing.data_handling import load_data
from src.pipelines.predict import generate_predictions 


@pytest.fixture

def single_prediction():
    data = load_data(config.TEST_FILE)
    single_data = data[:1]
    result = generate_predictions(single_data)
    return result

def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction):
    assert isinstance(single_prediction.get('prediction')[0],str)

def test_single_pred_validate(single_prediction):
    assert single_prediction.get('prediction')[0] == 'Y'


if __name__ == '__main__':
    data = load_data(config.TEST_FILE)
    single_data = data[:1]
    result = generate_predictions(single_data)
    # return result
    print(result.get('prediction')[0])
    assert isinstance(result.get('prediction')[0],str)