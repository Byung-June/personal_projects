import requests
from settings import api_exchange_address

def _get_json(endpoint: str, params: dict) -> dict:
    """
    Internal helper to send GET request and return the JSON 'result' field.
    Raises:
        requests.HTTPError: for bad status codes
        ValueError: if JSON is malformed or 'result' missing
    """
    url = api_exchange_address.rstrip('/') + endpoint
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if 'result' not in data:
        raise ValueError(f"No 'result' in response: {data}")
    return data['result']

def get_volatility_index_data(currency: str, start_timestamp: int, end_timestamp: int, resolution: int) -> dict:
    """
    Fetch volatility index data for a given currency and time range.
    """
    endpoint = "/api/v2/public/get_volatility_index_data"
    params = {
        'currency': currency,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp,
        'resolution': resolution,
    }
    return _get_json(endpoint, params)

def get_book_summary_by_currency(currency: str, kind: str) -> dict:
    """
    Fetch the book summary for a given currency and contract kind.
    """
    endpoint = "/api/v2/public/get_book_summary_by_currency"
    params = {
        'currency': currency,
        'kind': kind,
    }
    return _get_json(endpoint, params)
