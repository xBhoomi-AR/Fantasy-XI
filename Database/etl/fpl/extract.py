import requests

from common.config import (
    FPL_BASE_URL,
    REQUEST_TIMEOUT,
    HEADERS
)


def fetch_bootstrap_static():
    response = requests.get(
        f"{FPL_BASE_URL}/bootstrap-static/",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()


def fetch_fixtures():
    response = requests.get(
        f"{FPL_BASE_URL}/fixtures/",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()


def fetch_player_summary(player_id):
    response = requests.get(
        f"{FPL_BASE_URL}/element-summary/{player_id}/",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()


def fetch_event_status():
    response = requests.get(
        f"{FPL_BASE_URL}/event-status/",
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()