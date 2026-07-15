import psycopg2
from psycopg2.extras import RealDictCursor

from .config import DATABASE


def get_connection():
    return psycopg2.connect(
        host=DATABASE["host"],
        port=DATABASE["port"],
        database=DATABASE["database"],
        user=DATABASE["user"],
        password=DATABASE["password"],
        cursor_factory=RealDictCursor
    )


def close_connection(connection):
    if connection is not None:
        connection.close()


def execute_query(query, params=None):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(query, params)

        connection.commit()

    except Exception as e:
        if connection:
            connection.rollback()
        raise e

    finally:
        if cursor:
            cursor.close()
        if connection:
            close_connection(connection)


def fetch_all(query, params=None):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(query, params)

        return cursor.fetchall()

    finally:
        if cursor:
            cursor.close()
        if connection:
            close_connection(connection)


def fetch_one(query, params=None):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(query, params)

        return cursor.fetchone()

    finally:
        if cursor:
            cursor.close()
        if connection:
            close_connection(connection)