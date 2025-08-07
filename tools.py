import sys
from typing import Any, Dict, List, Optional

import requests
from langchain.tools import tool

from rag import faiss_retriever

RAG_DESC = """
    Searches the local document database for information related to the specified identifier.

    This function performs a query on the local FAISS-backed document store using the provided
    identifier, which could represent a unique person, event, or keyword. It returns the
    concatenated content of all matching documents for further analysis or display.

    Args:
        identifier (str): Unique term, name, or keyword to search in the local document database.

    Returns:
        str: Joined text content of all relevant documents found by the search. If no results are found,
        the returned string will be empty.

    Example:
        >>> search_local_database("John Doe")
        'John Doe was last seen ...\n\nKnown associates include ...'
    """


@tool(parse_docstring=True)
def dehashed_search_by_email_proxy() -> List[Any]:
    """
    Returns sample breach data for a given email address (proxy for live DeHashed search).

    This function simulates querying the DeHashed API and returns static sample
    breach results. Intended for demonstration or testing environments.

    Returns:
        list[dict[str, str]]: A list of dictionaries, each representing a breach record
        with personal and breach details.

    Example:
        >>> dehashed_search_by_email_proxy()
        [
            {
                "id": "1234567890",
                "first_name": "Adam",
                ...
            },
            ...
        ]
    """

    return [
        {
            "id": "1234567890",
            "first_name": "Adam",
            "last_name": "Baker",
            "wife_name": "Sarah Baker",
            "address": "1234 Maple Ave, Sydney, FK 99999",
            "mobile": "+1-555-123-4567",
            "email": "adam.baker@example.com",
            "source": "infostealer-leak",
            "breached_date": "2023-11-10",
            "details": {
                "ip_address": "158.97.10.21",
                "hostname": "adam-laptop.local",
                "password": "passw0rd123",
                "browsers": ["Chrome", "Firefox"],
                "cookies": ["sessionid=abc123; path=/;"],
            },
        },
        {
            "id": "8112283456",
            "first_name": "Adam",
            "last_name": "Baker",
            "address": "1234 Maple Ave, Sydney, FK 99999",
            "mobile": "+1-555-987-6543",
            "email": "adam.baker@mail.com",
            "source": "fitness-fanatics.com",
            "breached_date": "2024-04-22",
            "details": {
                "membership_status": "active",
                "last_login": "2024-04-21 08:10:00",
                "favorite_class": "Yoga",
                "emergency_contact": "Sarah Baker, +1-555-121-1212",
            },
        },
    ]


@tool
def dehashed_search_by_email(
    api_key: str, email: str, page_size: int = 25
) -> Optional[Dict[str, Any]]:
    """
    Search DeHashed API for breach data associated with an email address.

    Args:
        api_key (str): Your DeHashed API key.
        email (str): Email address to search for.
        size (int): Number of results on a page, default = 25

    Returns:
        Optional[Dict[str, Any]]: JSON response from DeHashed API if successful, None otherwise.
    """
    URL = "https://api.dehashed.com/search"

    headers = {
        "Accept": "application/json",
    }

    params = {
        "query": f"email:{email}",
        "size": page_size,
    }

    try:
        response = requests.get(
            URL,
            headers=headers,
            params=params,
            auth=(api_key, ""),
        )

        response.raise_for_status()

        return response.json()

    except requests.HTTPError as e:
        print(f"HTTP error occurred: {e}.")

    except requests.RequestException as e:
        print(f"Request error occurred: {e}")

    return None


@tool
def search_local_database(identifier: str) -> str:
    """
    Searches the local document database for information related to the specified identifier.

    This function performs a query on the local FAISS-backed document store using the provided
    identifier, which could represent a unique person, event, or keyword. It returns the
    concatenated content of all matching documents for further analysis or display.

    Args:
        identifier (str): Unique term, name, or keyword to search in the local document database.

    Returns:
        str: Joined text content of all relevant documents found by the search. If no results are found,
        the returned string will be empty.

    Example:
        >>> search_local_database("John Doe")
        'John Doe was last seen ...\n\nKnown associates include ...'
    """

    results = faiss_retriever.get_relevant_documents(identifier)
    return "\n\n".join([doc.page_content for doc in results])
