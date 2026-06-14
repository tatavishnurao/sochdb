#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _grpc_stubs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "agent_memory_qa" / "grpc_stubs"


def _load_grpc_stubs():
    stubs_dir = _grpc_stubs_dir()
    stubs_dir_str = str(stubs_dir)
    if stubs_dir_str not in sys.path:
        sys.path.insert(0, stubs_dir_str)

    import sochdb_pb2  # type: ignore
    import sochdb_pb2_grpc  # type: ignore

    return sochdb_pb2, sochdb_pb2_grpc


def _flatten_queries(queries: Sequence[Sequence[float]]) -> List[float]:
    flat: List[float] = []
    for query in queries:
        flat.extend(float(x) for x in query)
    return flat


def _extract_metadata(client) -> Optional[list[tuple[str, str]]]:
    metadata = getattr(client, "_metadata", None)
    if metadata:
        return list(metadata)

    api_key = getattr(client, "api_key", None) or getattr(client, "_api_key", None)
    if api_key:
        return [("x-api-key", str(api_key))]

    return None


def _get_client_channel(client):
    channel = getattr(client, "_channel", None)
    if channel is not None:
        return channel, False

    address = (
        getattr(client, "address", None)
        or getattr(client, "_address", None)
        or getattr(client, "grpc_address", None)
    )
    if not address:
        raise AttributeError(
            "SochDB client does not expose a usable gRPC channel or address for SearchBatch"
        )

    secure = bool(getattr(client, "secure", False) or getattr(client, "_secure", False))
    import grpc

    if secure:
        channel = grpc.secure_channel(address, grpc.ssl_channel_credentials())
    else:
        channel = grpc.insecure_channel(address)

    return channel, True


def search_batch(
    client,
    index_name: str,
    queries: List[List[float]],
    k: int = 10,
    ef: int = 0,
):
    """Batch vector search against SochDB via SDK method or direct gRPC fallback."""
    if not queries:
        return []

    if hasattr(client, "search_batch"):
        return client.search_batch(index_name=index_name, queries=queries, k=k, ef=ef)

    sochdb_pb2, sochdb_pb2_grpc = _load_grpc_stubs()
    channel, should_close = _get_client_channel(client)
    metadata = _extract_metadata(client)

    try:
        stub = sochdb_pb2_grpc.VectorIndexServiceStub(channel)
        request = sochdb_pb2.SearchBatchRequest(
            index_name=index_name,
            queries=_flatten_queries(queries),
            num_queries=len(queries),
            k=k,
            ef=ef,
        )
        response = stub.SearchBatch(request, metadata=metadata)
        return [list(group.results) for group in response.results]
    finally:
        if should_close:
            channel.close()
