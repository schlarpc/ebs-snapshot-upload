#!/usr/bin/env python3

"""
AWS EBS direct snapshot uploader

This tool allows you to upload a disk image file as an EBS snapshot directly,
without using an intermediary EC2 instance or the VMIE API (ec2:ImportImage).
"""


import argparse
import base64
import concurrent.futures
import contextlib
import hashlib
import logging
import os
import pathlib
import threading
import time
from typing import Any, Callable, ClassVar, Dict, Generator, List, Optional

import boto3

logger = logging.getLogger(__name__)


class ThreadLocalBotoSession(boto3.session.Session, threading.local):
    """
    boto3 Session subclass that reinitializes for each thread that it's used in,
    to avoid sharing non-thread-safe state. Additionally, created clients are cached
    and reused on a per-thread basis.
    """

    def client(self, service_name: str):
        cache = self.__dict__.setdefault("_client_cache", {})
        if service_name not in cache:
            cache[service_name] = super().client(service_name)
        return cache[service_name]


class SnapshotWriter:
    BYTES_PER_SECTOR: ClassVar[int] = 512
    SECTORS_PER_PUT: ClassVar[int] = 1024
    BYTES_PER_PUT: ClassVar[int] = BYTES_PER_SECTOR * SECTORS_PER_PUT

    def __init__(self, session, snapshot_id: str, threads: int):
        self._session = session
        self._snapshot_id = snapshot_id
        self._block_index = 0
        self._block_checksums: Dict[int, bytes] = {}
        self._upload_pool = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
        self._upload_futures: List[concurrent.futures.Future] = []
        self._upload_max_futures = threads * 2

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def blocks_written(self) -> int:
        return len(self._block_checksums)

    @property
    def checksum(self) -> bytes:
        sorted_block_checksums = [v for k, v in sorted(self._block_checksums.items())]
        return hashlib.sha256(b"".join(sorted_block_checksums)).digest()

    def _upload_block(self, block_index: int, data: bytes) -> None:
        digest = hashlib.sha256(data).digest()
        self._session.client("ebs").put_snapshot_block(
            SnapshotId=self._snapshot_id,
            BlockIndex=block_index,
            BlockData=data,
            DataLength=self.BYTES_PER_PUT,
            Checksum=base64.b64encode(digest).decode("ascii"),
            ChecksumAlgorithm="SHA256",
        )
        self._block_checksums[block_index] = digest

    def _wait_for_any_future_to_complete(self) -> None:
        # TODO upload progress indicator
        completed = next(concurrent.futures.as_completed(self._upload_futures))
        completed.result()
        self._upload_futures.remove(completed)

    def write(self, data: bytes) -> None:
        if len(data) != self.BYTES_PER_PUT:
            raise ValueError(
                f"Snapshot data must be exactly {self.BYTES_PER_PUT} bytes"
            )
        while len(self._upload_futures) >= self._upload_max_futures:
            self._wait_for_any_future_to_complete()
        submitted = self._upload_pool.submit(
            self._upload_block, self._block_index, data
        )
        self._upload_futures.append(submitted)
        self._block_index += 1

    def join(self) -> None:
        while self._upload_futures:
            self._wait_for_any_future_to_complete()
        self._upload_pool.shutdown(wait=True)


def calculate_required_gibibytes(source_size_bytes: int) -> int:
    quotient, remainder = divmod(source_size_bytes, 1024 ** 3)
    return quotient + (1 if remainder else 0)


def filter_dict(func: Callable[[Any, Any], bool], d: dict) -> dict:
    return {k: v for k, v in d.items() if func(k, v)}


def wait_for_snapshot_completion(
    client, snapshot_id, *, max_attempts: int = 100, delay: int = 3
):
    for attempt in range(max_attempts):
        # we sleep first because there's some eventual consistency and caching in the API.
        # so if we query EC2 before the the snapshot has transitioned from pending,
        # it'll keep returning pending until it falls out of the server-side cache :(
        time.sleep(delay)
        # we use a paginator rather than a waiter because botocore's waiter doesn't
        # recognize "error" as a terminal state.
        paginator = client.get_paginator("describe_snapshots")
        iterator = paginator.paginate(SnapshotIds=[snapshot_id], OwnerIds=["self"])
        for status in iterator.search("Snapshots[].State"):
            if status == "completed":
                return
            elif status == "pending":
                continue
            elif status == "error":
                raise Exception("Snapshot reached error state; maybe a checksum issue?")
            else:
                raise Exception(f"Unknown snapshot status {status!r}")
    else:
        raise Exception(f"Snapshot in non-terminal state after {max_attempts} checks")


@contextlib.contextmanager
def create_snapshot(
    *,
    session,
    volume_size: int,
    encrypted: bool,
    description: Optional[str] = None,
    kms_key_arn: Optional[str] = None,
    threads: int,
) -> Generator[SnapshotWriter, None, None]:
    ebs = session.client("ebs")
    ec2 = session.client("ec2")
    response = ebs.start_snapshot(
        **filter_dict(
            lambda k, v: v is not None,
            {
                "VolumeSize": volume_size,
                "Encrypted": encrypted,
                "KmsKeyArn": kms_key_arn,
                "Description": description,
            },
        )
    )
    snapshot_id = response["SnapshotId"]
    try:
        logger.info("Snapshot upload started with id %s", snapshot_id)
        writer = SnapshotWriter(session, snapshot_id, threads=threads)
        yield writer
        writer.join()
        ebs.complete_snapshot(
            SnapshotId=writer.snapshot_id,
            ChangedBlocksCount=writer.blocks_written,
            Checksum=base64.b64encode(writer.checksum).decode("ascii"),
            ChecksumAlgorithm="SHA256",
            ChecksumAggregationMethod="LINEAR",
        )
        logger.info("Fully uploaded, waiting for server-side completion")
        wait_for_snapshot_completion(ec2, snapshot_id)
    except:
        logger.error("Exception caught, deleting snapshot in progress")
        ec2.delete_snapshot(SnapshotId=snapshot_id)
        raise


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path, help="Disk image file to upload.")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug-level logging."
    )
    parser.add_argument(
        "--threads",
        type=int,
        metavar="N",
        default=min(32, os.cpu_count() * 2),
        help="Concurrent upload threads. Defaults to a heuristic based on core count.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Use a specific profile from your credential file.",
    )
    parser.add_argument(
        "--region", help="The region to use. Overrides config/env settings."
    )
    parser.add_argument(
        "--minimum-size",
        type=int,
        default=0,
        help="Minimum snapshot size in GiB. If the source file is larger, this is ignored.",
    )
    parser.add_argument(
        "--encrypted",
        action="store_true",
        help="Enable KMS encryption on the snapshot.",
    )
    parser.add_argument(
        "--kms-key-arn",
        help="Manually specify a KMS key ARN used to encrypt the snapshot.",
    )
    parser.add_argument(
        "--description", help="Description associated with the snapshot."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    if not args.debug:
        logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    session = ThreadLocalBotoSession(profile_name=args.profile, region_name=args.region)
    volume_size = max(
        1, args.minimum_size, calculate_required_gibibytes(args.source.stat().st_size)
    )
    logging.info(
        "Creating a %d GiB snapshot with %d threads", volume_size, args.threads
    )
    with create_snapshot(
        session=session,
        volume_size=volume_size,
        encrypted=args.encrypted,
        description=args.description,
        kms_key_arn=args.kms_key_arn,
        threads=args.threads,
    ) as writer:
        with args.source.open("rb") as f:
            while chunk := f.read(writer.BYTES_PER_PUT):
                if len(chunk) < writer.BYTES_PER_PUT:
                    chunk += b"\x00" * (writer.BYTES_PER_PUT - len(chunk))
                writer.write(chunk)
    logger.info("Snapshot created successfully")
    print(writer.snapshot_id)


if __name__ == "__main__":
    main()
