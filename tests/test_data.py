from pathlib import Path

from prism_live_rag.data import load_query_tasks, validate_domain


DATA_DIR = Path("data")


def test_cloud_query_join_is_available() -> None:
    tasks = load_query_tasks(DATA_DIR, "cloud")
    assert tasks
    assert all(task.query for task in tasks)
    assert all(task.qrel_passage_ids for task in tasks)


def test_govt_dataset_validates() -> None:
    stats = validate_domain(DATA_DIR, "govt")
    assert stats["tasks"] > 0
    assert stats["qrel_passages"] > 0
    assert stats["corpus_passages"] > stats["qrel_passages"]

