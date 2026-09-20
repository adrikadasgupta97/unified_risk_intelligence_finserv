import pytest
from src.data.database import init_db, save_complaint, get_complaint, update_complaint
from src.data.models import ComplaintRecord, ComplaintUpdate, ComplaintStatus


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.database.DB_PATH", tmp_path / "test.db")
    init_db()


def test_save_and_get_complaint():
    record = ComplaintRecord(
        customer_id="CUST-001",
        raw_text="My card was charged incorrectly.",
        anonymized_text="My card was charged incorrectly.",
    )
    cid = save_complaint(record)
    assert cid == record.complaint_id

    fetched = get_complaint(cid)
    assert fetched is not None
    assert fetched.customer_id == "CUST-001"


def test_update_complaint_status():
    record = ComplaintRecord(customer_id="CUST-002", raw_text="Test complaint.")
    save_complaint(record)

    updated = update_complaint(record.complaint_id, ComplaintUpdate(status=ComplaintStatus.RESOLVED))
    assert updated.status == ComplaintStatus.RESOLVED
    assert updated.resolved_at is not None
