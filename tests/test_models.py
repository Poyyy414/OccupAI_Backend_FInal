import pytest
from pydantic import ValidationError

from backend.models import PushFrame, UserRegister, YoloUpdate


def test_yolo_update_rejects_inconsistent_counts():
    with pytest.raises(ValidationError):
        YoloUpdate(occupied=3, free=2, total=4)


def test_yolo_update_uses_independent_mutable_defaults():
    first = YoloUpdate(occupied=0, total=4)
    second = YoloUpdate(occupied=0, total=4)
    first.zones['Z1'] = True
    assert second.zones == {}


def test_large_or_weak_account_payloads_are_rejected():
    with pytest.raises(ValidationError):
        UserRegister(
            first_name='A',
            last_name='B',
            email='a@example.com',
            password='x' * 129,
            verification_code='123456',
        )


def test_user_register_requires_exactly_six_numeric_verification_digits():
    valid = {
        'first_name': 'Ada',
        'last_name': 'Lovelace',
        'email': 'ada@example.com',
        'password': 'secure123',
    }
    assert UserRegister(**valid, verification_code='012345').verification_code == '012345'

    for invalid_code in ('12345', '1234567', '12A456'):
        with pytest.raises(ValidationError):
            UserRegister(**valid, verification_code=invalid_code)


def test_frame_payload_has_a_size_limit():
    with pytest.raises(ValidationError):
        PushFrame(frame='x' * 5_000_001)
