"""Envelope encryption for face embeddings and database key management.

Implements AES-256-GCM envelope encryption where each face embedding has
its own Data Encryption Key (DEK) encrypted by a Key Encryption Key (KEK)
stored in the macOS Keychain.  Also manages the SQLCipher database password.
"""

from dataclasses import dataclass


@dataclass
class EncryptedPayload:
    """Container for an envelope-encrypted value."""

    ciphertext: bytes
    nonce: bytes
    dek_encrypted: bytes


def encrypt_embedding(raw_bytes: bytes) -> EncryptedPayload:
    """Encrypt a face embedding using envelope encryption.

    Parameters
    ----------
    raw_bytes : bytes
        Raw embedding bytes to encrypt.

    Returns
    -------
    EncryptedPayload
        The encrypted embedding with its wrapped DEK.
    """
    pass


def decrypt_embedding(payload: EncryptedPayload) -> bytes:
    """Decrypt an envelope-encrypted face embedding.

    Parameters
    ----------
    payload : EncryptedPayload
        The encrypted payload to decrypt.

    Returns
    -------
    bytes
        The decrypted embedding bytes.
    """
    pass


def get_database_password() -> str:
    """Retrieve or create the SQLCipher database password from Keychain.

    Returns
    -------
    str
        The database encryption password.
    """
    pass
