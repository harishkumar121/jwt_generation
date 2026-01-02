#!/usr/bin/env python3
"""
Test suite for generate_jwt.py

Run with: python test_generate_jwt.py
"""

import unittest
import datetime as dt
import time
from generate_jwt import (
    create_jwt,
    decode_jwt,
    peek_jwt,
    require_scope,
    require_permission,
    require_claim,
    JWTError,
    InvalidTokenError,
    InvalidAlgorithmError,
    SignatureVerificationError,
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidIssuerError,
    InvalidAudienceError,
    InvalidClaimError,
)


class TestJWTBasic(unittest.TestCase):
    """Basic JWT creation and verification tests."""
    
    def setUp(self):
        self.secret = "test-secret-key"
        self.payload = {"sub": "user-123", "role": "admin"}
    
    def test_create_and_decode(self):
        """Test basic JWT creation and decoding."""
        token = create_jwt(self.payload, self.secret)
        header, claims = decode_jwt(token, self.secret)
        
        self.assertEqual(claims["sub"], "user-123")
        self.assertEqual(claims["role"], "admin")
        self.assertIn("iat", claims)
        self.assertIn("jti", claims)
    
    def test_different_algorithms(self):
        """Test all supported algorithms."""
        for alg in ['HS256', 'HS384', 'HS512']:
            token = create_jwt(self.payload, self.secret, algorithm=alg)
            header, claims = decode_jwt(token, self.secret, algorithms=(alg,))
            self.assertEqual(header['alg'], alg)
    
    def test_invalid_algorithm(self):
        """Test invalid algorithm handling."""
        with self.assertRaises(InvalidAlgorithmError):
            create_jwt(self.payload, self.secret, algorithm="RS256")
    
    def test_wrong_key(self):
        """Test signature verification with wrong key."""
        token = create_jwt(self.payload, self.secret)
        with self.assertRaises(SignatureVerificationError):
            decode_jwt(token, "wrong-key")
    
    def test_invalid_token_format(self):
        """Test invalid token format."""
        with self.assertRaises(InvalidTokenError):
            decode_jwt("invalid.token", self.secret)
        
        with self.assertRaises(InvalidTokenError):
            decode_jwt("only.two.parts", self.secret)


class TestJWTClaims(unittest.TestCase):
    """Test JWT standard claims."""
    
    def setUp(self):
        self.secret = "test-secret-key"
    
    def test_expiration(self):
        """Test expiration claim."""
        token = create_jwt(
            {"sub": "user-123"},
            self.secret,
            expires_in=1  # 1 second
        )
        
        # Should work immediately
        header, claims = decode_jwt(token, self.secret)
        self.assertIn("exp", claims)
        
        # Should fail after expiration
        time.sleep(2)
        with self.assertRaises(ExpiredSignatureError):
            decode_jwt(token, self.secret)
    
    def test_expiration_with_leeway(self):
        """Test expiration with leeway."""
        token = create_jwt(
            {"sub": "user-123"},
            self.secret,
            expires_in=1
        )
        
        time.sleep(2)
        # Should work with leeway
        header, claims = decode_jwt(token, self.secret, leeway=5)
    
    def test_not_before(self):
        """Test not-before claim."""
        token = create_jwt(
            {"sub": "user-123"},
            self.secret,
            not_before_in=2  # valid in 2 seconds
        )
        
        # Should fail immediately
        with self.assertRaises(ImmatureSignatureError):
            decode_jwt(token, self.secret)
        
        # Should work after wait
        time.sleep(3)
        header, claims = decode_jwt(token, self.secret)
        self.assertIn("nbf", claims)
    
    def test_issuer(self):
        """Test issuer claim."""
        token = create_jwt(
            {"sub": "user-123"},
            self.secret,
            issuer="https://test.example"
        )
        
        # Should work with correct issuer
        header, claims = decode_jwt(
            token,
            self.secret,
            expected_issuer="https://test.example"
        )
        
        # Should fail with wrong issuer
        with self.assertRaises(InvalidIssuerError):
            decode_jwt(
                token,
                self.secret,
                expected_issuer="https://wrong.example"
            )
    
    def test_audience(self):
        """Test audience claim."""
        token = create_jwt(
            {"sub": "user-123"},
            self.secret,
            audience=["app-A", "app-B"]
        )
        
        # Should work with matching audience
        header, claims = decode_jwt(
            token,
            self.secret,
            expected_audience="app-A"
        )
        
        # Should fail with non-matching audience
        with self.assertRaises(InvalidAudienceError):
            decode_jwt(
                token,
                self.secret,
                expected_audience="app-C"
            )
    
    def test_subject(self):
        """Test subject claim."""
        token = create_jwt(
            {"role": "admin"},
            self.secret,
            subject="user-123"
        )
        
        header, claims = decode_jwt(token, self.secret)
        self.assertEqual(claims["sub"], "user-123")
    
    def test_no_iat(self):
        """Test token without issued-at."""
        token = create_jwt(
            {"sub": "user-123"},
            self.secret,
            issued_at=False
        )
        
        header, claims = decode_jwt(token, self.secret)
        self.assertNotIn("iat", claims)


class TestDeterministicJTI(unittest.TestCase):
    """Test deterministic JTI generation."""
    
    def setUp(self):
        self.secret = "test-secret-key"
        self.payload = {"sub": "user-123", "role": "admin"}
    
    def test_deterministic_jti_same_payload(self):
        """Test that same payload produces same JTI."""
        token1 = create_jwt(
            self.payload,
            self.secret,
            deterministic_jti=True
        )
        
        token2 = create_jwt(
            self.payload,
            self.secret,
            deterministic_jti=True
        )
        
        header1, claims1 = decode_jwt(token1, self.secret)
        header2, claims2 = decode_jwt(token2, self.secret)
        
        self.assertEqual(claims1["jti"], claims2["jti"])
    
    def test_deterministic_jti_different_payload(self):
        """Test that different payloads produce different JTIs."""
        token1 = create_jwt(
            {"sub": "user-123"},
            self.secret,
            deterministic_jti=True
        )
        
        token2 = create_jwt(
            {"sub": "user-456"},
            self.secret,
            deterministic_jti=True
        )
        
        header1, claims1 = decode_jwt(token1, self.secret)
        header2, claims2 = decode_jwt(token2, self.secret)
        
        self.assertNotEqual(claims1["jti"], claims2["jti"])
    
    def test_deterministic_jti_with_salt(self):
        """Test deterministic JTI with salt."""
        token1 = create_jwt(
            self.payload,
            self.secret,
            deterministic_jti=True,
            jti_salt="salt1"
        )
        
        token2 = create_jwt(
            self.payload,
            self.secret,
            deterministic_jti=True,
            jti_salt="salt2"
        )
        
        header1, claims1 = decode_jwt(token1, self.secret)
        header2, claims2 = decode_jwt(token2, self.secret)
        
        self.assertNotEqual(claims1["jti"], claims2["jti"])


class TestCustomValidators(unittest.TestCase):
    """Test custom claim validators."""
    
    def setUp(self):
        self.secret = "test-secret-key"
    
    def test_require_scope_string(self):
        """Test scope validator with string format."""
        token = create_jwt(
            {"sub": "user-123", "scope": "read write admin"},
            self.secret
        )
        
        # Should work with required scope
        header, claims = decode_jwt(
            token,
            self.secret,
            custom_validators=[require_scope("read")]
        )
        
        # Should fail with missing scope
        with self.assertRaises(InvalidClaimError):
            decode_jwt(
                token,
                self.secret,
                custom_validators=[require_scope("delete")]
            )
    
    def test_require_scope_list(self):
        """Test scope validator with list format."""
        token = create_jwt(
            {"sub": "user-123", "scope": ["read", "write", "admin"]},
            self.secret
        )
        
        # Should work with required scope
        header, claims = decode_jwt(
            token,
            self.secret,
            custom_validators=[require_scope("read", "write")]
        )
    
    def test_require_permission(self):
        """Test permission validator."""
        token = create_jwt(
            {"sub": "user-123", "permissions": ["admin", "user:write"]},
            self.secret
        )
        
        # Should work with required permission
        header, claims = decode_jwt(
            token,
            self.secret,
            custom_validators=[require_permission("admin")]
        )
        
        # Should fail with missing permission
        with self.assertRaises(InvalidClaimError):
            decode_jwt(
                token,
                self.secret,
                custom_validators=[require_permission("superadmin")]
            )
    
    def test_require_claim(self):
        """Test generic claim validator."""
        token = create_jwt(
            {"sub": "user-123", "role": "admin"},
            self.secret
        )
        
        # Should work with matching claim
        header, claims = decode_jwt(
            token,
            self.secret,
            custom_validators=[require_claim("role", "admin")]
        )
        
        # Should fail with mismatched claim
        with self.assertRaises(InvalidClaimError):
            decode_jwt(
                token,
                self.secret,
                custom_validators=[require_claim("role", "user")]
            )
        
        # Should fail with missing claim
        with self.assertRaises(InvalidClaimError):
            decode_jwt(
                token,
                self.secret,
                custom_validators=[require_claim("department")]
            )
    
    def test_multiple_validators(self):
        """Test multiple validators together."""
        token = create_jwt(
            {
                "sub": "user-123",
                "role": "admin",
                "scope": "read write",
                "permissions": ["admin"]
            },
            self.secret
        )
        
        # Should work with all validators passing
        header, claims = decode_jwt(
            token,
            self.secret,
            custom_validators=[
                require_scope("read"),
                require_permission("admin"),
                require_claim("role", "admin")
            ]
        )


class TestPeek(unittest.TestCase):
    """Test peek functionality."""
    
    def setUp(self):
        self.secret = "test-secret-key"
    
    def test_peek_without_verification(self):
        """Test peeking at token without verification."""
        token = create_jwt(
            {"sub": "user-123", "role": "admin"},
            self.secret
        )
        
        header, claims = peek_jwt(token)
        
        self.assertEqual(claims["sub"], "user-123")
        self.assertEqual(claims["role"], "admin")
    
    def test_peek_invalid_token(self):
        """Test peeking at invalid token."""
        with self.assertRaises(InvalidTokenError):
            peek_jwt("invalid.token")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        self.secret = "test-secret-key"
    
    def test_empty_payload(self):
        """Test token with empty payload."""
        token = create_jwt({}, self.secret)
        header, claims = decode_jwt(token, self.secret)
        self.assertIn("iat", claims)
        self.assertIn("jti", claims)
    
    # def test_custom_jti(self):
    #     """Test custom JTI."""
    #     token = create_jwt(
    #         {"sub": "user-123"},
    #         self.secret,
    #         jti="custom-jti-123"
    #     )
        
    #     header, claims = decode_jwt(token, self.secret)
    #     print(header,claims)
    #     self.assertEqual(claims["jti"], "custom-jti-123")
    
    def test_kid_header(self):
        """Test key ID in header."""
        token = create_jwt(
            {"sub": "user-123"},
            self.secret,
            kid="key-123"
        )
        
        header, claims = decode_jwt(token, self.secret)
        self.assertEqual(header["kid"], "key-123")
    
    def test_extra_headers(self):
        """Test extra headers."""
        token = create_jwt(
            {"sub": "user-123"},
            self.secret,
            extra_headers={"x-custom": "value"}
        )
        
        header, claims = decode_jwt(token, self.secret)
        self.assertEqual(header["x-custom"], "value")
    
    def test_timedelta_expiration(self):
        """Test expiration with timedelta."""
        token = create_jwt(
            {"sub": "user-123"},
            self.secret,
            expires_in=dt.timedelta(seconds=1)
        )
        
        header, claims = decode_jwt(token, self.secret)
        self.assertIn("exp", claims)
        
        time.sleep(2)
        with self.assertRaises(ExpiredSignatureError):
            decode_jwt(token, self.secret)


if __name__ == "__main__":
    unittest.main(verbosity=2)

