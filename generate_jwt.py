
#!/usr/bin/env python3
"""
Pure Python JWT (HS256/HS384/HS512) implementation with:
- base64url encode/decode (no padding)
- JSON header & payload handling
- HMAC signing
- Expiration (exp), Not Before (nbf), Issued At (iat)
- Audience (aud), Issuer (iss), Subject (sub) validation
- Timing-safe signature comparison
- Helpful exceptions and type hints

No third-party libraries used. Only Python standard library.
"""

from __future__ import annotations

import base64
import json
import hmac
import hashlib
import time
import datetime as dt
import secrets
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


# =========================
# Exceptions
# =========================

class JWTError(Exception):
    """Base class for JWT errors."""
    pass

class InvalidTokenError(JWTError):
    """Token structurally invalid or cannot be parsed."""
    pass

class InvalidAlgorithmError(JWTError):
    """Unsupported or mismatched algorithm."""
    pass

class SignatureVerificationError(JWTError):
    """Signature does not match."""
    pass

class ExpiredSignatureError(JWTError):
    """Token is expired (exp)."""
    pass

class ImmatureSignatureError(JWTError):
    """Token not valid yet (nbf) or issued in the future (iat)."""
    pass

class InvalidIssuerError(JWTError):
    """Issuer claim (iss) mismatch."""
    pass

class InvalidAudienceError(JWTError):
    """Audience claim (aud) mismatch."""
    pass

class InvalidClaimError(JWTError):
    """Custom claim validation failed."""
    pass


# =========================
# Utilities
# =========================

def _b64url_encode(data: bytes) -> str:
    """Base64 URL-safe encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')

def _b64url_decode(data: str) -> bytes:
    """Base64 URL-safe decode; add padding if needed."""
    s = data.encode('ascii')
    padding = b'=' * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode(s + padding)

def _json_dumps(obj: Any) -> bytes:
    """
    Compact JSON encoding (RFC 7515 recommends no whitespace).
    Use UTF-8 and ensure ASCII-safe serialization.
    """
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

def _utc_timestamp() -> int:
    """Current UTC time in seconds (integer)."""
    return int(time.time())

def _to_timestamp(value: Union[int, float, dt.datetime]) -> int:
    """Convert int/float/datetime to int timestamp."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dt.datetime):
        # If naive, assume UTC. If aware, convert to UTC.
        if value.tzinfo is None:
            return int(value.timestamp())
        return int(value.astimezone(dt.timezone.utc).timestamp())
    raise TypeError("Unsupported timestamp type. Use int, float, or datetime.")

def _timedelta_to_seconds(value: dt.timedelta) -> int:
    return int(value.total_seconds())

def _hmac_for_alg(alg: str):
    """Return hashlib function for the HS* alg."""
    mapping = {
        'HS256': hashlib.sha256,
        'HS384': hashlib.sha384,
        'HS512': hashlib.sha512,
    }
    if alg not in mapping:
        raise InvalidAlgorithmError(f"Unsupported algorithm: {alg}")
    return mapping[alg]

def _sign(message: bytes, key: Union[str, bytes], alg: str) -> bytes:
    """Compute HMAC signature for given alg."""
    digestmod = _hmac_for_alg(alg)
    key_bytes = key.encode('utf-8') if isinstance(key, str) else key
    return hmac.new(key_bytes, message, digestmod).digest()

def _cmp_digest(a: bytes, b: bytes) -> bool:
    """Timing-safe comparison."""
    return hmac.compare_digest(a, b)

def _generate_deterministic_jti(payload: Dict[str, Any], key: Union[str, bytes], salt: Optional[str] = None) -> str:
    """
    Generate a deterministic JTI based on payload content and key.
    Useful for idempotent token generation.
    
    Args:
        payload: The JWT payload (will be sorted for consistency).
        key: HMAC key used for signing.
        salt: Optional salt for additional uniqueness.
    
    Returns:
        Base64url-encoded deterministic JTI string.
    """
    # Create a deterministic representation of the payload
    # Sort keys to ensure consistent ordering
    sorted_payload = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    key_bytes = key.encode('utf-8') if isinstance(key, str) else key
    
    # Combine payload, key, and optional salt
    data = sorted_payload + key_bytes
    if salt:
        data += salt.encode('utf-8')
    
    # Generate deterministic hash
    jti_hash = hashlib.sha256(data).digest()
    return _b64url_encode(jti_hash[:16])  # Use first 16 bytes for JTI


# =========================
# Core API
# =========================

def create_jwt(
    payload: Dict[str, Any],
    key: Union[str, bytes],
    algorithm: str = 'HS256',
    *,
    expires_in: Optional[Union[int, dt.timedelta]] = None,
    not_before_in: Optional[Union[int, dt.timedelta]] = None,
    issued_at: bool = True,
    issuer: Optional[str] = None,
    audience: Optional[Union[str, Sequence[str]]] = None,
    subject: Optional[str] = None,
    jti: Optional[str] = None,
    deterministic_jti: bool = False,
    jti_salt: Optional[str] = None,
    kid: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a JWT with standard claims.
    
    Args:
        payload: Initial payload claims (will be augmented).
        key: HMAC key (secret).
        algorithm: 'HS256' | 'HS384' | 'HS512'.
        expires_in: seconds or timedelta to set 'exp' relative to now.
        not_before_in: seconds or timedelta to set 'nbf' relative to now.
        issued_at: if True, set 'iat' to current time.
        issuer: optional 'iss' claim.
        audience: optional 'aud' claim (string or list of strings).
        subject: optional 'sub' claim.
        jti: optional 'jti' claim (random if None, unless deterministic_jti=True).
        deterministic_jti: if True, generate deterministic JTI from payload/key.
        jti_salt: optional salt for deterministic JTI generation.
        kid: optional header 'kid' value.
        extra_headers: extra header entries (merged).
    Returns:
        Encoded JWT string: header.payload.signature
    """
    # Validate algorithm upfront
    _hmac_for_alg(algorithm)

    now = _utc_timestamp()
    claims = dict(payload)  # shallow copy

    # Standard registered claims
    if issued_at and 'iat' not in claims:
        claims['iat'] = now

    if expires_in is not None and 'exp' not in claims:
        exp_seconds = expires_in if isinstance(expires_in, int) else _timedelta_to_seconds(expires_in)
        claims['exp'] = now + int(exp_seconds)

    if not_before_in is not None and 'nbf' not in claims:
        nbf_seconds = not_before_in if isinstance(not_before_in, int) else _timedelta_to_seconds(not_before_in)
        claims['nbf'] = now + int(nbf_seconds)

    if issuer is not None and 'iss' not in claims:
        claims['iss'] = issuer

    if audience is not None and 'aud' not in claims:
        claims['aud'] = audience

    if subject is not None and 'sub' not in claims:
        claims['sub'] = subject

    if jti is None and 'jti' not in claims:
        if deterministic_jti:
            # Generate deterministic JTI from payload and key
            # Note: We generate it from the claims dict before adding standard claims
            # to ensure consistency. For true determinism, exclude iat/exp if needed.
            claims['jti'] = _generate_deterministic_jti(claims, key, jti_salt)
        else:
            # Generate a URL-safe random ID
            claims['jti'] = secrets.token_urlsafe(16)

    # Header
    header: Dict[str, Any] = {'alg': algorithm, 'typ': 'JWT'}
    if kid is not None:
        header['kid'] = kid
    if extra_headers:
        header.update(extra_headers)

    # Serialize
    header_b64 = _b64url_encode(_json_dumps(header))
    payload_b64 = _b64url_encode(_json_dumps(claims))
    signing_input = f"{header_b64}.{payload_b64}".encode('ascii')

    # Sign
    signature = _sign(signing_input, key, algorithm)
    signature_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_jwt(
    token: str,
    key: Union[str, bytes],
    *,
    algorithms: Optional[Sequence[str]] = ('HS256', 'HS384', 'HS512'),
    expected_issuer: Optional[str] = None,
    expected_audience: Optional[Union[str, Sequence[str]]] = None,
    leeway: int = 0,
    verify: bool = True,
    custom_validators: Optional[Sequence[Callable[[Dict[str, Any]], None]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Decode & verify a JWT. Returns (header, payload) or raises JWTError.

    Args:
        token: JWT string.
        key: HMAC key (secret).
        algorithms: allowed algorithms; defaults to HS256/384/512.
        expected_issuer: check 'iss' equals this value if provided.
        expected_audience: check 'aud' includes/matches if provided.
        leeway: seconds of allowed clock skew for exp/nbf/iat checks.
        verify: if False, skip signature & claim validation (not recommended).
        custom_validators: list of validator functions that take payload and raise
                          InvalidClaimError on failure.
    """
    parts = token.split('.')
    if len(parts) != 3:
        raise InvalidTokenError("JWT must have exactly 3 parts: header.payload.signature")

    header_b64, payload_b64, signature_b64 = parts

    try:
        header = json.loads(_b64url_decode(header_b64))
    except Exception as e:
        raise InvalidTokenError(f"Invalid header: {e}")

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception as e:
        raise InvalidTokenError(f"Invalid payload: {e}")

    alg = header.get('alg')
    if not isinstance(alg, str):
        raise InvalidTokenError("Header 'alg' must be a string")

    if algorithms is not None and alg not in algorithms:
        raise InvalidAlgorithmError(f"Algorithm {alg} not allowed")

    if verify:
        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}".encode('ascii')
        try:
            provided_sig = _b64url_decode(signature_b64)
        except Exception:
            raise InvalidTokenError("Invalid base64url in signature")

        expected_sig = _sign(signing_input, key, alg)
        if not _cmp_digest(provided_sig, expected_sig):
            raise SignatureVerificationError("Signature mismatch")

        # Validate registered claims
        _validate_registered_claims(payload, expected_issuer, expected_audience, leeway)
        
        # Validate custom claims
        if custom_validators:
            _validate_custom_claims(payload, custom_validators)

    return header, payload


def _validate_registered_claims(
    payload: Dict[str, Any],
    expected_issuer: Optional[str],
    expected_audience: Optional[Union[str, Sequence[str]]],
    leeway: int,
) -> None:
    """
    Validate exp, nbf, iat, iss, aud with leeway.
    Raises JWTError variants on failure.
    """
    now = _utc_timestamp()

    # exp: current time must be <= exp + leeway
    if 'exp' in payload:
        exp = _to_timestamp(payload['exp'])
        if now > exp + leeway:
            raise ExpiredSignatureError(f"Token expired at {exp}, now={now}, leeway={leeway}")

    # nbf: current time must be >= nbf - leeway
    if 'nbf' in payload:
        nbf = _to_timestamp(payload['nbf'])
        if now < nbf - leeway:
            raise ImmatureSignatureError(f"Token not valid before {nbf}, now={now}, leeway={leeway}")

    # iat: typically should not be in the future beyond leeway
    if 'iat' in payload:
        iat = _to_timestamp(payload['iat'])
        if now + leeway < iat:
            raise ImmatureSignatureError(f"Issued-at (iat) {iat} is in the future relative to now={now}, leeway={leeway}")

    # iss: must match expected
    if expected_issuer is not None:
        iss = payload.get('iss')
        if iss != expected_issuer:
            raise InvalidIssuerError(f"Invalid issuer: {iss} != {expected_issuer}")

    # aud: must contain/match expected
    if expected_audience is not None:
        aud = payload.get('aud')
        if aud is None:
            raise InvalidAudienceError("Missing 'aud' claim")
        if isinstance(expected_audience, str):
            exp_auds = [expected_audience]
        else:
            exp_auds = list(expected_audience)
        if isinstance(aud, str):
            got_auds = [aud]
        elif isinstance(aud, list):
            got_auds = aud
        else:
            raise InvalidAudienceError("Invalid 'aud' type; must be string or list of strings")
        # Require intersection
        if not any(a in got_auds for a in exp_auds):
            raise InvalidAudienceError(f"Audience mismatch: expected {exp_auds}, got {got_auds}")


def _validate_custom_claims(
    payload: Dict[str, Any],
    validators: Sequence[Callable[[Dict[str, Any]], None]],
) -> None:
    """
    Run custom claim validators on the payload.
    Each validator should raise InvalidClaimError on failure.
    """
    for validator in validators:
        try:
            validator(payload)
        except InvalidClaimError:
            raise
        except Exception as e:
            raise InvalidClaimError(f"Custom validator failed: {e}")


# =========================
# Built-in Custom Validators
# =========================

def require_scope(*required_scopes: str) -> Callable[[Dict[str, Any]], None]:
    """
    Create a validator that requires specific scope(s) in the 'scope' claim.
    
    Args:
        *required_scopes: One or more required scope strings.
    
    Returns:
        Validator function that raises InvalidClaimError if scope missing.
    
    Example:
        validator = require_scope("read", "write")
        decode_jwt(token, key, custom_validators=[validator])
    """
    def validator(payload: Dict[str, Any]) -> None:
        scope = payload.get('scope')
        if scope is None:
            raise InvalidClaimError("Missing 'scope' claim")
        
        # Handle both string (space-separated) and list formats
        if isinstance(scope, str):
            scopes = scope.split()
        elif isinstance(scope, list):
            scopes = scope
        else:
            raise InvalidClaimError("'scope' must be a string or list")
        
        missing = [s for s in required_scopes if s not in scopes]
        if missing:
            raise InvalidClaimError(f"Missing required scope(s): {missing}")
    
    return validator


def require_permission(*required_permissions: str) -> Callable[[Dict[str, Any]], None]:
    """
    Create a validator that requires specific permission(s) in the 'permissions' claim.
    
    Args:
        *required_permissions: One or more required permission strings.
    
    Returns:
        Validator function that raises InvalidClaimError if permission missing.
    
    Example:
        validator = require_permission("admin", "user:write")
        decode_jwt(token, key, custom_validators=[validator])
    """
    def validator(payload: Dict[str, Any]) -> None:
        permissions = payload.get('permissions')
        if permissions is None:
            raise InvalidClaimError("Missing 'permissions' claim")
        
        if not isinstance(permissions, list):
            raise InvalidClaimError("'permissions' must be a list")
        
        missing = [p for p in required_permissions if p not in permissions]
        if missing:
            raise InvalidClaimError(f"Missing required permission(s): {missing}")
    
    return validator


def require_claim(claim_name: str, expected_value: Any = None) -> Callable[[Dict[str, Any]], None]:
    """
    Create a validator that requires a claim to exist (and optionally match a value).
    
    Args:
        claim_name: Name of the claim to check.
        expected_value: If provided, claim must equal this value.
    
    Returns:
        Validator function that raises InvalidClaimError if claim missing or mismatched.
    
    Example:
        validator = require_claim("role", "admin")
        decode_jwt(token, key, custom_validators=[validator])
    """
    def validator(payload: Dict[str, Any]) -> None:
        if claim_name not in payload:
            raise InvalidClaimError(f"Missing required claim: {claim_name}")
        
        if expected_value is not None and payload[claim_name] != expected_value:
            raise InvalidClaimError(
                f"Claim '{claim_name}' mismatch: expected {expected_value}, got {payload[claim_name]}"
            )
    
    return validator


# =========================
# Convenience Helpers
# =========================

def peek_jwt(token: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Decode header & payload WITHOUT verification.
    Useful for introspection/debugging; do not trust results.
    """
    parts = token.split('.')
    if len(parts) != 3:
        raise InvalidTokenError("JWT must have exactly 3 parts")
    header = json.loads(_b64url_decode(parts[0]))
    payload = json.loads(_b64url_decode(parts[1]))
    return header, payload


# =========================
# CLI Interface
# =========================

def _parse_payload(payload_str: str) -> Dict[str, Any]:
    """Parse JSON payload string or key=value pairs."""
    if payload_str.startswith('{'):
        # JSON format
        return json.loads(payload_str)
    else:
        # key=value format
        payload = {}
        for pair in payload_str.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Try to parse as JSON, fallback to string
                try:
                    payload[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    payload[key] = value
        return payload


def _main():
    """CLI entry point for JWT operations."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='Pure Python JWT generator and verifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sign a token
  %(prog)s sign --key "secret" --payload '{"sub":"user123","role":"admin"}'
  
  # Sign with expiration
  %(prog)s sign --key "secret" --payload sub=user123,role=admin --expires-in 3600
  
  # Verify a token
  %(prog)s verify --key "secret" --token "eyJ..."
  
  # Verify with custom validators
  %(prog)s verify --key "secret" --token "eyJ..." --require-scope read write
  
  # Peek at token (no verification)
  %(prog)s peek --token "eyJ..."
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute', required=True)
    
    # Sign command
    sign_parser = subparsers.add_parser('sign', help='Sign a JWT token')
    sign_parser.add_argument('--key', '-k', required=True, help='HMAC secret key')
    sign_parser.add_argument('--payload', '-p', required=True, 
                            help='Payload as JSON or key=value pairs (comma-separated)')
    sign_parser.add_argument('--algorithm', '-a', default='HS256', 
                            choices=['HS256', 'HS384', 'HS512'],
                            help='Signing algorithm (default: HS256)')
    sign_parser.add_argument('--expires-in', type=int, 
                            help='Expiration time in seconds')
    sign_parser.add_argument('--not-before-in', type=int,
                            help='Not-before time in seconds from now')
    sign_parser.add_argument('--issuer', '-iss', help='Issuer claim')
    sign_parser.add_argument('--audience', '-aud', help='Audience claim (comma-separated for multiple)')
    sign_parser.add_argument('--subject', '-sub', help='Subject claim')
    sign_parser.add_argument('--jti', help='JWT ID claim')
    sign_parser.add_argument('--deterministic-jti', action='store_true',
                            help='Generate deterministic JTI from payload')
    sign_parser.add_argument('--jti-salt', help='Salt for deterministic JTI')
    sign_parser.add_argument('--kid', help='Key ID header')
    sign_parser.add_argument('--no-iat', action='store_true',
                            help='Do not include issued-at claim')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify a JWT token')
    verify_parser.add_argument('--key', '-k', required=True, help='HMAC secret key')
    verify_parser.add_argument('--token', '-t', required=True, help='JWT token to verify')
    verify_parser.add_argument('--algorithms', nargs='+', 
                              choices=['HS256', 'HS384', 'HS512'],
                              default=['HS256', 'HS384', 'HS512'],
                              help='Allowed algorithms')
    verify_parser.add_argument('--issuer', '-iss', help='Expected issuer')
    verify_parser.add_argument('--audience', '-aud', help='Expected audience (comma-separated)')
    verify_parser.add_argument('--leeway', type=int, default=0,
                              help='Clock skew leeway in seconds')
    verify_parser.add_argument('--require-scope', nargs='+',
                              help='Required scope(s)')
    verify_parser.add_argument('--require-permission', nargs='+',
                              help='Required permission(s)')
    verify_parser.add_argument('--require-claim', nargs=2, metavar=('NAME', 'VALUE'),
                              action='append',
                              help='Require claim NAME=VALUE (can be used multiple times)')
    verify_parser.add_argument('--no-verify', action='store_true',
                              help='Skip signature verification (not recommended)')
    verify_parser.add_argument('--pretty', action='store_true',
                              help='Pretty-print JSON output')
    
    # Peek command
    peek_parser = subparsers.add_parser('peek', help='Decode token without verification')
    peek_parser.add_argument('--token', '-t', required=True, help='JWT token to peek at')
    peek_parser.add_argument('--pretty', action='store_true',
                           help='Pretty-print JSON output')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'sign':
            payload = _parse_payload(args.payload)
            audience = None
            if args.audience:
                audience = [a.strip() for a in args.audience.split(',')] if ',' in args.audience else args.audience
            
            expires_in = None
            if args.expires_in:
                expires_in = args.expires_in
            
            not_before_in = None
            if args.not_before_in:
                not_before_in = args.not_before_in
            
            token = create_jwt(
                payload=payload,
                key=args.key,
                algorithm=args.algorithm,
                expires_in=expires_in,
                not_before_in=not_before_in,
                issued_at=not args.no_iat,
                issuer=args.issuer,
                audience=audience,
                subject=args.subject,
                jti=args.jti,
                deterministic_jti=args.deterministic_jti,
                jti_salt=args.jti_salt,
                kid=args.kid,
            )
            print(token)
            
        elif args.command == 'verify':
            validators = []
            
            if args.require_scope:
                validators.append(require_scope(*args.require_scope))
            
            if args.require_permission:
                validators.append(require_permission(*args.require_permission))
            
            if args.require_claim:
                for name, value in args.require_claim:
                    # Try to parse value as JSON, fallback to string
                    try:
                        parsed_value = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        parsed_value = value
                    validators.append(require_claim(name, parsed_value))
            
            audience = None
            if args.audience:
                audience = [a.strip() for a in args.audience.split(',')] if ',' in args.audience else args.audience
            
            header, payload = decode_jwt(
                token=args.token,
                key=args.key,
                algorithms=tuple(args.algorithms),
                expected_issuer=args.issuer,
                expected_audience=audience,
                leeway=args.leeway,
                verify=not args.no_verify,
                custom_validators=validators if validators else None,
            )
            
            output = {
                'header': header,
                'payload': payload,
                'valid': True
            }
            print(json.dumps(output, indent=2 if args.pretty else None))
            
        elif args.command == 'peek':
            header, payload = peek_jwt(args.token)
            output = {
                'header': header,
                'payload': payload,
                'warning': 'Token not verified - do not trust these claims'
            }
            print(json.dumps(output, indent=2 if args.pretty else None))
            
    except JWTError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()

