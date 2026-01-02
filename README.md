# Pure Python JWT Implementation

A pure Python implementation of JSON Web Tokens (JWT) using only the Python standard library. Supports HS256, HS384, and HS512 algorithms with comprehensive claim validation, custom validators, and a CLI interface.

## Features

- ✅ **Pure Python** - No third-party dependencies, only standard library
- ✅ **Multiple Algorithms** - HS256, HS384, HS512
- ✅ **Standard Claims** - exp, nbf, iat, iss, aud, sub, jti
- ✅ **Deterministic JTI** - Generate consistent JTI from payload
- ✅ **Custom Validators** - Built-in validators for scope, permissions, and custom claims
- ✅ **CLI Interface** - Command-line tools for signing and verifying tokens
- ✅ **Comprehensive Tests** - Full test suite included
- ✅ **Timing-Safe** - Uses constant-time comparison for signature verification

## Installation

No installation required! Just ensure you have Python 3.6+ and copy the files:

```bash
# Files needed:
# - generate_jwt.py
# - test_generate_jwt.py (optional, for testing)
# - README.md (this file)
```

## Quick Start

### Python API

```python
from generate_jwt import create_jwt, decode_jwt
import datetime as dt

# Create a token
secret = "your-secret-key"
token = create_jwt(
    payload={"sub": "user-123", "role": "admin"},
    key=secret,
    algorithm="HS256",
    expires_in=dt.timedelta(hours=1),
    issuer="https://your-app.com",
    audience="api"
)

print(f"Token: {token}")

# Verify the token
header, claims = decode_jwt(
    token,
    key=secret,
    expected_issuer="https://your-app.com",
    expected_audience="api"
)

print(f"Claims: {claims}")
```

### CLI Usage

#### Sign a Token

```bash
# Basic signing
python generate_jwt.py sign --key "secret" --payload '{"sub":"user123","role":"admin"}'

# With expiration (1 hour = 3600 seconds)
python generate_jwt.py sign --key "secret" --payload sub=user123,role=admin --expires-in 3600

# With issuer and audience
python generate_jwt.py sign --key "secret" --payload '{"sub":"user123"}' \
    --issuer "https://app.com" --audience "api"

# Deterministic JTI
python generate_jwt.py sign --key "secret" --payload sub=user123 \
    --deterministic-jti

# Full example
python generate_jwt.py sign \
    --key "secret" \
    --payload '{"sub":"user123","scope":"read write"}' \
    --algorithm HS256 \
    --expires-in 3600 \
    --issuer "https://app.com" \
    --audience "api" \
    --subject "user123"
```

#### Verify a Token

```bash
# Basic verification
python generate_jwt.py verify --key "secret" --token "eyJ..."

# With issuer/audience validation
python generate_jwt.py verify --key "secret" --token "eyJ..." \
    --issuer "https://app.com" --audience "api"

# With scope validation
python generate_jwt.py verify --key "secret" --token "eyJ..." \
    --require-scope read write

# With permission validation
python generate_jwt.py verify --key "secret" --token "eyJ..." \
    --require-permission admin user:write

# With custom claim validation
python generate_jwt.py verify --key "secret" --token "eyJ..." \
    --require-claim role admin

# Multiple validators
python generate_jwt.py verify --key "secret" --token "eyJ..." \
    --require-scope read \
    --require-permission admin \
    --require-claim role admin

# Pretty-printed output
python generate_jwt.py verify --key "secret" --token "eyJ..." --pretty
```

#### Peek at Token (No Verification)

```bash
# Decode without verification (for debugging)
python generate_jwt.py peek --token "eyJ..."

# Pretty-printed
python generate_jwt.py peek --token "eyJ..." --pretty
```

## API Reference

### `create_jwt()`

Create a JWT token.

**Parameters:**
- `payload` (dict): Initial payload claims
- `key` (str|bytes): HMAC secret key
- `algorithm` (str): 'HS256' | 'HS384' | 'HS512' (default: 'HS256')
- `expires_in` (int|timedelta, optional): Expiration time
- `not_before_in` (int|timedelta, optional): Not-before time
- `issued_at` (bool): Include 'iat' claim (default: True)
- `issuer` (str, optional): Issuer claim
- `audience` (str|list, optional): Audience claim
- `subject` (str, optional): Subject claim
- `jti` (str, optional): JWT ID (random if not provided)
- `deterministic_jti` (bool): Generate deterministic JTI (default: False)
- `jti_salt` (str, optional): Salt for deterministic JTI
- `kid` (str, optional): Key ID header
- `extra_headers` (dict, optional): Additional header fields

**Returns:** JWT token string

**Example:**
```python
token = create_jwt(
    payload={"user_id": 123},
    key="secret",
    algorithm="HS256",
    expires_in=3600,
    issuer="https://app.com",
    deterministic_jti=True
)
```

### `decode_jwt()`

Decode and verify a JWT token.

**Parameters:**
- `token` (str): JWT token string
- `key` (str|bytes): HMAC secret key
- `algorithms` (list, optional): Allowed algorithms (default: all HS*)
- `expected_issuer` (str, optional): Expected issuer
- `expected_audience` (str|list, optional): Expected audience
- `leeway` (int): Clock skew leeway in seconds (default: 0)
- `verify` (bool): Verify signature (default: True)
- `custom_validators` (list, optional): List of validator functions

**Returns:** Tuple of (header, payload) dictionaries

**Raises:** Various `JWTError` subclasses on failure

**Example:**
```python
header, claims = decode_jwt(
    token,
    key="secret",
    expected_issuer="https://app.com",
    custom_validators=[require_scope("read")]
)
```

### `peek_jwt()`

Decode token without verification (for debugging only).

**Parameters:**
- `token` (str): JWT token string

**Returns:** Tuple of (header, payload) dictionaries

**Warning:** Do not trust the results - this skips signature verification!

## Custom Validators

### `require_scope(*scopes)`

Validate that token has required scope(s). Supports both string (space-separated) and list formats.

**Example:**
```python
from generate_jwt import decode_jwt, require_scope

# Token has: {"scope": "read write admin"}
header, claims = decode_jwt(
    token,
    key="secret",
    custom_validators=[require_scope("read", "write")]
)
```

### `require_permission(*permissions)`

Validate that token has required permission(s). Expects 'permissions' claim as a list.

**Example:**
```python
from generate_jwt import decode_jwt, require_permission

# Token has: {"permissions": ["admin", "user:write"]}
header, claims = decode_jwt(
    token,
    key="secret",
    custom_validators=[require_permission("admin")]
)
```

### `require_claim(name, value=None)`

Validate that a claim exists and optionally matches a value.

**Example:**
```python
from generate_jwt import decode_jwt, require_claim

# Require claim exists
header, claims = decode_jwt(
    token,
    key="secret",
    custom_validators=[require_claim("role")]
)

# Require claim matches value
header, claims = decode_jwt(
    token,
    key="secret",
    custom_validators=[require_claim("role", "admin")]
)
```

### Custom Validator Function

You can create your own validators:

```python
def validate_custom_claim(payload: dict) -> None:
    """Custom validator that raises InvalidClaimError on failure."""
    if payload.get("department") != "engineering":
        raise InvalidClaimError("Must be in engineering department")

header, claims = decode_jwt(
    token,
    key="secret",
    custom_validators=[validate_custom_claim]
)
```

## Deterministic JTI

Generate consistent JTI values from the same payload and key. Useful for idempotent token generation.

**Example:**
```python
# Same payload + key = same JTI
token1 = create_jwt(
    {"user_id": 123},
    "secret",
    deterministic_jti=True
)

token2 = create_jwt(
    {"user_id": 123},
    "secret",
    deterministic_jti=True
)

# Both tokens will have the same JTI
header1, claims1 = decode_jwt(token1, "secret")
header2, claims2 = decode_jwt(token2, "secret")
assert claims1["jti"] == claims2["jti"]
```

**With Salt:**
```python
# Add salt for additional uniqueness
token = create_jwt(
    {"user_id": 123},
    "secret",
    deterministic_jti=True,
    jti_salt="session-abc"
)
```

## Standard Claims

### Registered Claims

- **exp** (Expiration): Token expiration time (Unix timestamp)
- **nbf** (Not Before): Token not valid before this time
- **iat** (Issued At): Token issuance time
- **iss** (Issuer): Token issuer identifier
- **aud** (Audience): Intended audience (string or list)
- **sub** (Subject): Subject identifier
- **jti** (JWT ID): Unique token identifier

### Setting Claims

```python
token = create_jwt(
    payload={"custom": "data"},
    key="secret",
    expires_in=3600,              # Expires in 1 hour
    not_before_in=60,            # Valid after 1 minute
    issued_at=True,              # Include iat (default)
    issuer="https://app.com",    # Set iss
    audience=["api", "web"],     # Set aud
    subject="user-123",          # Set sub
    jti="custom-id"              # Set jti
)
```

### Validating Claims

```python
header, claims = decode_jwt(
    token,
    key="secret",
    expected_issuer="https://app.com",
    expected_audience="api",
    leeway=10  # Allow 10 seconds clock skew
)
```

## Error Handling

The library raises specific exceptions for different error conditions:

```python
from generate_jwt import (
    JWTError,                    # Base exception
    InvalidTokenError,           # Malformed token
    InvalidAlgorithmError,       # Unsupported algorithm
    SignatureVerificationError,  # Signature mismatch
    ExpiredSignatureError,       # Token expired
    ImmatureSignatureError,      # Token not valid yet
    InvalidIssuerError,          # Issuer mismatch
    InvalidAudienceError,        # Audience mismatch
    InvalidClaimError,           # Custom claim validation failed
)

try:
    header, claims = decode_jwt(token, key)
except ExpiredSignatureError:
    print("Token has expired")
except SignatureVerificationError:
    print("Invalid signature")
except InvalidClaimError as e:
    print(f"Claim validation failed: {e}")
```

## Testing

Run the test suite:

```bash
python test_generate_jwt.py
```

The test suite includes:
- Basic JWT creation and verification
- All supported algorithms
- Standard claim validation
- Deterministic JTI generation
- Custom validators
- Edge cases and error handling

## Security Considerations

1. **Key Management**: Never hardcode secrets. Use environment variables or secure key management systems.

2. **Algorithm Selection**: Only use HS256/HS384/HS512 for symmetric keys. For asymmetric keys, use a library that supports RS256/ES256.

3. **Token Storage**: Store tokens securely. Consider httpOnly cookies for web applications.

4. **Expiration**: Always set reasonable expiration times.

5. **Leeway**: Use leeway sparingly and only when necessary for clock synchronization.

6. **Custom Validators**: Always validate custom claims to prevent privilege escalation.

## Examples

### Example 1: User Authentication Token

```python
import datetime as dt
from generate_jwt import create_jwt, decode_jwt, require_scope

# Create token
token = create_jwt(
    payload={
        "user_id": 123,
        "username": "alice",
        "scope": "read write"
    },
    key=os.environ["JWT_SECRET"],
    algorithm="HS256",
    expires_in=dt.timedelta(hours=24),
    issuer="https://api.example.com",
    audience="api",
    subject="user-123"
)

# Verify token
header, claims = decode_jwt(
    token,
    key=os.environ["JWT_SECRET"],
    expected_issuer="https://api.example.com",
    expected_audience="api",
    custom_validators=[require_scope("read")]
)
```

### Example 2: API Key with Permissions

```python
from generate_jwt import create_jwt, decode_jwt, require_permission

# Create API key token
token = create_jwt(
    payload={
        "api_key_id": "key-abc123",
        "permissions": ["users:read", "users:write", "admin"]
    },
    key=os.environ["JWT_SECRET"],
    expires_in=dt.timedelta(days=30),
    deterministic_jti=True  # Same key = same JTI
)

# Verify with permission check
header, claims = decode_jwt(
    token,
    key=os.environ["JWT_SECRET"],
    custom_validators=[require_permission("users:read")]
)
```

### Example 3: Session Token

```python
import secrets
from generate_jwt import create_jwt, decode_jwt, require_claim

# Create session token
session_id = secrets.token_urlsafe(32)
token = create_jwt(
    payload={
        "session_id": session_id,
        "ip_address": "192.168.1.1",
        "user_agent": "Mozilla/5.0..."
    },
    key=os.environ["JWT_SECRET"],
    expires_in=dt.timedelta(hours=2),
    jti=session_id  # Use session ID as JTI
)

# Verify session
header, claims = decode_jwt(
    token,
    key=os.environ["JWT_SECRET"],
    custom_validators=[
        require_claim("session_id", session_id),
        require_claim("ip_address", "192.168.1.1")
    ]
)
```

## CLI Examples

### Sign a Token

```bash
# Simple token
python generate_jwt.py sign \
    --key "my-secret" \
    --payload '{"user_id":123}'

# With all options
python generate_jwt.py sign \
    --key "my-secret" \
    --payload user_id=123,role=admin \
    --algorithm HS256 \
    --expires-in 3600 \
    --issuer "https://app.com" \
    --audience "api" \
    --subject "user-123" \
    --deterministic-jti
```

### Verify a Token

```bash
# Basic verification
python generate_jwt.py verify \
    --key "my-secret" \
    --token "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# With validators
python generate_jwt.py verify \
    --key "my-secret" \
    --token "eyJ..." \
    --issuer "https://app.com" \
    --audience "api" \
    --require-scope read write \
    --require-permission admin \
    --require-claim role admin \
    --pretty
```

## License

This implementation is provided as-is for educational and practical use. No warranty is provided.

## Contributing

Feel free to extend this implementation with additional features:
- RS256/ES256 support (asymmetric algorithms)
- JWK support
- Token refresh mechanisms
- Additional validators

## References

- [RFC 7519 - JSON Web Token (JWT)](https://tools.ietf.org/html/rfc7519)
- [RFC 7515 - JSON Web Signature (JWS)](https://tools.ietf.org/html/rfc7515)
- [jwt.io](https://jwt.io) - JWT debugger and information

