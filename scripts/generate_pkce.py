import base64
import hashlib
import secrets

# Generate a secure random code verifier (43-128 characters)
code_verifier = secrets.token_urlsafe(64)

# Generate the SHA256 hash
digest = hashlib.sha256(code_verifier.encode()).digest()

# Convert to Base64 URL-safe format without '=' padding
code_challenge = (
    base64.urlsafe_b64encode(digest)
    .decode()
    .rstrip("=")
)

print("\n===== PKCE VALUES =====\n")
print(f"Code Verifier : {code_verifier}\n")
print(f"Code Challenge: {code_challenge}\n")
print("Method        : S256")

