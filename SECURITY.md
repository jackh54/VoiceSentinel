# Security Policy

## Reporting a Vulnerability

If you find a security issue, please let us know. We appreciate responsible disclosure and will do our best to respond and fix problems quickly.

### Reporting Process

1. **DO NOT** create public GitHub issues for security vulnerabilities
2. Send your report privately to: [security@pandadevv.dev](mailto:security@pandadevv.dev)

### What to Include

Please include the following in your report:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact of the vulnerability
- Any suggested fixes (if available)
- Your contact information for follow-up questions

### Response Timeline

- Initial response: Within 48 hours
- Status update: Within 5 business days
- Fix timeline: Depends on complexity, typically within 3 days

### Scope

#### In Scope
- VoiceSentinel Processor (backend)
- Authentication mechanisms
- API endpoints
- Audio processing pipeline
- Data handling and storage

#### Out of Scope
- Issues in third-party dependencies (report to their security teams)
- Issues requiring physical access to servers
- Social engineering attacks
- DOS/DDOS attacks

### Safe Harbor

We consider security research conducted under this policy as "authorized" and will not initiate legal action against researchers who:

- Make a good faith effort to avoid privacy violations, data destruction, and service interruption
- Only interact with test accounts they own or have explicit permission to test
- Do not exploit a security issue for purposes other than verification
- Report any vulnerability found through our reporting process

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ |

## Security Best Practices

When deploying VoiceSentinel:

1. Always use strong, unique API keys
2. Keep all dependencies updated
3. Monitor system logs regularly
4. Follow the principle of least privilege
5. Use secure communication channels (HTTPS)
6. Implement rate limiting
7. Regular security audits

## Security Features

VoiceSentinel includes several security features:

- API key authentication
- Rate limiting
- Secure audio file handling
- Automatic cleanup of temporary files
- Input validation and sanitization
- Configurable access controls

## Contact

- Security: [security@pandadevv.dev](mailto:security@pandadevv.dev)
- General: [Discord](https://discord.gg/JAJyuzdgHZ)

---

Last updated: 2025-07-16