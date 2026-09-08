# Security policy

PeopleOS is a local-first beta that processes sensitive workforce information. This repository does not establish an audited production security guarantee or a maintained security-support matrix for historical versions. Include the exact commit when reporting a suspected vulnerability.

## Reporting a vulnerability

Do not put exploitable vulnerability details, credentials or workforce data in a public issue. If this repository's GitHub **Security** tab offers **Report a vulnerability**, use that private reporting flow. Its availability must be checked on GitHub; this file does not enable it.

No dedicated private security contact is declared here. If private reporting is unavailable, open a minimal public issue asking the maintainers to provide a private reporting channel, without exploit instructions or sensitive details. Once a private channel is established, provide affected commit/version, impact, prerequisites and a minimal reproduction using fictional data. No response-time commitment is made by this policy.

For ordinary nonsecurity bugs, use the bug report template and remove employee data, tokens, private paths and identifiers from all attachments.

## Deployment boundary

The API defaults to loopback, with trusted local owner access. Keep the beta on a trusted personal machine and bind the development frontend to loopback as shown in the public beta guide. Remote API access requires explicit server-side token/role configuration; that boundary is not a multi-user identity provider. Protect the operating-system account and stored files.

Local-first does not mean every optional configuration is offline. Review configured LLM endpoints/providers before enabling synthesis with sensitive data. Do not assume logs, exports, backups or generated summaries are safe to publish. Git exclusions prevent accidental tracking of some runtime paths; they do not encrypt or erase files.

See the [Public Beta Guide](docs/PUBLIC_BETA_GUIDE.md) for supported scope, storage locations and the limits of metadata recovery.
