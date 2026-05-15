# Account management

Customers manage their account from **Settings → Account**. From there they
can update name, email, password, two-factor authentication, and connected
integrations.

Email changes require confirmation from both the old and new addresses. If
the customer no longer has access to the old address, support can override
this after verifying identity via the last 4 digits of the payment card on
file plus the most recent invoice number.

Password resets are self-service via the login page. Reset links expire in
60 minutes. We never ask for or send passwords over email or chat.

Account deletion is self-service from **Settings → Account → Delete
account**. Deletion is soft for 30 days (recoverable by support), then
hard-deleted with no recovery path. Outstanding invoices must be paid before
deletion.

Two-factor authentication uses TOTP. If a customer loses their authenticator,
they recover with a backup code; if they have no backup code, support
verifies identity (card last 4 + invoice number) and disables 2FA.
