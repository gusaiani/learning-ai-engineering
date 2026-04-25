# NovaCRM — Account Management

## Team members and roles

NovaCRM supports three roles:

| Role | Permissions |
|------|------------|
| **Admin** | Full access. Manage billing, API keys, team members, SSO, data export. Can delete the workspace. |
| **Member** | Create/edit/delete contacts, deals, activities. Run reports. Cannot manage billing or team settings. |
| **Viewer** | Read-only access to contacts, deals, reports. Cannot create or modify anything. |

### Inviting team members

1. Go to **Settings > Team**
2. Click **Invite member**
3. Enter their email address and select a role
4. They receive an email with a link to join (expires in 7 days)

Free plan is single-user. Pro allows up to 25 users. Enterprise is unlimited.

### Changing roles

Admins can change any member's role at Settings > Team. Role changes take effect immediately. You cannot remove the last admin — promote someone else first.

### Removing members

Admins can remove team members at Settings > Team. When a member is removed:
- Their contacts and deals are reassigned to the admin who removed them
- Their API keys are revoked immediately
- Their login is disabled
- Their activity history is preserved (for audit purposes)

## Single Sign-On (SSO)

SSO is available on the **Enterprise plan** only.

### Supported providers
- SAML 2.0 (Okta, Azure AD, OneLogin, any SAML-compliant IdP)
- OpenID Connect (OIDC) (Google Workspace, Auth0)

### Setup

1. Go to **Settings > Security > SSO**
2. Select your provider type (SAML or OIDC)
3. Enter your IdP metadata URL (SAML) or client ID + discovery URL (OIDC)
4. NovaCRM provides an ACS URL and Entity ID — configure these in your IdP
5. Test the connection with the **Test SSO** button
6. Enable **Enforce SSO** to require SSO for all team members (optional)

When SSO is enforced, password login is disabled for all non-admin users. Admins retain password login as a backup in case of IdP outages.

### Just-in-time provisioning

When SSO is enabled, new users who authenticate via your IdP are automatically created as Members in NovaCRM. Admins can change their role after first login.

## Data export

All plans can export their data.

### How to export

1. Go to **Settings > Data > Export**
2. Select what to export: Contacts, Deals, Activities, or All
3. Choose format: CSV or JSON
4. Click **Export** — you'll receive a download link via email within 5 minutes

### What's included

- **Contacts:** All fields including custom fields, tags, and notes
- **Deals:** All fields, stage history, associated contacts
- **Activities:** Calls, emails, notes with timestamps and participants
- **Files:** Attached files are exported as a ZIP archive (separate from the data export)

Export files are available for download for 7 days after generation.

### API data export

Enterprise users can also export data via the API:

```
POST /v1/exports
{
  "type": "contacts",
  "format": "json",
  "filters": {"created_after": "2025-01-01"}
}
```

The response includes a `download_url` field (available once the export completes, typically 1–5 minutes).

## Account deletion

To delete your NovaCRM workspace:

1. Go to **Settings > Account > Delete workspace**
2. Type your workspace name to confirm
3. All data is permanently deleted within 30 days

Only the workspace owner (the original admin who created the account) can delete the workspace. Deletion cannot be undone. We recommend exporting your data first.

For GDPR data deletion requests: contact privacy@novacrm.com with the subject "GDPR Deletion Request" and include the workspace ID and requestor's email. We process these within 72 hours.

## Security settings (Enterprise)

Enterprise plans have additional security features:

- **Audit logs:** View all team actions (logins, data changes, exports) at Settings > Security > Audit Log. Logs are retained for 1 year.
- **IP allowlisting:** Restrict dashboard and API access to specific IP addresses or CIDR ranges.
- **Two-factor authentication:** Enforce 2FA for all team members.
- **Session timeout:** Configure automatic logout after inactivity (default: 8 hours, range: 15 minutes to 24 hours).
