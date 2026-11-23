# IT handbook

## Error codes

| Code | Meaning | What to do |
|------|---------|------------|
| E-1001 | VPN server not reachable | Check your internet connection, then reconnect |
| E-1002 | VPN certificate expired | Renew the certificate in the self-service portal |
| E-2001 | Password older than 90 days | Choose a new password in the self-service portal |
| E-2002 | Account locked after 5 failed logins | Wait 15 minutes or call the service desk |
| E-3001 | Disk almost full | Delete old downloads or ask for more space |
| E-3002 | Backup failed | Leave the laptop on and plugged in overnight |
| E-4001 | Printer offline | Restart the printer and check its network cable |
| E-4002 | Badge not recognised by the printer | Register your badge again at the service desk |
| E-5001 | Mailbox full | Archive mails older than one year |
| E-5002 | Shared calendar not syncing | Remove the calendar and add it again |

## Setting up a new laptop

Before your first login you need:
- your badge
- the one-time password from your manager's email
- a wired network connection for the first update

Run the setup script once from a terminal:

```bash
# install the company certificates

sudo company-setup --certificates

# then join the device management

sudo company-setup --enroll
```

> **Warning** Never run the setup script on a private computer: it gives the company
> full control of the device.

## Lost or stolen devices

Report a lost laptop or phone to the service desk within one hour. The service desk
locks the device remotely and wipes it after 24 hours. You get a replacement on the
next working day, and your files come back from the last backup.
