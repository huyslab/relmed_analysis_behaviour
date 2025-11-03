# GitHub Actions Setup Guide

This guide explains how to set up automated weekly dashboard updates using GitHub Actions.

## Quick Setup (5 minutes)

### Step 1: Add GitHub Secrets

The workflow needs REDCap credentials to pull data and optionally a Slack webhook for notifications. Add these as GitHub secrets:

1. Go to your repository on GitHub
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**

Add these secrets:

| Secret Name | Value | Required? |
|------------|-------|-----------|
| `REDCAP_URL` | `https://redcap.slms.ucl.ac.uk/api/` | ✅ Required |
| `REDCAP_TOKEN_TRIAL1` | Your REDCap API token from the trial1 project | ✅ Required |
| `SLACK_WEBHOOK_URL` | Your Slack incoming webhook URL | Optional (for notifications) |

**Where to find your REDCap API token:**
- Log into REDCap
- Open your trial1 project
- Go to "API" in the left menu
- Click "Generate API Token" if you don't have one
- Copy the token string

**How to create a Slack webhook URL (Optional):**

1. Go to <https://api.slack.com/apps>
2. Click **"Create New App"** → **"From scratch"**
3. Give it a name (e.g., "Dashboard Updates") and select your workspace
4. In the app settings, go to **"Incoming Webhooks"**
5. Toggle **"Activate Incoming Webhooks"** to On
6. Click **"Add New Webhook to Workspace"**
7. Select the channel where you want notifications (e.g., #data-updates)
8. Click **"Allow"**
9. Copy the webhook URL (looks like `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXX`)
10. Add it as the `SLACK_WEBHOOK_URL` secret in GitHub

### Step 2: Enable Workflow Permissions

The workflow needs permission to commit and push changes:

1. Go to **Settings** → **Actions** → **General**
2. Scroll to "Workflow permissions"
3. Select **"Read and write permissions"**
4. Click **"Save"**

### Step 3: Test the Workflow

1. Go to the **Actions** tab
2. Click **"Weekly Dashboard Update"** in the left sidebar
3. Click **"Run workflow"** (button on the right)
4. Select branch: `main`
5. Click **"Run workflow"** (green button)

The workflow will run and you can watch the progress. It should:
- Pull the Docker image
- Run the Julia script
- Commit any updated figures
- Show a summary of what changed

## Automatic Schedule

Once set up, the workflow runs automatically:
- **Every Friday at 8:00 AM UTC**
- You'll see a new commit from `github-actions[bot]` when the dashboard is regenerated
  - If only the timestamp changed: commit message says "update dashboard timestamp"
  - If figures changed: commit message says "update behaviour dashboard visualizations"
- Check the Actions tab to see run history and logs
- **If Slack webhook is configured**: You'll receive a notification in Slack with a direct link to the updated dashboard (only when new figures are generated and running on schedule)

## Monitoring

### View Workflow Runs
- **Actions** tab → **Weekly Dashboard Update**
- Click any run to see:
  - Detailed logs for each step
  - Run summary showing what files changed
  - Any errors if the workflow failed

### Email Notifications
GitHub can email you when workflows fail:
- Go to your GitHub notification settings
- Enable "Actions" notifications

## Troubleshooting

### "Error: Resource not accessible by integration"
**Problem:** Workflow can't push commits

**Solution:** Enable write permissions (Step 2 above)

### "REDCap authentication failed"
**Problem:** Invalid or missing REDCap credentials

**Solutions:**
1. Verify secrets are named exactly as shown: `REDCAP_URL` and `REDCAP_TOKEN_TRIAL1`
2. Check your REDCap token is still valid in REDCap
3. Make sure there are no extra spaces in the secret values

### Workflow doesn't run on schedule
**Problem:** Scheduled runs aren't happening

**Solutions:**
1. Check repository is active (GitHub may disable schedules for inactive repos)
2. Verify the workflow file is on the `main` branch
3. Look for any errors in previous runs

### Docker image pull fails
**Problem:** Can't download the Docker image

**Solutions:**
1. Usually temporary - wait and try again
2. Check Docker Hub status: https://status.docker.com/
3. GitHub Actions has good network connectivity, this should be rare

### Permission denied errors
**Problem:** Errors like "mkdir: permission denied" or CmdStan compilation errors

**Solution:** The workflow has been configured to handle permissions by:
1. Creating required directories (`data`, `tmp`, `.cache/fontconfig`) before running Docker
2. Running Docker with `--user root` and `--entrypoint bash` to override default user switching
3. Making CmdStan directory writable for model compilation

If you see permission errors, verify the workflow includes these setup steps in the "Create required directories" step

## Customization

### Change Schedule

Edit [.github/workflows/weekly-dashboard-update.yml](.github/workflows/weekly-dashboard-update.yml):

```yaml
schedule:
  - cron: '0 8 * * 5'  # Friday at 8 AM UTC
```

Common schedules:
- Monday 9 AM: `'0 9 * * 1'`
- Every day midnight: `'0 0 * * *'`
- Monday & Thursday 8 AM: `'0 8 * * 1,4'`

Use [crontab.guru](https://crontab.guru/) to create cron expressions.

### Add More REDCap Projects

If you need data from additional REDCap projects:

1. Add a new secret: `REDCAP_TOKEN_PILOT1` (or whatever name)
2. Edit the workflow file, add to the `docker run` command:
   ```yaml
   -e REDCAP_TOKEN_pilot1="${{ secrets.REDCAP_TOKEN_PILOT1 }}" \
   ```

Note: The environment variable name format should match your `env.list` file format.

### Use Different Docker Image

If you update the Docker image version:

1. Edit the workflow file
2. Change `ghcr.io/huyslab/relmed:latest` to your new version
3. Commit the change

## Architecture

The workflow:
1. **Runs in GitHub's Ubuntu container**
2. **Pulls your Docker image** (same as local development)
3. **Mounts repository** as a volume to `/home/jovyan`
4. **Passes secrets** as environment variables to Docker
5. **Runs Julia script** inside Docker container
6. **Commits results** back to repository

This ensures consistency between:
- Local development environment
- GitHub Actions environment
- No need to install Julia/packages separately

## Security

- ✅ Secrets are encrypted by GitHub
- ✅ Secrets never appear in logs
- ✅ Docker container is isolated
- ✅ Only specific secrets are passed to Docker
- ✅ `GITHUB_TOKEN` is automatically scoped to this repository only
- ✅ Workflow requires explicit permissions to write

## Questions?

- See [.github/workflows/README.md](.github/workflows/README.md) for detailed documentation
- Check the Actions tab for run logs
- File an issue in the repository

## Related Documentation

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Repository README](../README.md)
