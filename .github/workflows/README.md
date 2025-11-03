# GitHub Actions Workflows

This directory contains automated workflows for the RELMED repository.

## Weekly Dashboard Update

The `weekly-dashboard-update.yml` workflow automatically generates and updates the behaviour dashboard every week.

### Schedule

- **Runs**: Every Friday at 8:00 AM UTC
- **Can be triggered manually**: Go to Actions → Weekly Dashboard Update → Run workflow

### How It Works

1. **Pulls the Docker image**: Uses the same ` ghcr.io/huyslab/relmed:latest` image used for local development
2. **Sets up directories**: Creates required directories (`data`, `tmp`, `.cache/fontconfig`) with proper permissions
3. **Connects to REDCap**: Uses stored secrets to access the REDCap database
4. **Generates dashboard**: Runs `generate-figure-dashboard.jl` to create all visualizations
5. **Commits changes**: Automatically commits and pushes any updated figures to the repository

### Required GitHub Secrets

To enable this workflow, you need to configure the following secrets in your GitHub repository:

1. **Go to**: Repository Settings → Secrets and variables → Actions → New repository secret

2. **Add these secrets**:

   | Secret Name | Description | Required? |
   |------------|-------------|-----------|
   | `REDCAP_URL` | REDCap API endpoint URL | ✅ Required |
   | `REDCAP_TOKEN_TRIAL1` | REDCap API token for trial1 project | ✅ Required |
   | `SLACK_WEBHOOK_URL` | Slack incoming webhook URL for notifications | Optional |

### Setting Up GitHub Secrets

#### Step 1: Navigate to Repository Secrets
```
Your Repository → Settings → Secrets and variables → Actions → Repository secrets
```

#### Step 2: Add REDCAP_URL Secret
1. Click "New repository secret"
2. Name: `REDCAP_URL`
3. Secret: `https://redcap.slms.ucl.ac.uk/api/`
4. Click "Add secret"

#### Step 3: Add REDCAP_TOKEN_TRIAL1 Secret
1. Click "New repository secret"
2. Name: `REDCAP_TOKEN_TRIAL1`
3. Secret: Your REDCap API token (found in REDCap under API settings)
4. Click "Add secret"

#### Step 4: Add SLACK_WEBHOOK_URL Secret (Optional)

To receive Slack notifications when the dashboard updates:

1. Create a Slack Incoming Webhook:
   - Go to <https://api.slack.com/apps>
   - Click "Create New App" → "From scratch"
   - Name it (e.g., "Dashboard Updates") and select your workspace
   - Go to "Incoming Webhooks" and toggle it on
   - Click "Add New Webhook to Workspace"
   - Select a channel (e.g., #data-updates) and click "Allow"
   - Copy the webhook URL
2. Add it as a GitHub secret:
   - Name: `SLACK_WEBHOOK_URL`
   - Secret: The webhook URL you just copied
   - Click "Add secret"

**Note**: The workflow will send Slack notifications only when:

- It runs on schedule (not manual triggers)
- **New figures are generated** (not just timestamp updates)
- The `SLACK_WEBHOOK_URL` secret is configured

The workflow distinguishes between:

- **Meaningful changes**: SVG figure files are updated → Sends notification
- **Timestamp-only changes**: Only README.md timestamp updated → No notification

### Testing the Workflow

#### Option 1: Manual Trigger (Recommended for testing)
1. Go to the "Actions" tab in your GitHub repository
2. Click "Weekly Dashboard Update" in the left sidebar
3. Click "Run workflow" button (top right)
4. Select the branch (usually `main`)
5. Click "Run workflow"

#### Option 2: Wait for Scheduled Run
The workflow will automatically run every Friday at 8:00 AM UTC.

### Monitoring the Workflow

1. **View runs**: Go to Actions tab → Weekly Dashboard Update
2. **Check logs**: Click on any workflow run to see detailed logs
3. **View summary**: Each run creates a summary showing:
   - Whether changes were detected
   - What files were updated
   - Timestamp of the run

### Troubleshooting

#### Workflow fails with authentication error
- Check that `REDCAP_URL` and `REDCAP_TOKEN_TRIAL1` secrets are correctly set
- Verify your REDCap token is still valid in REDCap

#### Workflow fails to push changes
- Ensure the workflow has write permissions:
  - Go to Settings → Actions → General → Workflow permissions
  - Select "Read and write permissions"
  - Save

#### Docker image pull fails
- The workflow uses the public Docker image ` ghcr.io/huyslab/relmed:latest`
- If this image is updated, change the version in the workflow file

#### Permission denied errors
If you see errors like "mkdir: permission denied" or CmdStan compilation failures:
- The workflow creates necessary directories (`data`, `tmp`, `.cache/fontconfig`) before running Docker
- The Docker container runs with `--user root` and `--entrypoint bash` to ensure proper permissions
- CmdStan directory is made writable for model compilation
- These are handled automatically in the workflow configuration

### Customization

#### Change Schedule
Edit the cron expression in `weekly-dashboard-update.yml`:
```yaml
schedule:
  - cron: '0 8 * * 5'  # Minute Hour Day Month DayOfWeek
```

Examples:
- Every Monday at 9 AM: `'0 9 * * 1'`
- Every day at midnight: `'0 0 * * *'`
- Twice per week (Monday and Thursday at 8 AM): `'0 8 * * 1,4'`

Use [crontab.guru](https://crontab.guru/) to help create cron expressions.

#### Add More REDCap Projects
If you need to pull data from additional REDCap projects:

1. Add a new secret (e.g., `REDCAP_TOKEN_PILOT1`)
2. Update the workflow to pass it to Docker:
```yaml
-e REDCAP_TOKEN_pilot1="${{ secrets.REDCAP_TOKEN_PILOT1 }}" \
```

#### Adjust Docker Resources
To change the number of Julia threads:
```yaml
-e JULIA_NUM_THREADS=8 \
```

### Security Notes

- Secrets are encrypted and never exposed in logs
- The Docker container runs in an isolated environment
- Secrets are only passed to the container at runtime
- GitHub's built-in `GITHUB_TOKEN` is used for pushing commits (automatically scoped to this repository)

## Other Workflows

### Semantic Release Core Scripts
See the main [CORE_VERSIONING.md](../../CORE_VERSIONING.md) for details on how core scripts are automatically versioned.
