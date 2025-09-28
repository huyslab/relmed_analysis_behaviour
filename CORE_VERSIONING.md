# Core Scripts Semantic Versioning Guide

This repository uses **semantic versioning** for individual Julia scripts in the `core/` directory. Each script maintains its own independent version number following the format `MAJOR.MINOR.PATCH`.

## How It Works

### Automatic Versioning
- When you modify any Julia script in the `core/` directory and push to the `main` branch, the GitHub workflow automatically detects changes and updates version numbers.
- Each script has version information stored in comments at the top of the file:
  ```julia
  # Script description
  # Version: 1.2.3
  # Last Modified: 2025-09-28
  ```

### Version Bump Rules
The version bump is determined by your **commit message**:

| Commit Message Pattern | Version Bump | Example |
|------------------------|--------------|---------|
| `feat:` or `feat!:` | **Minor** (or Major if `!`) | `feat: add new parameter validation` → 1.1.0 |
| `BREAKING CHANGE:` | **Major** | Any commit with breaking change → 2.0.0 |
| `fix:`, `docs:`, `style:`, etc. | **Patch** | `fix: correct parameter calculation` → 1.0.1 |

### Git Tags and Releases
- Each script version gets its own Git tag: `script-name-v1.2.3`
- GitHub releases are automatically created for each version
- Release notes include the commit message and change details

## Best Practices

### Commit Message Format
Use [Conventional Commits](https://www.conventionalcommits.org/) format:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Examples:**
```bash
# Minor version bump (new feature)
git commit -m "feat(model_utils): add support for custom priors"

# Patch version bump (bug fix)  
git commit -m "fix(fetch_redcap): handle missing environment variables gracefully"

# Major version bump (breaking change)
git commit -m "feat!: change function signature for backward compatibility"
# or
git commit -m "feat: new parameter structure

BREAKING CHANGE: The parameter structure has changed from dict to named tuple"
```

### Working with Core Scripts

1. **Before Making Changes:**
   - Check the current version in the script header comment
   - Understand what type of change you're making (feature, fix, breaking)

2. **Making Changes:**
   - Edit the script as needed
   - Don't manually update the version number (it's automatic)
   - Write a clear, descriptive commit message

3. **After Pushing:**
   - The workflow will automatically update version numbers
   - Check the Actions tab to see the versioning process
   - New tags and releases will appear in the repository

### Version History
You can track version history for any script:
```bash
# See all versions of a specific script
git tag | grep "model_utils-v" | sort -V

# See changes between versions
git diff model_utils-v1.0.0..model_utils-v1.1.0 -- core/model_utils.jl

# View release notes for a specific version
gh release view model_utils-v1.1.0
```

## Workflow Configuration

The versioning is handled by `.github/workflows/semantic-release-core.yml` which:
1. Detects changes to `core/*.jl` files
2. Analyzes commit messages for version bump type
3. Updates version comments in modified files
4. Creates Git tags for each script version
5. Generates GitHub releases with automated release notes
6. Validates version format consistency

## Troubleshooting

**Version not updated after push:**
- Check that your commit message follows conventional format
- Ensure the push was to the `main` branch
- Check GitHub Actions for any workflow failures

**Want to skip versioning for a commit:**
- Add `[skip ci]` to your commit message
- This will skip the entire workflow

**Manual version override:**
- Generally not recommended as it breaks automation
- If needed, manually edit the version comment and push with `[skip ci]`
