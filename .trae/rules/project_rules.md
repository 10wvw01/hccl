# Project Rules

## Environment Variables

When running any shell commands, always prepend the following environment variable exports:

```bash
export PATH="/home/crx/code/hccl_three_sequence_rs/hccl_2039/.npm-global/bin:/home/crx/code/hccl_three_sequence_rs/hccl_2039/node-v20.19.0-linux-x64/bin:$PATH"
export XDG_CONFIG_HOME="/home/crx/code/hccl_three_sequence_rs/hccl_2039/.config"
export npm_config_cache="/home/crx/code/hccl_three_sequence_rs/hccl_2039/.npm-cache"
export npm_config_prefix="/home/crx/code/hccl_three_sequence_rs/hccl_2039/.npm-global"
```

This is required because:
- Node.js is installed locally at `node-v20.19.0-linux-x64/` (not system-wide)
- OpenSpec CLI is installed at `.npm-global/bin/openspec`
- Home directory `/home/crx/` is read-only, so npm cache and config must use project-local paths
