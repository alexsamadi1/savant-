# Client Configs

Each file in this folder is a per-tenant config override.
File naming: `{TENANT_PREFIX}.toml`

How it works:
- `config.toml` is the base config (defaults for all tenants)
- `clients/{tenant}.toml` overrides any key in config.toml for that tenant
- The active tenant is set via the TENANT_PREFIX environment variable
- Missing keys fall back to config.toml defaults silently

To add a new tenant:
1. Copy clients/innovim.toml → clients/{tenant}.toml
2. Update all brand, contact, onboarding, and assistant values
3. Create a new Railway service with TENANT_PREFIX={tenant}
4. Upload docs to S3 under {tenant}/ prefix
5. Run: make smoke TENANT_PREFIX={tenant}

Current tenants: demo, innovim, potencia (pending)
