# Savant Runbook — Common Issues

## "Could not load knowledge base"
1. Check Railway logs for [TENANT] tag
2. If "No index found — starting background rebuild":
   - Wait 3 minutes, refresh
   - If still failing: check S3 for {tenant}/index.faiss
3. If S3 file exists but still failing:
   - Trigger manual rebuild from admin panel

## "Something went wrong" on every query
1. Check Railway logs for [ERROR] tag
2. Most likely: OpenAI API key expired or rate limited
3. Check: Railway dashboard → Variables → OPENAI_API_KEY
4. If rate limit: wait 60 seconds, retry

## Admin dashboard shows no data
1. Check LOG_FILE path in Railway logs — should be query_logs_{tenant}.csv
2. Likely: TENANT_PREFIX mismatch
3. Fix: ensure TENANT_PREFIX env var is set correctly in Railway

## Index is stale after new document upload
1. Check Railway logs for rebuild completion message
2. If rebuild never completed: trigger manual rebuild from admin panel
3. Check S3 for {tenant}/manifest.json — look at created_at field

## Railway service is down
1. Check UptimeRobot alert for how long
2. Check Railway status page: status.railway.app
3. If Railway issue: nothing to do, wait
4. If app crash: check Railway logs, likely a dependency import error

## New tenant setup checklist
1. Create clients/{tenant}.toml in repo
2. Push to main
3. Create new Railway service
4. Set all env vars (see Client Registry in Notion)
5. Upload at least one test doc to S3 under {tenant}/
6. Wait ~2 min for auto-rebuild
7. Run: make smoke TENANT_PREFIX={tenant}
8. Confirm 23/23 passing

## Weekly health check (run every Monday)
make smoke TENANT_PREFIX=demo
make smoke TENANT_PREFIX=innovim

## FastAPI Design Notes (Task 73)

When building the FastAPI layer, design with these
benchmark endpoints in mind — even as stubs initially:

  GET  /organizations/{tenant}/health-history
       Returns: list of gap_analysis_{ts}.json results over time

  GET  /organizations/{tenant}/gaps/resolved
       Returns: gaps identified vs rebuild events (lifecycle)

  POST /benchmark
       Body: {tenant, industry, org_size, frameworks}
       Returns: how this org compares to anonymized peers

  GET  /organizations/{tenant}/compliance-readiness
       Body: {framework: "DCAA"|"ISO-9001"|"CMMI-DEV-2"}
       Returns: coverage % against framework requirements

These endpoints don't need to be built now.
Designing FastAPI routes with them in mind avoids
a second rewrite when the benchmark layer is built.
