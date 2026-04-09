run-demo:
	TENANT_PREFIX=demo streamlit run app.py

run-innovim:
	TENANT_PREFIX=innovim streamlit run app.py

eval:
	TENANT_PREFIX=demo python run_eval.py

smoke:
	TENANT_PREFIX=demo python test_smoke.py

TENANT_STATUS_SCRIPT = \
import boto3;\
from tools.s3_utils import get_secret, get_tenant_prefix;\
import os;\
prefix = os.environ['TENANT_PREFIX'];\
s3 = boto3.client('s3', aws_access_key_id=get_secret('AWS_ACCESS_KEY_ID'), aws_secret_access_key=get_secret('AWS_SECRET_ACCESS_KEY'), region_name=get_secret('AWS_REGION'));\
docs = s3.list_objects_v2(Bucket=get_secret('S3_DOCS_BUCKET'), Prefix=prefix+'/');\
idx = s3.list_objects_v2(Bucket=get_secret('S3_INDEX_BUCKET'), Prefix=prefix+'/');\
doc_count = len([o for o in docs.get('Contents',[]) if o['Key'].endswith(('.pdf','.docx'))]);\
idx_count = len(idx.get('Contents',[]));\
print(f'  Docs: {doc_count} | Index files: {idx_count}')

status:
	@echo "=== Demo ==="
	@TENANT_PREFIX=demo python3 -c "$(TENANT_STATUS_SCRIPT)"
	@echo "=== Innovim ==="
	@TENANT_PREFIX=innovim python3 -c "$(TENANT_STATUS_SCRIPT)"
	@echo "=== Potencia ==="
	@TENANT_PREFIX=potencia python3 -c "$(TENANT_STATUS_SCRIPT)"

weekly-check:
	@echo "--- Demo ---" && TENANT_PREFIX=demo python test_smoke.py
	@echo "--- Innovim ---" && TENANT_PREFIX=innovim python test_smoke.py

.PHONY: run-demo run-innovim eval smoke status weekly-check
