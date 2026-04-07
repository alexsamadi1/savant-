run-demo:
	TENANT_PREFIX=demo streamlit run app.py

run-innovim:
	TENANT_PREFIX=innovim streamlit run app.py

eval:
	TENANT_PREFIX=demo python run_eval.py

smoke:
	TENANT_PREFIX=demo python test_smoke.py

.PHONY: run-demo run-innovim eval smoke
