.PHONY: help artifacts mac-infer iphone-infer iphone-sim agreement-status package-check

help:
	@echo "Pillar-0 HeadCT mobile runner"
	@echo ""
	@echo "One-command paths:"
	@echo "  make mac-infer       Run Core ML int8 teacher inference on Mac and compare to PyTorch"
	@echo "  make iphone-infer    Build/install/launch the iOS simulator app and run full inference"
	@echo ""
	@echo "Setup / validation:"
	@echo "  make artifacts       Prepare compiled Core ML + raw iOS validation resources"
	@echo "  make agreement-status  Show background 100/100/100 int8 agreement progress"
	@echo "  make package-check   Syntax-check package scripts"

artifacts:
	./scripts/prepare_pillar0_mobile_artifacts.sh

mac-infer:
	./scripts/run_pillar0_mac_inference.sh

iphone-infer iphone-sim:
	./scripts/run_pillar0_iphone_simulator.sh

agreement-status:
	python3 scripts/pillar0_agreement_status.py

package-check:
	bash -n scripts/prepare_pillar0_mobile_artifacts.sh
	bash -n scripts/run_pillar0_mac_inference.sh
	bash -n scripts/run_pillar0_iphone_simulator.sh
	bash -n scripts/build_pillar0_mobile_artifacts.sh
	bash -n scripts/apply_rate_evals_mobile_patch.sh
	bash -n run_mac_inference.command
	bash -n run_iphone_inference.command
	python3 -m py_compile scripts/pillar0_agreement_status.py
