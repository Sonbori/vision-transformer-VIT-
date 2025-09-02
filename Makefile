.PHONY: help install clean all infer rect patchify pos train test lint

# Default target
help:
	@echo "ViT Experiments Package - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install    Install dependencies"
	@echo "  clean      Clean generated files"
	@echo ""
	@echo "Experiments:"
	@echo "  all        Run all experiments"
	@echo "  infer      01 - Inference demo with attention visualization"
	@echo "  rect       02 - Rectangular input experiments"
	@echo "  patchify   03 - Patch embedding equivalence verification"
	@echo "  pos        04 - Positional embedding analysis"
	@echo "  train      05 - CIFAR-100 fine-tuning with DeiT"
	@echo ""
	@echo "Quality:"
	@echo "  test       Run unit tests"
	@echo "  lint       Run code formatting and linting"
	@echo ""
	@echo "Output locations:"
	@echo "  - Figures: reports/figures/"
	@echo "  - Tables:  reports/tables/"
	@echo "  - Logs:    experiments/logs/"

# Environment setup
install:
	pip install -r requirements.txt
	@echo "Dependencies installed. You can also use: conda env create -f environment.yml"

# Clean generated files
clean:
	rm -rf reports/figures/* reports/tables/* experiments/logs/* experiments/checkpoints/*
	find notebooks -name "*.png" -delete
	find notebooks -name "*.csv" -delete
	find notebooks -name "*.json" -delete
	@echo "Cleaned all generated files"

# Run all experiments in order
all: infer patchify rect pos
	@echo "All core experiments completed! Training notebook requires manual execution."
	@echo "Run 'make train' separately for the fine-tuning experiment (takes 24-48h)"

# Individual experiments
infer:
	@echo "Running inference demo..."
	jupyter nbconvert --to notebook --execute notebooks/01_infer_vit.ipynb --output 01_infer_vit_executed.ipynb
	@echo "✓ Inference demo completed"

rect:
	@echo "Running rectangular input experiments..."
	jupyter nbconvert --to notebook --execute notebooks/02_rectangular_inputs.ipynb --output 02_rectangular_inputs_executed.ipynb
	@echo "✓ Rectangular input experiments completed"

patchify:
	@echo "Running patch embedding equivalence tests..."
	jupyter nbconvert --to notebook --execute notebooks/03_patchify_linear_vs_conv.ipynb --output 03_patchify_linear_vs_conv_executed.ipynb
	@echo "✓ Patch embedding tests completed"

pos:
	@echo "Running positional embedding analysis..."
	jupyter nbconvert --to notebook --execute notebooks/04_positional_embedding.ipynb --output 04_positional_embedding_executed.ipynb
	@echo "✓ Positional embedding analysis completed"

train:
	@echo "Running CIFAR-100 fine-tuning (this will take 24-48 hours)..."
	jupyter nbconvert --to notebook --execute notebooks/05_train_cifar100_deit.ipynb --output 05_train_cifar100_deit_executed.ipynb
	@echo "✓ Training experiment completed"

# Quality assurance
test:
	pytest scripts/test_*.py -v
	@echo "✓ All tests passed"

lint:
	black notebooks/ scripts/ --line-length 88
	isort notebooks/ scripts/
	@echo "✓ Code formatted"

# Batch script execution (alternative to notebooks)
batch-rect:
	python scripts/evaluate_rectangular.py --output reports/tables/rect_results.csv

batch-patchify:
	python scripts/patchify_bench.py --output reports/tables/patchify_results.csv

# Generate summary report
report:
	python scripts/generate_summary.py
	@echo "✓ Summary report generated in reports/"
