# Install dependencies
pip install -r requirements.txt

# Phase 1 - Causal Discovery to construct soft inductive prior
Run the causal discovery script to generate causal maps for your datasets:

python causal_discovery.py --dataset wine_white

# Phase 2 - Feature Engineering with CAFE

python cafe.py --file-name wine_white --episodes 30 --steps 15 --strategy-mode hybrid

#The results will be saved in /tmp directory and also logged in wandb repository if logged in.